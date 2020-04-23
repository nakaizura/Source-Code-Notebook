'''
Created on Apr 24, 2020
@author: nakaizura
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#pytorch官方函数，nn.Transformer


#---------------------搭建模型非常简单------------------------
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        #直接调用TransformerEncoder的部分，只需要实现decoder的部分
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout) #对input加入位置编码
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        #对连续输出的mask，将未来但是还未出现的单词给mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) #下三角矩阵对输出做mask
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))#填入负无穷大（未来词）或者0（允许出现的词）
        return mask

    def init_weights(self):
        #初始化权重
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src): #生成mask
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp) #嵌入之后这里多乘sqrt(self.ninp)是trick

        src = self.pos_encoder(src)#加入位置编码
        output = self.transformer_encoder(src, self.src_mask)#Transformer编码
        output = self.decoder(output)#线性层
        return output


class PositionalEncoding(nn.Module):
    """
    因为Transformer全是基于Attention的，所以不像RNN和CNN能得到顺序
    处理句子就需要加入相对/绝对位置，位置编码的维度和d_model一样方便相加
    位置编码有很多种，Transformer用的是正弦余弦相对位置编码（实际上这种编码不是很合理）
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #计算相对位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#按维度
        pe[:, 0::2] = torch.sin(position * div_term)#奇数位用sin
        pe[:, 1::2] = torch.cos(position * div_term)#偶数位用cos
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]#位置编码与词嵌入x直接相加
        return self.dropout(x)


#---------------------载入训练数据集和处理------------------------
#用torchtext的Wikitext-2 数据集
import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)#vocal对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    #将数据集排列为列，以截掉不够batch_size大小的剩余token
    data = TEXT.numericalize([data.examples[0].text])#总数
    nbatch = data.size(0) // bsz #应该有多少batch
    data = data.narrow(0, 0, nbatch * bsz) #剩下不够的就不管了
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)



#---------------------产生输入和目标序列------------------------
bptt = 35 #序列的长度
def get_batch(source, i):
    #输入和目标序列刚好错开一位（可以对应模型搭建的下三角mask）
    #input是：i love，那么目标序列是：love nakai，此时的bptt是2。以做到预测下一个词的目标。
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)#多加1就行
    return data, target



#---------------------设置超参和启用模型------------------------
ntokens = len(TEXT.vocab.stoi) # 词汇表大小
emsize = 200 # 嵌入维度
nhid = 200 # Transformer里面feedforward的维度大小
nlayers = 2 # TransformerEncoder堆叠的数目
nhead = 2 # 注意力多头的头数
dropout = 0.2 # dropout比率
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device) #设置好模型



#---------------------开始RUN模型------------------------
criterion = nn.CrossEntropyLoss() #交叉熵损失
lr = 5.0 # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) #在每个回合后调整学习率为gamma倍

import time
def train():
    model.train() #模型调到训练模式
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)#词汇总数量
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i) #得到批次
        optimizer.zero_grad() #梯度清零
        output = model(data) #得到模型输出
        loss = criterion(output.view(-1, ntokens), targets) #计算loss
        loss.backward() #反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) #梯度缩放防止爆炸
        optimizer.step()

        total_loss += loss.item() #记录总loss
        log_interval = 200 #打印输出间隔
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() #模型调到验证模式
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf") #记录最好的验证模型
epochs = 3 # 周期数
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


#---------------------最后测试模型------------------------
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
