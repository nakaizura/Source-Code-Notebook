'''
Created on Apr 20, 2020
@author: nakaizura
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


#Harvardnlp逐行实现的形式呈现了论文的“注释”版本。



#--------------模型部分---------
#代码结构上是从大到小，从泛到细。
#先给出Encoder-Decoder，再分别介绍两者，再详解多头注意力，最后是一些细节。
#--------------首先是整体的Encoder-Decoder---------
class EncoderDecoder(nn.Module):
    """
    和大多数机器翻译，神经序列转换一样，使用标准的Encoder-Decoder框架。
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        整个流程就是这一句话...
        先得到带mask的当前输出词和已经输出过的词的嵌入结果，然后encode，再在外面套一个decode
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        #encode用的是encoder，需要完整输入src和mask（句子不够可能需要padding）
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        #encoder需要目标词tgt，已经产生的memory和目前输入的词src_mask（padding和未来两种mask）
        #具体的mask处理在class Batch中。
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "这是在EncoderDecoder结束后，最后需要linear + softmax来投影预测一个词"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


#--------------分别堆叠的Encoder和Decoder---------
def clones(module, N):
    "深拷贝N个相同的模型（identical layers）然后堆叠在一起"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#先是Encoder
class Encoder(nn.Module):
    "编码器由N = 6个相同的结构堆叠组成"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """
    模型图里面的Add&Norm，Norm指的就是layernorm层做Normalization
    按照层归一样本维度，同时比普通的BN多两个参数进行可学习的缩放移动
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
         #计算样本的均值和方差归一化，然后用a、b缩放。
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    模型图里面的Add&Norm，Add是指这种跳跃连接（residual connection）
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "先sublayer由子层实现功能再norm，residual connection就再加上x保持特征不变，维度相等都是512维"
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "完整的Encoder由多头self-attn+Add&Norm和逐位置(position-wise)的全连接层(feed forward)+Add&Norm组成"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #克隆两次Add&Norm，sublayer[0]是注意力后面的，sublayer[1]是全连接后面的
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "模型图左边的Encoder结构就搭好了"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))#在注意力后面加
        return self.sublayer[1](x, self.feed_forward)#在全连接后面加，然后返回结果


#然后是Decoder
class Decoder(nn.Module):
    "解码器也是由N = 6个相同的结构堆叠组成"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    完整的Dncoder由masked self-attn, src-attn, 和全连接feed forward组成
    这个多出来的src-attn就是在编码解码之间搭桥的cross-attn。Q来自上一个解码器，而K和V是编码器的输出
    相当于此时带masked（还未输出的句子是未知的）的query去查询Eecoder的相关信息
    这允许解码器中的每个位置都参与输入序列中的所有位置
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)#多了一个cross所以需要三次的Add&Norm
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "搭好模型图右边的结构"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))#带masked的注意力后面加Add&Norm
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))#在cross注意力后面加Add&Norm
        return self.sublayer[2](x, self.feed_forward)#在全连接后面加Add&Norm

def subsequent_mask(size):
    "mask句子后面还没有出现的位置"
    attn_shape = (1, size, size)
    #不含对角的上三角矩阵，每行向后偏移一个位置信息以逐词预测
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
     


#--------------注意力部分---------
def attention(query, key, value, mask=None, dropout=None):
    """
    计算'Scaled Dot Product Attention'
    scale是针对Q和K的维度，1使点积和与维度无关 2矩阵乘法所以点积比mlp更快 3softmax对低值更敏感
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) #计算Q对所有K的相似度
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1) #然后softmax得到分数
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn #分数分配得到结果


class MultiHeadedAttention(nn.Module):
    """
    多头注意力，多组映射子空间，扩展模型对不同位置的关注能力，提高特征表达
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        #d_v==d_k，而且用8头注意力之后，维度是512/8=64。
        self.d_k = d_model // h
        self.h = h
        #4个Linear层，三个投影VKQ，一个用于最后多头concat了再投影
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            #统一所有h个头都是一样的mask
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) V、K、Q要先Liner投影一下，也是多头投影，维度是h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2)Scaled Dot Product注意力层
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3)将所有头的结果拼接concat之后再投影
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)#得到最后一层投影层的结果


#--------------其他处理细节---------
class PositionwiseFeedForward(nn.Module):
    """
    这里是全连接FFN层，即fully connected feed-forward network
    两层+ReLU的NN，即公式是FFN(x)=max(0,xW1+b1)W2+b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)#d_model还是512维
        self.w_2 = nn.Linear(d_ff, d_model)#中间的d_ff是2048，最后还原到512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    "学习对输入token词的嵌入"
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)#这里多乘sqrt(self.d_model)是trick


class PositionalEncoding(nn.Module):
    """
    因为Transformer全是基于Attention的，所以不像RNN和CNN能得到顺序
    处理句子就需要加入相对/绝对位置，位置编码的维度和d_model一样方便相加
    位置编码有很多种，Transformer用的是正弦余弦相对位置编码（实际上这种编码不是很合理）
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        #计算相对位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))#按维度
        pe[:, 0::2] = torch.sin(position * div_term)#奇数位用sin
        pe[:, 1::2] = torch.cos(position * div_term)#偶数位用cos
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)#位置编码与词嵌入直接相加
        return self.dropout(x)


#--------------完整模型---------
#编码解码套数N=6，总特征维度d_model=512, 隐层维度d_ff=2048, 头数h=8, dropout=0.1
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    #每一套的必要细节
    attn = MultiHeadedAttention(h, d_model)#多头
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)#FFN
    position = PositionalEncoding(d_model, dropout)#位置编码
    
    #整个模型是EncoderDecoder，分别有N套，然后各种细节填进去。
    #最后EncoderDecoder后，跟一个Generator（Linear+softmax）得到输出的词
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    #初始化参数用xavier
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

#一个小模型的例子
tmp_model = make_model(10, 10, 2)#10是词汇表大小



#--------------开始训练---------
class Batch:
    """
    如何mask处理，分Encoder和Decoder两种功能
    Encoder里面，mask让batch里面句子很短只能padding的部分不参与计算
    Dncoder里面，mask让未来需要预测的词不提前出现
    """
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        #对于src很简单就直接把pad的地方mask掉
        self.src_mask = (src != pad).unsqueeze(-2)
        #对于trg就要mask两种，pad和未来词
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "把还没生成的未来词和长度不足的pad都要mask"
        #不等于0就mask，即mask掉未来的词，可以对应上面生成上三角矩阵是用的是全1
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "标准的训练函数和日志函数"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)#得到模型输出
        loss = loss_compute(out, batch.trg_y, batch.ntokens)#计算loss
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:#定期打印日志
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "计算词和填充tokens + padding的总数"
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


#--------------硬件条件和用时---------
#8 NVIDIA P100 GPUs....
#base版12小时，big版3.5天....



#--------------优化器和参数---------
class NoamOpt:
    """
    非常重要的是模型一定要先热身。
    先用一个固定的warmup_steps进行学习率的线性增长（热身）
    到warmup_steps=4000之后会随着step_num的增长而逐渐减小
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "更新参数和速率"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "把热身结果应用在初试学习率factor上"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    #adam下是0.9，0.98，1e-9
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


#--------------其他细节---------
class LabelSmoothing(nn.Module):
    "label smoothing。也是一种soft方法，把绝对的one-hor的0，1标签用个置信度软化"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)#KL散度
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing#先将one-hor的1减去要平滑的部分
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))#然后其他地方给0平分
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:#mask的地方都得是0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        #计算KL散度
        return self.criterion(x, Variable(true_dist, requires_grad=False))

#标签平滑的例子
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))










#--------------A First Example---------
#一个简单的copy-task，即输入什么就输出什么
def data_gen(V, batch, nbatches):
    "先随机生成src-tgt，因为是copy-task，所以两者是一样的就好"
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))#V是词汇表
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "计算损失函数并反向传播"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
        
#训练这个copy-task
V = 11#特别小的词汇表
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))#提前热身

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))

#模型评估
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #一个一个预测
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)#预测下一个词
        next_word = next_word.data[0]
        #要更新一下当前已经输出的词和mask
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )#输入一堆数字，看是否打印出相同的结果
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
