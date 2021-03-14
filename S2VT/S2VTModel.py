'''
Created on Mar 14, 2021
@author: nakaizura
'''
import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.autograd import Variable

#按照S2VT的结构是2层LSTM，第一层是编码帧序列
#第二层是接受第一层的隐层状态+与0填充之后再编码（很多对应位置没有值，直接pad为0来填充）

class S2VTModel(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, sos_id=1, eos_id=0,
                 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2):
        super(S2VTModel, self).__init__()
        #可选择是LSTM或GRU两种
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)

        self.dim_vid = dim_vid #视频维度
        self.dim_output = vocab_size #词表大小
        self.dim_hidden = dim_hidden #隐层维度
        self.dim_word = dim_word #词维度
        self.max_length = max_len #最大长度
        self.sos_id = sos_id #开始符
        self.eos_id = eos_id #结束符
        self.embedding = nn.Embedding(self.dim_output, self.dim_word) #编码词表的词，因为是one-hot所以直接是词表大小

        self.out = nn.Linear(self.dim_hidden, self.dim_output) #用于输出的fc

    def forward(self, vid_feats, target_variable=None,
                mode='train', opt={}):
        batch_size, n_frames, _ = vid_feats.shape #视觉特征维度
        #两种pad填充，frame和word
        padding_words = Variable(vid_feats.data.new(batch_size, n_frames, self.dim_word)).zero_()
        padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.dim_vid)).zero_()
        #两种开始状态，frame和word都为none
        state1 = None
        state2 = None
        #self.rnn1.flatten_parameters()
        #self.rnn2.flatten_parameters()
        output1, state1 = self.rnn1(vid_feats, state1) #第一层LSTM
        input2 = torch.cat((output1, padding_words), dim=2) #凭借输出的隐层和0填充的pad
        output2, state2 = self.rnn2(input2, state2) #然后输到第二层

        seq_probs = []
        seq_preds = []
        if mode == 'train': #训练模式
            for i in range(self.max_length - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i]) #嵌入当前词
                #重置参数的数据指针，使内存更contiguous(连续性)，利用率高
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                #逐词的输出都要通过pad，过两层LSTM得到结果
                output1, state1 = self.rnn1(padding_frames, state1) 
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1)) #预测概率
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)

        else: #测试模式
            current_words = self.embedding(
                Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())#嵌入当前词
            for i in range(self.max_length - 1):
                 #重置参数的数据指针，使内存更contiguous(连续性)，利用率高
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                #逐词的输出都要通过pad，过两层LSTM得到结果
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds) #得到结果词
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds
