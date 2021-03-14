'''
Created on Mar 14, 2021
@author: nakaizura
'''
import torch.nn as nn

#编码器负责表示输入的视频特征

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid #视觉维度
        self.dim_hidden = dim_hidden #RNN隐层维度
        self.input_dropout_p = input_dropout_p #输出序列的dropout
        self.rnn_dropout_p = rnn_dropout_p #隐层之后的dropout
        self.n_layers = n_layers #rnn层数
        self.bidirectional = bidirectional #是否双向
        self.rnn_cell = rnn_cell #rnn种类，有LSTM和GRU可选

        self.vid2hid = nn.Linear(dim_vid, dim_hidden) #从视觉到隐层
        self.input_dropout = nn.Dropout(input_dropout_p) #dropout

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))#embedding
        vid_feats = self.input_dropout(vid_feats) #dropout
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)#维度变换
        self.rnn.flatten_parameters() #优化内存
        output, hidden = self.rnn(vid_feats) #输到rnn，得到中间结果
        return output, hidden
