'''
Created on Mar 14, 2021
@author: nakaizura
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    注意力机制用到从decoder出来的特征上，即为了更好的学习上下文特征
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        #self._init_hidden()

    def _init_hidden(self):
        #xavier初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        输入的参数:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        返回的结果:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size() #得到Encoder出来的中间层维度
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)#多增加seq_len的维度以便后面concat
        #拼接Encoder和hidden向量并展平
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        #两层FC
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1) #返回上下文
        return context
