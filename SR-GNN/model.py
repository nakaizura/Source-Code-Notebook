'''
Created on Jul 22, 2020
@author: nakaizura
'''
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2 #两个矩阵的拼接（入度和出度）
        self.gate_size = 3 * hidden_size #同时计算三个需要门的地方
        #三个门的参数
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        #因为A是入度和出度两个矩阵拼接得到的，所以要分0:A.shape[1]和A.shape[1]: 2 * A.shape[1]分别做linear变换
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)#然后再拼接
        gi = F.linear(inputs, self.w_ih, self.b_ih)#输入门
        gh = F.linear(hidden, self.w_hh, self.b_hh)#记忆门
        i_r, i_i, i_n = gi.chunk(3, 2)#沿2维度分3块，因为线性变换这三门是一起做的
        h_r, h_i, h_n = gh.chunk(3, 2)
        #三个gate
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self): #重置参数
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) #重新初始化

    def compute_scores(self, hidden, mask):
        #计算全局和局部，ht是最后一个item，作者认为它代表短期十分重要
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) #计算全局注意力权重
        # 不算被mask的部分的sum来代表全局的特征a
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid: #globe+local
            a = self.linear_transform(torch.cat([a, ht], 1))#将局部和全局s1和sg拼接
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0)) #再做一次特征变换
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable): #如果有GPU就变到GPU进行cuda变换，否则就不变
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable): #如果有GPU就进行cuda变换，否则就不变
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i) #载入数据得到input和邻接矩阵等等
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long()) #是否GPU
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A) #用gnn计算hidden
    get = lambda i: hidden[i][alias_inputs[i]] #得到所有item特征
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])#按session打包
    return targets, model.compute_scores(seq_hidden, mask)#再结合全局+局部计算最后的预测分数


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train() #模型调到训练模式
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) #生成训练batch
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad() #梯度清零
        targets, scores = forward(model, i, train_data) #使用模型计算分数
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1) #计算loss
        loss.backward()#反向传播
        model.optimizer.step()#梯度更新
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0: #定时打印
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval() #模型调到预测模式
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)#生成测试batch
    for i in slices:
        targets, scores = forward(model, i, test_data) #使用模型计算分数
        sub_scores = scores.topk(20)[1] #选取分数topk的item
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):#计算两个指标
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
