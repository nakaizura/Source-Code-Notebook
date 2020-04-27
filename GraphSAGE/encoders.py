'''
Created on Apr 27, 2020
@author: nakaizura
'''

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    GraphSage的聚合方式可改，这个py是使用"卷积"(GCN)的GraphSage去encode节点特征。
    下一个py是用mean方式聚合的，会调用这个函数完成功能。
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features #节点特征
        self.feat_dim = feature_dim #特征维度
        self.adj_lists = adj_lists #邻接矩阵
        self.aggregator = aggregator #聚合器
        self.num_sample = num_sample #采样的数目
        if base_model != None: #base模型
            self.base_model = base_model

        self.gcn = gcn #GCN聚合
        self.embed_dim = embed_dim #嵌入维度
        self.cuda = cuda #gpu
        self.aggregator.cuda = cuda
        #如果用gcn聚合，那么维度就直接是feat_dim。
        #如果不是gcn，那就仍然是需要mean邻居再与自己concat，维度是2倍。
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight) #Xavier初始化

    def forward(self, nodes):
        """
        为每个batch的node生成嵌入特征
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample) #按采样数采样节点后，把邻接矩阵输入模型
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            #如果不是gcn聚合，需要concat，维度是两倍
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats #如果是gcn就不变
        combined = F.relu(self.weight.mm(combined.t()))#最后乘权重再激活
        return combined
