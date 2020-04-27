'''
Created on Apr 27, 2020
@author: nakaizura
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Mean 聚合邻居的aggregator。
"""

class MeanAggregator(nn.Module):
    """
    GraphSage的聚合方式可改，这个py是使用MEAN的GraphSage去encode节点特征。
    这里的节点特征有可能是gcn已经嵌入过的。
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        为某个具体的图初始化聚合器aggregator
        features -- 是对node id的嵌入特征
        cuda -- 是否gpu加速
        gcn --- 用GraphSAGE-style聚合，还是用加了自环的GCN-style聚合
        """

        super(MeanAggregator, self).__init__()

        self.features = features #节点特征
        self.cuda = cuda #gpu
        self.gcn = gcn #gcn聚合
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- 一个batch的所有node列表
        to_neighs --- node的邻居节点集合
        num_sample --- 对邻居节点的采样数量
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None: #如果设置了采样数
            _sample = random.sample #按采样数在邻居集合中随机采样
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:#不然就是所有的邻居
            samp_neighs = to_neighs

        if self.gcn:#如果用gcn聚合，要加入自环
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs)) #去掉重复的采样
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        #一个(len(samp_neighs), len(unique_nodes)维度的mask矩阵
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))) 
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats
