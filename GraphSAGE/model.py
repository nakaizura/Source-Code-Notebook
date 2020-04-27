'''
Created on Apr 27, 2020
@author: nakaizura
'''

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator


"""
pytorch版本的Simple supervised GraphSAGE。
使用的数据集是Cora和Pubmed。
"""
class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc #嵌入的聚合方法
        self.xent = nn.CrossEntropyLoss() #交叉熵loss

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight) #Xavier初始化

    def forward(self, nodes):
        embeds = self.enc(nodes) #嵌入node特征
        scores = self.weight.mm(embeds) #预测分数
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())#计算分数和label的交叉熵

def load_cora():
    '''
    cora是2708个有引用关系的一堆论文，用监督的方式进行7分类（该论文的领域）。
    2708个节点，5429个边，1433维特征，7分类
    
    '''
    num_nodes = 2708 #节点数目
    num_feats = 1433 #每个节点的特征维度
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        #读节点
        for i,line in enumerate(fp):
            info = line.strip().split()#第1维是id
            feat_data[i,:] = map(float, info[1:-1])#中间的是特征
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)#最后一维是label
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)#集合
    with open("cora/cora.cites") as fp:
        #读边关系
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)#无向图，两个邻接矩阵都add
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1) #可复现随机种子
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()#载入数据集
    features = nn.Embedding(2708, 1433) #初试特征维度
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()

    agg1 = MeanAggregator(features, cuda=True)#mean聚合
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)#gcn聚合嵌入到128维
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False) #用gcn嵌入后的节点特征做mean聚合
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False) #agg2后再gcn聚合嵌入
    enc1.num_samples = 5 #一阶和二阶的邻居采样数
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)#cora数据集7分类
    #graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)#打乱数据集（新的数组）
    test = rand_indices[:1000] #划分数据集为3个部分
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256] #每次取前256个
        random.shuffle(train) #会在train数组本身打乱，所以每次256个会不同
        start_time = time.time()
        optimizer.zero_grad() #梯度清零
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))#loss
        loss.backward() #反向传播
        optimizer.step() #梯度更新
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.data[0]

    val_output = graphsage.forward(val) #验证集
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

def load_pubmed():
    #pubmed和cora一样，也是引文网络，由论文和他们的关系构成。
    ##pubmed有19717个节点，44338个边，500维特征，3分类
    #代码逻辑和cora也一样，不再一一注释。
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1 #不同点，第2维是label
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)#19717个节点，500维特征
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10 #一阶和二阶采样数不一样
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)#三分类
    #graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.data[0]

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

if __name__ == "__main__":
    run_cora() #默认run cora数据集
