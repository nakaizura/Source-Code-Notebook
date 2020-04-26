'''
Created on Apr 26, 2020
@author: nakaizura
'''

import numpy as np
import scipy.sparse as sp
import torch

#scipy.sparse库中提供了多种表示稀疏矩阵的格式，同时支持稀疏矩阵的加、减、乘、除和幂等。


def encode_onehot(labels):
    #标签one-hot
    classes = set(labels)
    #去重标签，cora数据集的维度0是id，1-1433是节点特征向量，1434是label
    #class为键，值是先创建一个对角为1的矩阵取其第i行（其实就是在对应的位置为1了）。
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """
    载入标准network数据集。
    cora是2707个有引用关系的一堆论文，利用GAT用半监督的方式进行7分类（该论文的领域）。
    """
    print('Loading {} dataset...'.format(dataset))

    #节点，#从文件中生成数组
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    #csr(压缩稀疏行矩阵)，即用行索引、列索引和值压缩。
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1]) #one-hot label

    #----------构建大图Graph----------------
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)} #从样本id到样本索引
    #边信息。即cite引用关系。
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    #邻接矩阵。但是sp稀疏矩阵的表示，所以非对称。
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    #构建对称的邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)#非对称归一化特征
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))#对称归一化邻接矩阵


    #划分数据集，按索引将数据集划分为训练集，验证集和测试集。
    idx_train = range(140) #范围是[0,140),前面是True，后面是False
    idx_val = range(200, 500) #同上
    idx_test = range(500, 1500) #同上

    #转换为pytorch下的tensor
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """对称归一化，Row-normalize sparse matrix"""
    #D^(-1/2) * A * D^(-1/2)
    rowsum = np.array(mx.sum(1)) #adj.sum(1)计算每行元素和，得到度矩阵再D^-0.5
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)#得到归一化的A_norm


def normalize_features(mx):
    """非对称归一化，Row-normalize sparse matrix"""
    #D^(-1) * A
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    #准确率评估函数
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
