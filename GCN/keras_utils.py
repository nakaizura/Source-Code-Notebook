'''
Created on Apr 25, 2020
@author: nakaizura
'''

from __future__ import print_function
#future处理新功能版本不兼容问题，加上这句话，所有的print函数将是3.x的模式（即便环境是2.x）

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

#scipy.sparse库中提供了多种表示稀疏矩阵的格式，同时支持稀疏矩阵的加、减、乘、除和幂等。
#稀疏矩阵中查找特征值/特征向量的函数

def encode_onehot(labels):
    #标签one-hot
    classes = set(labels) #去重标签，维度为1433
    #class为键，值是先创建一个对角为1的矩阵取其第i行（其实就是在对应的位置为1了）。
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """
    载入标准network数据集。
    cora是2707个有引用关系的一堆论文，利用GCN用半监督的方式进行7分类。
    """
    print('Loading {} dataset...'.format(dataset))

    #节点。
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))#从文件中生成数组
    #csr(压缩稀疏行矩阵)，即用行索引、列索引和值压缩。
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1]) #one-hot label

    #----------构建Graph----------------
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}#从样本id到样本索引

    #边信息。即cite引用关系。
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    #邻接矩阵。但是sp稀疏矩阵的表示，所以非对称。
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    #构建对称的邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    #归一化邻接矩阵
    if symmetric:#如果是对称，那就直接归一化：D^(-1/2) * A * D^(-1/2)
        #adj.sum(1)计算每行元素和，得到度矩阵再D^-0.5
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr() #左右各一次
    else:#如果不对称，就直接D^(-1) * A
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    #加入自环I
    adj = adj + sp.eye(adj.shape[0])#用eye实现自环
    adj = normalize_adj(adj, symmetric)#归一化
    return adj


def sample_mask(idx, l):
    #样本掩码。因为任务是做半监督的文本分类，需要mask
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    #划分数据集，按索引将数据集划分为训练集，验证集和测试集。
    idx_train = range(140) #范围是[0,140),前面是True，后面是False
    idx_val = range(200, 500) #同上
    idx_test = range(500, 1500) #同上
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])#加入样本掩码
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    #因为是做分类，用交叉熵损失函数loss
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    #准确率评估函数acc
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    #评估预测
    split_loss = list()#loss
    split_acc = list()#acc

    for y_split, idx_split in zip(labels, indices):#对每个样本都计算loss和acc
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    #归一化拉普拉斯矩阵
    adj_normalized = normalize_adj(adj, symmetric) #先归一化邻接矩阵
    laplacian = sp.eye(adj.shape[0]) - adj_normalized #再用I减，得到拉普拉斯矩阵
    return laplacian


def rescale_laplacian(laplacian):
    #对拉普拉斯矩阵进行重新放缩的调整以更好的收敛
    try:#除拉普拉斯的特征值
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:#不收敛特征值求不出就直接设为2，即不放缩。
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    #相当于多除一个largest_eigval，sl= 2 / largest_eigval * L - I
    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    #计算k阶切比雪夫近似
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr()) #T_0=I
    T_k.append(X) #T_1=L

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        #切比雪夫递推式，计算从T2到Tk
        X_ = sp.csr_matrix(X, copy=True)#要先压缩成csr
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
    #返回切比雪夫多项式列表
    return T_k


def sparse_to_tuple(sparse_mx):
    #把稀疏矩阵转化成tuple
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()#coo用三个矩阵分别存行 列 值。
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
