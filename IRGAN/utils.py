'''
Created on Apr 30, 2020
@author: nakaizura
'''

import linecache
import numpy as np

#linecache读取大文件。其将文件读到内存缓存而不是每次从硬盘，提高效率。
#这个py主要是得到数据+各种度量方式（precision，NDCG等）

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


#从训练集得到批次数据
def get_batch_data(file, index, size):  # 1,5->1,2,3,4,5
    user = []
    item = []
    label = []
    for i in range(index, index + size):#按批次大小
        line = linecache.getline(file, i)
        line = line.strip()
        line = line.split()
        user.append(int(line[0]))#append两次，为了构建正例负例对
        user.append(int(line[0]))
        item.append(int(line[1]))#正例
        item.append(int(line[2]))#负例
        label.append(1.)#正例
        label.append(0.)#负例
    return user, item, label

#topk准确率
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).（相关或者不相关的二分类）
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]#得到topk再mean
    return np.mean(r)

#topk的平均准确率
def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).（相关或者不相关的二分类）
    Returns:
        Average precision
    """
    r = np.asarray(r)
    #计算所有topk（从1到所有）的结果mean
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

#平均准确率
def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).（相关或者不相关的二分类）
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


#计算topk的DCG，累积折损增益
def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

#计算NDCG，归一化的DCG
def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)#得到最大DCG
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max #归一化

#计算topk的recall结果
def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num #正例/所有正例

#F1
def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.
