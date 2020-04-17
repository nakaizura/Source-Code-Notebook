'''
Created on Apr 17, 2020
@author: nakaizura
'''

import math
import heapq #优先队列堆排最后的topK结果
import multiprocessing
import numpy as np
from time import time
#numba是一个用于编译Python数组和数值计算函数的编译器
#from numba import jit, autojit

#全局变量。main只调用evaluate_model，其他的函数都在evaluate_model中被调用。
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    使用多线程处理得到 (Hit_Ratio, NDCG) 的评价指标分数。
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): #多线程
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))#map函数传入参数列表（所有的ID）给eval_one_rating
        pool.close()
        pool.join()#进程终结后要调用wait（join等同于wait，避免其成为僵尸进程）
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    #单线程
    #xrange生成器，生成一个ID取出一个，不是一次生成整个列表，能优化内存。
    for idx in xrange(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)#对每个用户计算评价指标
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    '''
    对单个用户用model预测，再topk计算评价指标。由于重要参数全局之后，只需要传入idx，同样是对性能的优化
    '''
    #根据用户id，得到相应的item
    rating = _testRatings[idx] #正例对[user, item]
    items = _testNegatives[idx] #负例
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)#把正例也放到负例列表中，方便计算。
    #利用model预测分数
    map_item_score = {}#存储该user对所有item的分数
    users = np.full(len(items), u, dtype = 'int32')#填充user矩阵，即[u,u,u,u,u,u...]，一一对应user-item对放入model得到分数
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]#填充预测分数
    items.pop()#pop掉正例（感觉用不上...topk也是对map_item_score进行的）
    
    #检索topk
    #堆排速度快，能快速按照score得到topk的ID。
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    #然后计算两个评估指标
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    #HR击中率，如果topk中有正例ID即认为正确
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    #NDCG归一化折损累计增益
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
