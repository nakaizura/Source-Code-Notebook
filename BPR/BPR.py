'''
Created on Apr 19, 2020
@author: nakaizura
'''

import numpy
import tensorflow as tf
import os
import random
from collections import defaultdict


#整个代码思路是载入数据集，构建数据格式，生成batch，然后构建模型，开始训练。

#----------载入数据集-------------
def load_data(data_path):
    '''
    bpr针对三元组排序而不是算得分score，所以具体的ratings值不需要
    '''
    user_ratings = defaultdict(set)#有默认值的字典，键key是userID，值value是user有交互的item
    max_u_id = -1#记录user和item的数量
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split("::")#只需要user和item其他信息不需要
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)#把itemID放到对应user的value中
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    print ("max_u_id:",max_u_id)
    print ("max_i_id:",max_i_id)
    return max_u_id, max_i_id, user_ratings
    
#使用movielens 1M数据集
data_path = os.path.join('./movielens', 'ratings.dat')
user_count, item_count, user_ratings = load_data(data_path)

def generate_test(user_ratings):
    '''
    测试集的构建是对每个user，都从它有交互的item随机选择一个item。
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]#随机采样一个item
    return user_test


#得到测试集
user_ratings_test = generate_test(user_ratings)


#----------生成batch-------------
def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512):
    '''
    均匀采样成：(user, item_rated, item_not_rated)的三元组格式。
    '''
    t = []
    for b in xrange(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]#随机采样一个user
        i = random.sample(user_ratings[u], 1)[0]#从user的交互列表中采样一个item
        while i == user_ratings_test[u]:#如果得到的item在测试集中出现过，就重新再采样
            i = random.sample(user_ratings[u], 1)[0]
        
        j = random.randint(1, item_count)#再从整个item集合中采样一个没有交互过的item
        while j in user_ratings[u]:#如果采样到有交互的就再采样一次
            j = random.randint(1, item_count)
        t.append([u, i, j])#得到三元组
    return numpy.asarray(t)

def generate_test_batch(user_ratings, user_ratings_test, item_count):
    '''
    测试集batch是为user除了一个有交互的item外，其他所有没有交互过的item三元组(u,i,j) ，方便计算AUC。
    '''
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]#测试集中有交互的item
        for j in xrange(1, item_count+1):
            if not (j in user_ratings[u]):#所有没有交互的item
                t.append([u, i, j])
        yield numpy.asarray(t)


#----------构建BPR模型-------------
def bpr_mf(user_count, item_count, hidden_dim):
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    with tf.device("/cpu:0"):
        user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], 
                            initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], 
                                initializer=tf.random_normal_initializer(0, 0.1))
        item_b = tf.get_variable("item_b", [item_count+1, 1], 
                                initializer=tf.constant_initializer(0.0))

        #对三元组元素进行嵌入即先做矩阵分解MF
        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        i_b = tf.nn.embedding_lookup(item_b, i)#偏置项
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)
        j_b = tf.nn.embedding_lookup(item_b, j)
    
    #按照公式，第一部分是i和j的差值计算: u_i > u_j
    x = i_b - j_b + tf.reduce_sum(tf.mul(u_emb, (i_emb - j_emb)), 1, keep_dims=True)
    
    #对每个user计算AUC，即x>0把有交互的正例排序更前的mean
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    #第二部分完美的正则项
    l2_norm = tf.add_n([
            tf.reduce_sum(tf.mul(u_emb, u_emb)), 
            tf.reduce_sum(tf.mul(i_emb, i_emb)),
            tf.reduce_sum(tf.mul(j_emb, j_emb))
        ])

    #整个loss
    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))#Sigmoid再log最后求和

    #梯度下降
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
    return u, i, j, mf_auc, bprloss, train_op

#----------开始训练-------------
with tf.Graph().as_default(), tf.Session() as session:#启动tf计算图
    u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, 20)#20是嵌入空间大小
    session.run(tf.initialize_all_variables())
    for epoch in range(1, 11):
        _batch_bprloss = 0#每个批次的loss
        for k in range(1, 5000): # 从训练集均匀采样
            uij = generate_train_batch(user_ratings, user_ratings_test, item_count)

            _bprloss, _ = session.run([bprloss, train_op], 
                                feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2]})
            _batch_bprloss += _bprloss
        
        print ("epoch: ", epoch)
        print ("bpr_loss: ", _batch_bprloss / k)

        user_count = 0
        _auc_sum = 0.0

        #测试结果的AUC平均
        for t_uij in generate_test_batch(user_ratings, user_ratings_test, item_count):

            _auc, _test_bprloss = session.run([mf_auc, bprloss],
                                    feed_dict={u:t_uij[:,0], i:t_uij[:,1], j:t_uij[:,2]}
                                )
            user_count += 1
            _auc_sum += _auc
        print ("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum/user_count)
        print ()
