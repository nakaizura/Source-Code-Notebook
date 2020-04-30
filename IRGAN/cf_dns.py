'''
Created on Apr 30, 2020
@author: nakaizura
'''

import tensorflow as tf
from dis_model_dns import DIS
import cPickle
import numpy as np
import multiprocessing

cores = multiprocessing.cpu_count()

#这个是带dns的代码

#########################################################################################
#超参设置
#########################################################################################
EMB_DIM = 5
USER_NUM = 943
ITEM_NUM = 1683
DNS_K = 5
all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'
DIS_MODEL_FILE = workdir + "model_dns.pkl"
#########################################################################################
#载入数据
#########################################################################################
user_pos_train = {} #训练集
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99: #电影评分为4认为是正例
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {} #测试集
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99: #电影评分为4认为是正例
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = user_pos_train.keys()
all_users.sort()#按user id排序


#动态生成负采样，用gan来“动态”。
def generate_dns(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]#得到正例
        #这里会run模型里面dns_rating！
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating)
        neg = []
        candidates = list(all_items - set(pos))#负例候选

        for _ in range(len(pos)):#随机采样负例
            choice = np.random.choice(candidates, DNS_K)
            choice_score = all_rating[choice]
            #得到分数最高的item，即负采样的时候选择了判别器最难分辨的item
            neg.append(choice[np.argmax(choice_score)])

        for i in range(len(pos)):#存入正例负例对
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))

#计算DCG
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

#计算NDCG
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

#对一个用户进行测试
def simple_test_one_user(x):
    rating = x[0]#这个已经是预测得到的结果，接下来算几个值就ok
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]#对分数进行排序

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    #准确率@3，5，10
    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])

    #NDCG@3，5，10
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)#多线程
    batch_size = 128
    test_users = user_pos_test.keys()
    test_user_num = len(test_users)#测试用户总数目
    index = 0
    while True:
        if index >= test_user_num:#超过了就break
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        #用模型得到预测值
        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        #对每个用户都进行计算
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


#这个是随机采样的函数
def generate_uniform(filename):
    data = []
    print 'uniform negative sampling...'
    for u in user_pos_train:
        pos = user_pos_train[u]#正例
        candidates = list(all_items - set(pos))
        neg = np.random.choice(candidates, len(pos))#随机从候选里面采
        pos = np.array(pos)

        for i in range(len(pos)):#存入正例负例对
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def main():
    np.random.seed(70)#可复现随机种子
    param = None
    #实例化判别器
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.05)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #gpu
    sess = tf.Session(config=config) #启动图
    sess.run(tf.global_variables_initializer()) #初始化

    dis_log = open(workdir + 'dis_log_dns.txt', 'w')
    print "dis ", simple_test(sess, discriminator)
    best_p5 = 0.

    # generate_uniform(DIS_TRAIN_FILE) # Uniformly sample negative examples

    for epoch in range(80):
        generate_dns(sess, discriminator, DIS_TRAIN_FILE)  #动态负采样，dynamic negative sample
        with open(DIS_TRAIN_FILE)as fin:
            for line in fin:
                line = line.split()
                u = int(line[0])
                i = int(line[1])#正例
                j = int(line[2])#负例
                _ = sess.run(discriminator.d_updates,
                             feed_dict={discriminator.u: [u], discriminator.pos: [i],
                                        discriminator.neg: [j]}) #更新参数

        result = simple_test(sess, discriminator)#测试结果
        print "epoch ", epoch, "dis: ", result
        if result[1] > best_p5:#记录最佳
            best_p5 = result[1]
            discriminator.save_model(sess, DIS_MODEL_FILE)
            print "best P@5: ", best_p5

        buf = '\t'.join([str(x) for x in result])#记录
        dis_log.write(str(epoch) + '\t' + buf + '\n')
        dis_log.flush()

    dis_log.close()


if __name__ == '__main__':
    main()
