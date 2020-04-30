'''
Created on Apr 30, 2020
@author: nakaizura
'''

import tensorflow as tf
import cPickle

#cPickle是python2.x，3.x改名为Pickcle。它可以序列化任何对象并保存，存网络参数很合适

#判别器。作用是对生成器得到的文档二分类（相关和不相关），进一步分类。
#生成器是多分类，判别器是二分类，两者对抗的结果感觉上是用GAN来找和正样本更相似的负样本（代替了随机采样）。
class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # 正则化参数
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):#参数初始化
            if self.param == None:#无设置就随机初始化
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]#嵌入权重和item偏置

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)#文档相关或者不相关的二分类label

        #嵌入user和item
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        #计算user和item的相似度，然后计算与label的交叉熵为判别器loss
        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)#梯度下降
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)#参数更新

        #生成器的reward是判别器相似度判别分数
        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias
        
        #重点要说这里！！！这个reward很paper中写的不一致！（不止这里，四个任务的reward的都不是原文的）
        #虽然reward这样设置不是不可以，因为Sigmoid之后处于0-1，小于0.5reward相减为负，这样避免reward都是正数。
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        #为了测试集算topk的rating分数, 是矩阵乘矩阵，得到所有得分，self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        #动态负采样算排名。公式一样喂入不同（这里所有的逻辑都在__init__里面....）
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):#保存参数
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
