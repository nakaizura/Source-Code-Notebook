'''
Created on Apr 30, 2020
@author: nakaizura
'''

import tensorflow as tf
import cPickle

#cPickle是python2.x，3.x改名为Pickcle。它可以序列化任何对象并保存，存网络参数很合适

#生成器实际上是softmax函数，对文档做多分类得到“生成”的候选。
#生成器是多分类，判别器是二分类，两者对抗的结果感觉上是用GAN来找和正样本更相似的负样本（代替了随机采样）。
class GEN():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # 正则化参数
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.g_params = []

        with tf.variable_scope('generator'):#参数初始化
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
                self.item_bias = tf.Variable(param[2])

            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]#嵌入权重和item偏置

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)#生成器的reward

        #嵌入user和item
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        #计算user和item的相似度，然后softmax得到相关或者不相关的概率
        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)

        #相似度*reward+正则就是生成器的loss，reward来自判别器
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))

        g_opt = tf.train.GradientDescentOptimizer(self.learning_rate)#梯度下降
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)#参数更新

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

    def save_model(self, sess, filename):#保存模型
        param = sess.run(self.g_params)
        cPickle.dump(param, open(filename, 'w'))
