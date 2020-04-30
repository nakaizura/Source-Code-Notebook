'''
Created on Apr 30, 2020
@author: nakaizura
'''

import tensorflow as tf
import cPickle

#cPickle是python2.x，3.x改名为Pickcle。它可以序列化任何对象并保存，存网络参数很合适


#带动态负采样的判别器。
#GAN思想的另一方面就是生成负采样，在cf_dns文件中，这里是利用正负例构建判别器。
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
        self.pos = tf.placeholder(tf.int32)#正例
        self.neg = tf.placeholder(tf.int32)#负例

        ##嵌入user和item（正负例）
        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.pos_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos)
        self.pos_bias = tf.gather(self.item_bias, self.pos)
        self.neg_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg)
        self.neg_bias = tf.gather(self.item_bias, self.neg)

        #计算pos和neg差值和user的相似度，然后log+正则得到loss
        self.pre_logits = tf.sigmoid(
            tf.reduce_sum(tf.multiply(self.u_embedding, self.pos_embedding - self.neg_embedding),
                          1) + self.pos_bias - self.neg_bias)
        self.pre_loss = -tf.reduce_mean(tf.log(self.pre_logits)) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) +
            tf.nn.l2_loss(self.pos_embedding) +
            tf.nn.l2_loss(self.pos_bias) +
            tf.nn.l2_loss(self.neg_embedding) +
            tf.nn.l2_loss(self.neg_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)#梯度下降
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)#参数更新

        #为了测试集算topk的rating分数, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        #动态负采样算排名。公式一样喂入不同（这里所有的逻辑都在__init__里面....）
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):#保存参数
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
