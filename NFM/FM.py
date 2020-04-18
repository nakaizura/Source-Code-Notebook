'''
Created on Apr 17, 2020
@author: nakaizura
'''

import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

#################### Arguments ####################
def parse_args():
    '''
    python自带的参数解析包argparse，方便读取命令行参数。
    参数主要有：数据集路径、数据集名称、是否预训练、批次大小、隐层大小、正则化系数、dropout比率、学习率、优化器、打印间隔、是否BN批次正则化。
    '''
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.5, 
                    help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')

    return parser.parse_args()

class FM(BaseEstimator, TransformerMixin):
     '''
    构建FM模型。
    '''
    def __init__(self, features_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size, learning_rate, lamda_bilinear, keep,
                 optimizer_type, batch_norm, verbose, random_seed=2016):
        #参数传入FM模型
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.loss_type = loss_type
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        #存储每个epoch的性能rmse结果
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        #初始化tensorflow graph的所有变量
        self._init_graph()

    def _init_graph(self):
        '''
        初始化模型构建图流程
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            #设置图的随机种子
            tf.set_random_seed(self.random_seed)
            #输入数据。主要是features和labels。None是批次数目，features_M是数据集有的特征one-hot维度
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            #初始化权重
            self.weights = self._initialize_weights()

            #FM的模型部分。Bi-Interaction Layer，计算FM中的二次项的过程。
            # _________ sum_square part _____________
            #即求公式中的(\sum x_iv_i)^2
            #先对嵌入后的特征（嵌入即起到了v的作用）求和
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1) # None * K，K是嵌入维度。
            #对每个元素求平方，得到element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            #即求公式中的\sum (x_iv_i)^2
            #先对嵌入后的特征求平方，再求和
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            #对Bi-Interaction Layer的两个部分相减(sub)再0.5得到完整的二次项结果
            self.FM = 0.5 * tf.sub(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            
            #需要BN。BN的目的是为了缓解协方差偏移(covariance shift)：由于参数的更新隐藏层的输入分布不断变化，模型参数反而需要去学习这些变化，使收敛速度变慢。
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep) #对FM进行dropout

            # _________out _________
            #得到预测的输出，除了二次项，还需要计算偏置项w0和一次项\sum w_ix_i
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            #FM的输出最后由三部分组成二次，一次，偏置
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

            #计算损失函数。支持分类也支持回归，ranking等，损失函数可多样。
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(tf.sub(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  #对权重多一个L2正则化
                else:
                    self.loss = tf.nn.l2_loss(tf.sub(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out)#分类时先Sigmoid。
                if self.lambda_bilinear > 0:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07, scope=None) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07, scope=None)

            #多种梯度下降优化器
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            #初始化图
            self.saver = tf.train.Saver()#Saver管理参数便于保存和读取
            init = tf.global_variables_initializer()#初始化模型参数，即run了所有global Variable的assign op。
            self.sess = tf.Session()#会话控制和输出
            self.sess.run(init)#然后运行图

            #计算整个模型的参数数量，这主要是为了证明NFM比其他并行神经网络拥有更少的参数量。
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() #每个参数变量的维度大小
                variable_parameters = 1
                for dim in shape:#所有维度的数量
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print "#params: %d" %total_parameters 

    def _initialize_weights(self):
        '''
        初始化权重。分是否有预训练过的模型两种初始化方法。
        '''
        all_weights = dict()
        if self.pretrain_flag > 0:
            #如果有预训练过的模型，就载入参数
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()#获取当前默认计算图。
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')#通过张量名称获取张量
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)#恢复参数
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:
            #如果没有，就随机初始化所有参数
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K，输入特征的数目features_M到嵌入特征的数目K
            all_weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1，一次项
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1，偏置项
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        '''
        BN的目的是为了缓解协方差偏移(covariance shift)，要最大限度地保证每次的正向传播输出在同一分布上，这样反向计算时参照的数据样本分布就会与正向计算时的数据分布一样了，保证分布的统一。
        所以BN将在mini-batch数据上，把输入转换成均值为0，方差为1的高斯分布。
        '''
        #decay可调，它移动平均值的衰减速度，使用的是平滑指数衰减的方法更新均值方差。值太小会导致更新太快，值太大会导致几乎没有衰减，容易出现过拟合。
        #scale进行变换，即会乘gamma进行缩放。
        #is_training=True，在训练中会不断更新样本集均值和方差，在测试时要设置为False，就默认使用训练样本的均值和方差。
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)#强制总phase相等
        return z

    def partial_fit(self, data):
        '''
        喂入某批次的数据给模型，得到loss
        '''
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):
        '''
        从训练集中抽样，生成训练的batch
        '''
        start_index = np.random.randint(0, len(data['Y']) - batch_size)#抽样开始的下标
        X , Y = [], []
        #先正向采样
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        #如果正向采样的数量不够但是已经超出边界，就再逆向采样以确保采样数目是batch_size
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b):
        '''
        同时打乱两个列表，并保持它们的一一对应关系。
        '''
        rng_state = np.random.get_state()#具有相同state的随机生成器(random)的随机效果相同，使得两次生成的随机数相同。
        np.random.shuffle(a)
        np.random.set_state(rng_state)#设置set相同的state
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):
        '''
        训练模型。输入训练集，验证集和测试集。
        '''
        #检查初始化时的模型表现
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" %(init_train, init_valid, init_test, time()-t2))

        for epoch in xrange(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])#打乱且保存一一对应关系
            total_batch = int(len(Train_data['Y']) / self.batch_size)#计算总batch数量
            for i in xrange(total_batch):
                #采样生成训练集的batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                #喂入模型开始训练
                self.partial_fit(batch_xs)
            t2 = time()

            #对每个周期训练完毕后都进行性能评估并记录
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f, test=%.4f [%.1f s]"
                      %(epoch+1, t2-t1, train_result, valid_result, test_result, time()-t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0:
            print "Save model to file as pretrain."
            self.saver.save(self.sess, self.save_file)#保存模型

    def eva_termination(self, valid):
        '''
        终止条件。连续5次的都上升的时候提前终止周期。
        '''
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):
        '''
        评估模型性能。
        '''
        num_example = len(data['Y'])#数据集条数
        feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']], self.dropout_keep: 1.0, self.train_phase: False}#测试阶段
        predictions = self.sess.run((self.out), feed_dict=feed_dict)#利用模型进行预测
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        #计算预测和真实的rmse或者logloss。
        if self.loss_type == 'square_loss':    
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  #y_pred与y_true最小值逐位比较取其大者（小于则为0），下值边界
            predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true)) #predictions_bounded与y_true最大值逐位比较取其小者（大于则为0），上值边界
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)
            return logloss
'''         #测试分类性能，先离散再计算。
            predictions_binary = [] 
            for item in y_pred:
                if item > 0.5:
                    predictions_binary.append(1.0)
                else:
                    predictions_binary.append(0.0)
            Accuracy = accuracy_score(y_true, predictions_binary)
            return Accuracy '''

if __name__ == '__main__':
    #载入命令行参数和数据
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    #打印初始配置
    if args.verbose > 0:
        print("FM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
              %(args.dataset, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm))
    #保存路径和命名
    save_file = '../pretrain/%s_%d/%s_%d' %(args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)
    #开始训练模型
    t1 = time()
    model = FM(data.features_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm, args.verbose)
    model.train(data.Train_data, data.Validation_data, data.Test_data)
    
    #迭代过程中的最佳验证集结果
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = max(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]" 
           %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch], time()-t1))
