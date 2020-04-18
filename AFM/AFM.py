'''
Created on Apr 18, 2020
@author: nakaizura
'''

import math
import os, sys
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

#################### Arguments ####################
def parse_args():
     '''
    python自带的参数解析包argparse，方便读取命令行参数。
    处理过程是训练还是测试、微观分析模式、数据集路径、数据集名称、周期、是否预训练、批次大小、是否注意力、隐层大小、正则化系数、dropout比率、学习率、冻结FM、优化器、打印间隔、是否BN批次正则化，BN的decay因子。
    微观分析是论文中探讨Attention的可解释性部分，主要是对User-Item，User-Tag，Item-Tag三种交互模式谁更重要。
    '''
    parser = argparse.ArgumentParser(description="Run DeepFM.")
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--mla', type=int, default=0,
                        help='Set the experiment mode to be Micro Level Analysis or not: 0-disable, 1-enable.')
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-tag',
                        help='Choose a dataset.')
    parser.add_argument('--valid_dimen', type=int, default=3,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file; 2: initialize from pretrain and save to pretrain file')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--attention', type=int, default=1,
                        help='flag for attention. 1: use attention; 0: no attention')
    parser.add_argument('--hidden_factor', nargs='?', default='[16,16]',
                        help='Number of hidden factors.')
    parser.add_argument('--lamda_attention', type=float, default=1e+2,
                        help='Regularizer for attention part.')
    parser.add_argument('--keep', nargs='?', default='[1.0,0.5]',
                        help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--freeze_fm', type=int, default=0,
                        help='Freese all params of fm and learn attention params only.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--decay', type=float, default=0.999,
                    help='Decay value for batch norm')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')

    return parser.parse_args()

class AFM(BaseEstimator, TransformerMixin):
    '''
    构建AFM模型。
    '''
    def __init__(self, features_M, pretrain_flag, save_file, attention, hidden_factor, valid_dimension, activation_function, num_variable, 
                 freeze_fm, epoch, batch_size, learning_rate, lamda_attention, keep, optimizer_type, batch_norm, decay, verbose, micro_level_analysis, random_seed=2016):
        #参数传入AFM模型
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attention = attention
        self.hidden_factor = hidden_factor
        self.valid_dimension = valid_dimension
        self.activation_function = activation_function
        self.num_variable = num_variable
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_attention = lamda_attention
        self.keep = keep
        self.freeze_fm = freeze_fm
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.decay = decay
        self.verbose = verbose
        self.micro_level_analysis = micro_level_analysis
        #存储每个epoch的性能rmse结果
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        #初始化tensorflow graph的所有变量
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            #设置图的随机种子
            tf.set_random_seed(self.random_seed)
            #输入数据。主要是features和labels。None是批次数目，features_M是数据集有的特征one-hot维度
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name="train_features_afm")  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_afm")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_afm")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase_afm")

            #初始化变量权重
            self.weights = self._initialize_weights()

            #模型part
            #先嵌入
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features) # None * M' * K，嵌入维度是K，M'为field size

            #对特征两两组合，再相乘。
            element_wise_product_list = []
            count = 0
            for i in range(0, self.valid_dimension):#用2层for实现组合
                for j in range(i+1, self.valid_dimension):
                    element_wise_product_list.append(tf.multiply(self.nonzero_embeddings[:,i,:], self.nonzero_embeddings[:,j,:]))#再相乘
                    count += 1
            self.element_wise_product = tf.stack(element_wise_product_list) # (M'*(M'-1)) * None * K，一共(M'*(M'-1))对组合
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1,0,2], name="element_wise_product") # None * (M'*(M'-1)) * K，调整一下维度
            self.interactions = tf.reduce_sum(self.element_wise_product, 2, name="interactions")#再求和
            # _________ MLP Layer / attention part _____________
            #开始计算Attention部分。按照公式首先算h^T Relu((W v_iv_jx_ix_j)+b)，再softmax的都分数，最后再回乘为每个组合的重要性进行加权。
            num_interactions = self.valid_dimension*(self.valid_dimension-1)/2#总的组合数，因为内层是从i+1开始的，可以理解为矩阵的一半
            if self.attention:
                self.attention_mul = tf.reshape(tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.hidden_factor[1]]), \
                    self.weights['attention_W']), shape=[-1, num_interactions, self.hidden_factor[0]])#先算(W v_iv_jx_ix_j)+b
                # self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                #     self.weights['attention_b'])), 2, keep_dims=True)) # None * (M'*(M'-1)) * 1
                # self.attention_sum = tf.reduce_sum(self.attention_exp, 1, keep_dims=True) # None * 1 * 1
                # self.attention_out = tf.div(self.attention_exp, self.attention_sum, name="attention_out") # None * (M'*(M'-1)) * 1
                self.attention_relu = tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                    self.weights['attention_b'])), 2, keep_dims=True) # None * (M'*(M'-1)) * 1，进行Relu
                self.attention_out = tf.nn.softmax(self.attention_relu)#再softmax得到分数
                self.attention_out = tf.nn.dropout(self.attention_out, self.dropout_keep[0]) # dropout
            
            # _________ Attention-aware Pairwise Interaction Layer _____________
            if self.attention:#用分数给element_wise_product即组合进行加权
                self.AFM = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), 1, name="afm") # None * K
            else:#如果不用注意力加权就是element_wise_product本身
                self.AFM = tf.reduce_sum(self.element_wise_product, 1, name="afm") # None * K
            #算AFM_FM是为了论文中的微观分析。
            self.AFM_FM = tf.reduce_sum(self.element_wise_product, 1, name="afm_fm") # None * K
            self.AFM_FM = self.AFM_FM / num_interactions
            self.AFM = tf.nn.dropout(self.AFM, self.dropout_keep[1]) # dropout

            # _________ out _____________
            #得到预测的输出，除了二次项，还需要计算偏置项w0和一次项\sum w_ix_i
            if self.micro_level_analysis:#微观分析对比两种变体。
                self.out = tf.reduce_sum(self.AFM, 1, keep_dims=True, name="out_afm")
                self.out_fm = tf.reduce_sum(self.AFM_FM, 1, keep_dims=True, name="out_fm")
            else:
                self.prediction = tf.matmul(self.AFM, self.weights['prediction']) # None * 1
                Bilinear = tf.reduce_sum(self.prediction, 1, keep_dims=True)  # None * 1
                self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
                Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
                #FM的输出最后由三部分组成二次，一次，偏置
                self.out = tf.add_n([Bilinear, self.Feature_bias, Bias], name="out_afm")  # None * 1

            #计算损失函数。
            if self.attention and self.lamda_attention > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_attention)(self.weights['attention_W'])  # regulizer
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

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

            #计算整个模型的参数数量，这主要是为了证明AFM比其他并行神经网络拥有更少的参数量。
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() #每个参数变量的维度大小
                variable_parameters = 1
                for dim in shape:#所有维度的数量
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print "#params: %d" %total_parameters 
    
    def _init_session(self):
        #限制GPU资源的使用
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #动态申请显存
        return tf.Session(config=config)

    def _initialize_weights(self):
        '''
        初始化权重。分是否有预训练过的模型两种初始化方法。
        '''
        all_weights = dict()
        #如果选择冻结FM层，则其他的参数都是不可训练的
        trainable = self.freeze_fm == 0
        if self.pretrain_flag > 0 or self.micro_level_analysis:
            #如果有预训练过的模型，就载入参数
            from_file = self.save_file
            # if self.micro_level_analysis: #围观分析都需要计算两种得分。
            from_file = self.save_file.replace('afm', 'fm')
            weight_saver = tf.train.import_meta_graph(from_file + '.meta')
            pretrain_graph = tf.get_default_graph()#获取当前默认计算图。
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with self._init_session() as sess:
                weight_saver.restore(sess, from_file)#恢复参数
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            # all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32, name='feature_embeddings')
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32, name='feature_embeddings', trainable=trainable)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32, name='feature_bias', trainable=trainable)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32, name='bias', trainable=trainable)
        else:
            #如果没有，就随机初始化所有参数
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor[1]], 0.0, 0.01),
                name='feature_embeddings', trainable=trainable)  # features_M * K，输入特征的数目features_M到嵌入特征的数目K
            all_weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias', trainable=trainable)  # features_M * 1，一次项
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias', trainable=trainable)  # 1 * 1，偏置项

        # attention，共有三种参数W，b，p
        if self.attention:
            glorot = np.sqrt(2.0 / (self.hidden_factor[0]+self.hidden_factor[1]))
            all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor[1], self.hidden_factor[0])), dtype=np.float32, name="attention_W")  # K * AK
            all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_factor[0])), dtype=np.float32, name="attention_b")  # 1 * AK
            all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.hidden_factor[0])), dtype=np.float32, name="attention_p") # AK

        # prediction layer
        all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor[1], 1), dtype=np.float32))  # hidden_factor * 1

        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        '''
        BN的目的是为了缓解协方差偏移(covariance shift)，要最大限度地保证每次的正向传播输出在同一分布上，这样反向计算时参照的数据样本分布就会与正向计算时的数据分布一样了，保证分布的统一。
        所以BN将在mini-batch数据上，把输入转换成均值为0，方差为1的高斯分布。
        '''
        #decay可调，它移动平均值的衰减速度，使用的是平滑指数衰减的方法更新均值方差。值太小会导致更新太快，值太大会导致几乎没有衰减，容易出现过拟合。
        #scale进行变换，即会乘gamma进行缩放。
        #is_training=True，在训练中会不断更新样本集均值和方差，在测试时要设置为False，就默认使用训练样本的均值和方差。
        bn_train = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
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
        从训练集中随机抽样，生成训练的batch
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
    
    def get_ordered_block_from_data(self, data, batch_size, index):
        '''
        从训练集中顺序抽样，生成训练的batch
        '''
        start_index = index*batch_size
        X , Y = [], []
        #顺序抽样
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
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
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            print("Init: \t train=%.4f, validation=%.4f [%.1f s]" %(init_train, init_valid, time()-t2))

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
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f [%.1f s]"
                      %(epoch+1, t2-t1, train_result, valid_result, time()-t2))

            # test_result = self.evaluate(Test_data)
            # print("Epoch %d [%.1f s]\ttest=%.4f [%.1f s]"
            #       %(epoch+1, t2-t1, test_result, time()-t2))
            if self.eva_termination(self.valid_rmse):#如果早停
                break

        if self.pretrain_flag < 0 or self.pretrain_flag == 2:
            print "Save model to file as pretrain."
            self.saver.save(self.sess, self.save_file)#保存模型

    def eva_termination(self, valid):
        '''
        提前终止条件。连续5次的都上升的时候提前终止周期。
        '''
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def evaluate(self, data):
        '''
        评估模型性能。
        '''
        num_example = len(data['Y'])#数据集条数
        #从0开始顺序采样，喂入第一个batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        # batch_xs = data
        y_pred = None
        # if len(batch_xs['X']) > 0:
        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [[y] for y in batch_xs['Y']], self.dropout_keep: list(1.0 for i in range(len(self.keep))), self.train_phase: False}
            a_out, batch_out = self.sess.run((self.attention_out, self.out), feed_dict=feed_dict)#利用模型进行预测
            
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            #生成下一个batch
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)

        y_true = np.reshape(data['Y'], (num_example,))

        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  #y_pred与y_true最小值逐位比较取其大者（小于则为0），下值边界
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  #predictions_bounded与y_true最大值逐位比较取其小者（大于则为0），上值边界
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        return RMSE

def make_save_file(args):#保存模型和命名
    pretrain_path = '../pretrain/%s_%d' %(args.dataset, eval(args.hidden_factor)[1])
    if args.mla:
        pretrain_path += '_mla'
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path+'/%s_%d' %(args.dataset, eval(args.hidden_factor)[1])
    return save_file

def train(args):
    #载入命令行参数和数据
    data = DATA.LoadData(args.path, args.dataset)
    #打印初始配置
    if args.verbose > 0:
        print("AFM: dataset=%s, factors=%s, attention=%d, freeze_fm=%d, #epoch=%d, batch=%d, lr=%.4f, lambda_attention=%.1e, keep=%s, optimizer=%s, batch_norm=%d, decay=%f, activation=%s"
              %(args.dataset, args.hidden_factor, args.attention, args.freeze_fm, args.epoch, args.batch_size, args.lr, args.lamda_attention, args.keep, args.optimizer, 
              args.batch_norm, args.decay, args.activation))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function == tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity
    
    save_file = make_save_file(args)
    #开始训练模型
    t1 = time()

    num_variable = data.truncate_features()#截取特征为一样的维度
    if args.mla:
        args.freeze_fm = 1
    model = AFM(data.features_M, args.pretrain, save_file, args.attention, eval(args.hidden_factor), args.valid_dimen, 
        activation_function, num_variable, args.freeze_fm, args.epoch, args.batch_size, args.lr, args.lamda_attention, eval(args.keep), args.optimizer, 
        args.batch_norm, args.decay, args.verbose, args.mla)
    
    model.train(data.Train_data, data.Validation_data, data.Test_data)
    
    #迭代过程中的最佳验证集结果
    best_valid_score = 0
    best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f [%.1f s]" 
           %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time()-t1))

def evaluate(args):
    '''
    测试模式。将载入数据集，恢复训练好的模型，再开始评估。
    '''
    #载入测试集
    data = DATA.LoadData(args.path, args.dataset).Test_data
    save_file = make_save_file(args)
    
    #载入计算图
    weight_saver = tf.train.import_meta_graph(save_file + '.meta')
    pretrain_graph = tf.get_default_graph()
    # 载入特征tensor
    # feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
    # feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
    # bias = pretrain_graph.get_tensor_by_name('bias:0')
    # afm = pretrain_graph.get_tensor_by_name('afm:0')
    out_of_afm = pretrain_graph.get_tensor_by_name('out_afm:0')
    interactions = pretrain_graph.get_tensor_by_name('interactions:0')
    attention_out = pretrain_graph.get_tensor_by_name('attention_out:0')
    #afm的参数
    train_features_afm = pretrain_graph.get_tensor_by_name('train_features_afm:0')
    train_labels_afm = pretrain_graph.get_tensor_by_name('train_labels_afm:0')
    dropout_keep_afm = pretrain_graph.get_tensor_by_name('dropout_keep_afm:0')
    train_phase_afm = pretrain_graph.get_tensor_by_name('train_phase_afm:0')

    #fm的参数
    if args.mla:
         out_of_fm = pretrain_graph.get_tensor_by_name('out_fm:0')
         element_wise_product = pretrain_graph.get_tensor_by_name('element_wise_product:0')
         train_features_fm = pretrain_graph.get_tensor_by_name('train_features_fm:0')
         train_labels_fm = pretrain_graph.get_tensor_by_name('train_labels_fm:0')
         dropout_keep_fm = pretrain_graph.get_tensor_by_name('dropout_keep_fm:0')
         train_phase_fm = pretrain_graph.get_tensor_by_name('train_phase_fm:0')

    #恢复session
    #限制GPU资源的使用
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #动态申请显存
    sess = tf.Session(config=config)
    weight_saver.restore(sess, save_file)#重载参数

    #开始评估
    num_example = len(data['Y'])#数据集条数
    if args.mla:
        feed_dict = {train_features_afm: data['X'], train_labels_afm: [[y] for y in data['Y']], dropout_keep_afm: [1.0,1.0], train_phase_afm: False, \
                     train_features_fm: data['X'], train_labels_fm: [[y] for y in data['Y']], dropout_keep_fm: 1.0, train_phase_fm: False}
        ao, inter, out_fm, predictions = sess.run((attention_out, interactions, out_of_fm, out_of_afm), feed_dict=feed_dict)
    else:
        feed_dict = {train_features_afm: data['X'], train_labels_afm: [[y] for y in data['Y']], dropout_keep_afm: [1.0,1.0], train_phase_afm: False}
        predictions = sess.run((out_of_afm), feed_dict=feed_dict)

    #计算rmse表现
    y_pred_afm = np.reshape(predictions, (num_example,))
    y_true = np.reshape(data['Y'], (num_example,))
    
    predictions_bounded = np.maximum(y_pred_afm, np.ones(num_example) * min(y_true))  #y_pred与y_true最小值逐位比较取其大者（小于则为0），下值边界
    predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  #predictions_bounded与y_true最大值逐位比较取其小者（大于则为0），上值边界
    RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))#算rmse

    print("Test RMSE: %.4f"%(RMSE))

    if args.mla:
        #显著性测试
        ao = np.reshape(ao, (num_example, 3))
        y_pred_fm = np.reshape(out_fm, (num_example,))
        pred_abs_fm = abs(y_pred_fm - y_true)
        pred_abs_afm = abs(y_pred_afm - y_true)
        pred_abs = pred_abs_afm - pred_abs_fm

        ids = np.arange(0, num_example, 1)

        sorted_ids = sorted(ids, key=lambda k: pred_abs_afm[k]+abs(ao[k][0]*ao[k][1]*ao[k][2]))
        # sorted_ids = sorted(ids, key=lambda k: abs(ao[k][0]*ao[k][1]*ao[k][2]))
        for i in range(3):
            _id = sorted_ids[i]
            print('## %d: %d'%(i+1, y_true[_id]))
            print('0.33*%.2f + 0.33*%.2f + 0.33*%.2f = %.2f'%(inter[_id][0], inter[_id][1], inter[_id][2], y_pred_fm[_id]))
            print('%.2f*%.2f + %.2f*%.2f + %.2f*%.2f = %.2f\n'%(\
                          ao[_id][0], inter[_id][0], \
                          ao[_id][1], inter[_id][1], \
                          ao[_id][2], inter[_id][2], y_pred_afm[_id]))


    

if __name__ == '__main__':
    args = parse_args()

    # if args.mla:
    #     args.lr = 0.1
    #     args.keep = '[1.0,1.0]'
    #     args.lamda_attention = 10.0
    # else:
    #     args.lr = 0.1
    #     args.keep = '[1.0,0.5]'
    #     args.lamda_attention = 100.0

    if args.process == 'train':#训练
        train(args)
    elif args.process == 'evaluate':#测试
        evaluate(args)
