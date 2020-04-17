'''
Created on Apr 17, 2020
@author: nakaizura
'''

import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

#整体代码与GMF一致，模型的部分对于user和item的交互由MF变成MLP。

#################### 载入参数 ####################
def parse_args():
     '''
    python自带的参数解析包argparse，方便读取命令行参数。
    参数主要有：数据集路径、数据集名称、周期数、批次大小、MLP逐层大小、正则化、负采样数目、梯度下降学习率、优化器、打印间隔，是否保存模型。
    '''
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    '''
    预定义初始化方法
    '''
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    '''
    构建MLP模型，输入是user、item数目，layer逐层，对嵌入的正则项regs。
    '''
    assert len(layers) == len(reg_layers)#强制相等才能正则化。
    num_layer = len(layers) #MLP总层数
    
    #Input():用来实例化一个keras张量,shape为(1,)预计输入是一批1维的int向量
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    #embedding随机初始化用于学习user和item的嵌入
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = 'user_embedding',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'item_embedding',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   
    
    #embedding后的向量会是2D的，需要flatten压平到1D，方便后面做Dense层。
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    #第0层是两个嵌入向量的拼接
    vector = merge([user_latent, item_latent], mode = 'concat')
    
    #构建MLP层
    for idx in xrange(1, num_layer):#按层数构建多个Dense
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    #最后还是用Dense做投影预测得分。sigmoid限制在0-1的分数区间。
    #LeCun均匀初始化器，从[-sqrt(3 / fan_in),sqrt(3 / fan_in)]抽样本，fan_in是输入层数量。下一行是keras更新后版本的写法。
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)

    #keras的函数式模型，给定input和output就行。
    #如果使用Sequential顺序式模型，需要把中间层都写出来如seq_model =Sequential(layers=[input, hiden1,..., hiden2, output])
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def get_train_instances(train, num_negatives):
    '''
    生成训练时的实例。用于生成真正用于训练的函数，即对正例进行随机负采样。
    在测试时候不会随机生成，而是会提前处理好这样便于公平的验证结果，所以可以看到数据集是3个（train正例，test的正例和负例）
    '''
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():#用户和物品有交互的（u，i），即正例
        # 正例就直接append
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # 负例即没有交互的item
        for t in xrange(num_negatives):#按照负例采样数目
            j = np.random.randint(num_items)#随机进行选取
            while train.has_key((u, j)):#且不存在交互的物品j
                j = np.random.randint(num_items)
            user_input.append(u)#然后再append
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    #导入所有参数
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)#恢复成列表不是字符串
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()线程数
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    #载入训练集，测试集正例和负例。
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    #构建MLP模型
    model = get_model(num_users, num_items, layers, reg_layers)
    #根据不同的命令行输出，确定不同的梯度下降优化器。
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    #检查初始化时候的表现。即没有训练过的model在test上的hits, ndcgs性能。
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    #开始训练模型
    best_hr, best_ndcg, best_iter = hr, ndcg, -1#记录最佳值
    for epoch in xrange(epochs):
        t1 = time()
        #用train生成训练实例，即针对正例随机负采样。
        user_input, item_input, labels = get_train_instances(train, num_negatives)
    
        #模型开始训练    
        hist = model.fit([np.array(user_input), np.array(item_input)], #用户和含有正负例的item列表，由模型预测分数topk排序
                         np.array(labels), #对应item的标签
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        #再次用test评估模型，按间隔隔一段时间就打印结果。
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:#保存best结果
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
