'''
Created on Apr 18, 2020
@author: nakaizura
'''

import numpy as np
import os

class LoadData(object):
    '''输入：数据集路径。返回：处理好格式的Train_data，Test_data和Validation_data。
    三个dataset都是字典集合, 'Y'是标签; 'X'是FM维度的one-hot向量，其中特征维度是features_M。
    '''

    #得到train.libfm，test.libfm和validation.libfm的路径。
    def __init__(self, path, dataset, loss_type="square_loss"):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset +".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = self.map_features( ) #计算总特征数目
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type ) #处理libfm数据成矩阵形式

    def map_features(self): # 映射特征并保存在字典中，实际上是为了得到特征的总维度features_M
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        # print("features_M:", len(self.features))
        return  len(self.features)

    def read_features(self, file): #读入特征文件
        f = open( file ) #打开文件
        line = f.readline() #逐行读取
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:#第0列是Y，后面的是特征
                if item not in self.features:#不在特征集合的就加入到特征集合中并计数，最后得到的是无重复的总特征数目features_M
                    self.features[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):#构造Train_data，Test_data和Validation_data的数据
        X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        #按照不同的loss，使用Y_或者Y_for_logloss
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Train:" , len(Y_))

        X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Validation:", len(Y_))

        X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Test:", len(Y_))

        return Train_data,  Validation_data,  Test_data

    def read_data(self, file):
        #读数据文件，对于每一行，数据的第一列是Y_
        #其他列会变成X_ 然后被映射到self.features里面保存。
        f = open( file )
        X_ = []
        Y_ = []
        Y_for_logloss = []#离散后的Y_，对应着两种数据集构造方法，视方法而构造不同的形态。
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# 第一列如果>0则视为1否则就认为是0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )#其他列都放入到X_中
            line = f.readline()#读下一行
        f.close()
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_] #每个样本的特征数量
        indexs = np.argsort(X_lens) #从小到大的索引值
        Data_Dic['Y'] = [ Y_[i] for i in indexs] #按索引构造数据集
        Data_Dic['X'] = [ X_[i] for i in indexs]
        return Data_Dic
    
    def truncate_features(self):
        """
        确保每个特征的长度都是一致的，所以按照样本的特征长度（最小的）对其他特征进行截断
        """
        num_variable = len(self.Train_data['X'][0])
        for i in xrange(len(self.Train_data['X'])):#找到最小的长度
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        #截断train, validation and test
        for i in xrange(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in xrange(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in xrange(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable
