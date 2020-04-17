'''
Created on Apr 17, 2020
@author: nakaizura
'''

import scipy.sparse as sp
import numpy as np

#关于import库，numpy就不说了。
#scipy.sparse库中提供了多种表示稀疏矩阵的格式，同时支持稀疏矩阵的加、减、乘、除和幂等。

#关于数据集，原作者使用MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20).

class Dataset(object):
    '''
    数据集类，用于载入数据。
    主要有三个操作函数：载入rating训练集，rating测试集的正例和负例。
    训练集之所以没有负例是因为在模型的训练过程中过随机负采样进行训练，而测试集测试提前负采样好了之后便于公正的验证结果。
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        #需要强制测试集（positive instances正例）和负采样（negative instances负例）的大小一致
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        '''
        正例，这个载入的数据集形式为userID\t itemID\t rating\t timestamp (if have)
        其中没有使用时间戳这一属性。
        '''
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])#第一列和第二列分别为user和item的ID
                ratingList.append([user, item])#组合在一起放入到List中
                line = f.readline()#读下一行
        return ratingList
    
    def load_negative_file(self, filename):
        '''
        负例，一个test.rating的正例对应99个负例，形式为 (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
        '''
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:#第一列是正例的(userID,itemID)，[1:]是后面所有的负例
                    negatives.append(int(x))#存该正例对应的所有负例
                negativeList.append(negatives)
                line = f.readline()#读下一行
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        读训练集，返回稀疏矩阵（dok matrix），形式为userID\t itemID\t rating\t timestamp (if have)
        '''
        #得到users和items的数目
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                #最大的id即是数目
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        #构建矩阵
        #使用dok(Dictionary Of Keys based sparse matrix)构建稀疏矩阵。使用字典保存非0值元素的(行，列)。
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                #评分大于0即认为是正例。ml中是(0,5,0.5)的分数，pinterest是用户'pin'或者'no pin'（即0-1）的lable。
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
