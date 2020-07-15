'''
Created on Jul 15, 2020
@author: nakaizura
'''
from __future__ import print_function


import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# 分类器是使用sklearn的OneVsRestClassifier处理多分类任务。
# 对n类，会建立n个二分类器，每个分类器针对其中一个类别和剩余类别进行分类。

class TopKRanker(OneVsRestClassifier):
    #注意这里OneVsRestClassifier
    def predict(self, X, top_k_list):
        #预测分类概率
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list): #对所有Y选择概率最大的类别
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()#排序得到label
            probs_[:] = 0 #one-hot操作，只有label处为1，其他地方都为0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y) #多标签二值化
        self.clf.fit(X_train, Y) #训练分类器

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)#预测一个类别
        Y = self.binarizer.transform(Y) #多标签二值化
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages: #算F1
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        print('-------------------')
        print(results)
        return results
        print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        #设定状态，记录下数组被打乱的操作，以使打乱前后实例与标签的一一对应
        state = numpy.random.get_state() 

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed) #固定随机种子便于复现结果
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        #前80训练，后20测试
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)#恢复打乱前的状态
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y
