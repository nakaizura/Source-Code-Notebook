'''
Created on Jul 22, 2020
@author: nakaizura
'''

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')#批次大小
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')#隐层size
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')#训练周期
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 学习率，[0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')#学习率衰减
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')#衰减步
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # L2，[0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')#GNN的传播步数
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')#早停
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')#只使用global，没有local
parser.add_argument('--validation', action='store_true', help='validation')#是否设置验证集
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')#比率
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))#载入数据集
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)#切分数据集
        test_data = valid_data #此时测试集和验证集一样
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)#打乱训练集
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node)) #实例化模型

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch): #开始训练
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]: #计算hit和mrr
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience: #早停
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
