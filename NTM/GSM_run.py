#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Created on Jun 23, 2021
@author: nakaizura
'''

# NVDM-GSM是基于VAE的一种神经主题模型，模型代码的架构和VAE是类似的。
# 输入词袋，然后建模分布，然后采样计算VAE的两大损失。

import os
import re
import torch
import pickle
import argparse
import logging
import time
from models import GSM
from utils import *
from dataset import DocDataset
from multiprocessing import cpu_count
#from torch.utils.data import Dataset,DataLoader

parser = argparse.ArgumentParser('GSM topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=10,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--rebuild',action='store_true',help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
parser.add_argument('--batch_size',type=int,default=512,help='Batch size (default=512)')
parser.add_argument('--criterion',type=str,default='cross_entropy',help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--ckpt',type=str,default=None,help='Checkpoint path')

args = parser.parse_args() #载入参数

def main():
    global args
    taskname = args.taskname # 数据集名字
    no_below = args.no_below # 文档频率小于阈值的词会被过滤掉
    no_above = args.no_above # 文档频率小于阈值的词将被过滤掉
    num_epochs = args.num_epochs # 训练周期
    n_topic = args.n_topic # 主题数
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue # 是否在之前的checkoint上继续训练
    use_tfidf = args.use_tfidf # 是否用tfidf作为BOW输入
    rebuild = args.rebuild # 是否重建语料，默认不会
    batch_size = args.batch_size # 批次大小
    criterion = args.criterion # loss的种类
    auto_adj = args.auto_adj # 是否自动调整频率，如去掉top20
    ckpt = args.ckpt # ckpt路径

    device = torch.device('cpu')
    docSet = DocDataset(taskname,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)# 载入数据集，并分词
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)
    
    voc_size = docSet.vocabsize
    print('voc size:',voc_size)

    if ckpt:# 载入ckpt
        checkpoint=torch.load(ckpt)
        param.update({"device": device})
        model = GSM(**param)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion,ckpt=checkpoint)
    else:
        # 初始化模型并开始执行train程序
        model = GSM(bow_dim=voc_size,n_topic=n_topic,taskname=taskname,device=device)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion)
    model.evaluate(test_data=docSet)# 用训练之后的模型做评估
    # 存模型，特征，统计等等结果
    save_name = f'./ckpt/GSM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    torch.save(model.vae.state_dict(),save_name)
    txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
    with open('topic_dist_gsm.txt','w',encoding='utf-8') as wfp:
        for t,e in zip(txt_lst,embeds):
            wfp.write(f'{e}:{t}\n')
    pickle.dump({'txts':txt_lst,'embeds':embeds},open('gsm_embeds.pkl','wb'))

if __name__ == "__main__":
    main()
