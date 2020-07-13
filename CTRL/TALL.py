'''
Created on Jul 13, 2020
@author: nakaizura
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def weights_init(m):#初始化权重
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)#均值0方差0.01的高斯分布
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)


class TALL(nn.Module):
    def __init__(self):
        super(TALL, self).__init__()
        self.semantic_size = 1024 # 视觉和文本要投影的共同语义维度
        self.sentence_embedding_size = 4800 #sentence2vec得到的维度
        self.visual_feature_dim = 4096*3 #中心+上下文一共3个，每个由C3D得到是4096维
        self.v2s_lt = nn.Linear(self.visual_feature_dim, self.semantic_size) #投影视觉
        self.s2s_lt = nn.Linear(self.sentence_embedding_size, self.semantic_size) #投影文本
        self.fc1 = torch.nn.Conv2d(4096, 1000, kernel_size=1, stride=1)#2层FC得到预测结果
        self.fc2 = torch.nn.Conv2d(1000, 3, kernel_size=1, stride=1)
        # 初始化权重
        self.apply(weights_init)

    def cross_modal_comb(self, visual_feat, sentence_embed):
        #这是完成特征交叉的模块，会分别做加法、乘法和拼接
        batch_size = visual_feat.size(0)
        # shape_matrix = torch.zeros(batch_size,batch_size,self.semantic_size)

        #因为视频会有多个，而句子只有一个，所以要做一下维度变化
        vv_feature = visual_feat.expand([batch_size,batch_size,self.semantic_size])
        ss_feature = sentence_embed.repeat(1,1,batch_size).view(batch_size,batch_size,self.semantic_size)

        concat_feature = torch.cat([vv_feature, ss_feature], 2)#横向拼接（第0维度是batch）

        mul_feature = vv_feature * ss_feature # 56,56,1024，乘法
        add_feature = vv_feature + ss_feature # 56,56,1024，加法

        #将各个特征一起合并起来得到组合特征
        comb_feature = torch.cat([mul_feature, add_feature, concat_feature], 2)

        return comb_feature


    def forward(self, visual_feature_train, sentence_embed_train):
        #对视觉特征投影到语义空间并norm
        transformed_clip_train = self.v2s_lt(visual_feature_train)
        transformed_clip_train_norm = F.normalize(transformed_clip_train, p=2, dim=1)

        #对本文特征投影到语义空间并norm
        transformed_sentence_train = self.s2s_lt(sentence_embed_train)
        transformed_sentence_train_norm = F.normalize(transformed_sentence_train, p=2, dim=1)

        #做特征交叉
        cross_modal_vec_train = self.cross_modal_comb(transformed_clip_train_norm, transformed_sentence_train_norm)

        cross_modal_vec_train = cross_modal_vec_train.unsqueeze(0).permute(0, 3, 1, 2)
        #2层FC得到预测结果
        mid_output = self.fc1(cross_modal_vec_train)
        mid_output = F.relu(mid_output)
        sim_score_mat_train = self.fc2(mid_output).squeeze(0)

        return sim_score_mat_train
