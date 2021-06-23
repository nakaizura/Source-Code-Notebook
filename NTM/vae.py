#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Created on Jun 23, 2021
@author: nakaizura
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE model
# 输入，建模分布的mu和var，采样得到向量，然后重建+KL约束
class VAE(nn.Module):
    def __init__(self, encode_dims=[2000,1024,512,20],decode_dims=[20,1024,2000],dropout=0.0):

        super(VAE, self).__init__()
        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1]) # 学习mu和var
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1])

        self.decoder = nn.ModuleDict({
            f'dec_{i}':nn.Linear(decode_dims[i],decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])
        
        
    def encode(self, x):# 编码
        hid = x
        for i,layer in self.encoder.items():# 多层fc
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)# 得到mu和var
        return mu, log_var

    def inference(self,x):# 推断
        mu, log_var = self.encode(x)# 得到分布
        theta = torch.softmax(x,dim=1)# 得到向量
        return theta
    
    def reparameterize(self, mu, log_var):# 重参数技巧，使训练可微
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)# 采样
        z = mu + eps * std
        return z

    def decode(self, z):# 解码
        hid = z
        for i,(_,layer) in enumerate(self.decoder.items()):# 多层fc
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        return hid
    
    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)# 得到分布的mu和var
        _theta = self.reparameterize(mu, log_var)# 重参数采样得到向量
        _theta = self.fc1(_theta) 
        if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta
        x_reconst = self.decode(theta)# 重建loss
        return x_reconst, mu, log_var # 返回重建和两个分布参数，KL散度在模型中计算，不在此处

if __name__ == '__main__':
    model = VAE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var = model(inpt)
    print(out.shape)
    print(mu.shape)
