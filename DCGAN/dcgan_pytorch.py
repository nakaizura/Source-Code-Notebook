'''
Created on Apr 24, 2020
@author: nakaizura
'''

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


#设置可再现的随机种子
manualSeed = 999
#manualSeed = random.randint(1, 10000) #如果想要得到新的结果，就随机挑种子
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


#数据集路径
dataroot = "./data"
#dataloader加载数据的工作线程数目
workers = 2
#批次大小
batch_size = 128
#图片大小
image_size = 64
#图片通道数
nc = 3
#随机噪音维度
nz = 100

#生成器特征映射维度
ngf = 64
#判别器特征映射维度
ndf = 64
#训练周期
num_epochs = 5

#学习率
lr = 0.0002
#Adam优化器的超参
beta1 = 0.5

#0是cpu，大于1在gpu。
ngpu = 1



def weights_init(m):
    #初始化参数
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: #卷积的均值是0，方差是0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1: #BN的均值是1，方差是0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    #生成器
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #随机噪音反卷积+接个BN加快收敛+ReLU激活一下
            #nn.ConvTranspose2d(输入通道，输出通道，卷积核，步长，padding，bias）
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #特征大小：(ngf*8) x 4 x 4
            #H_out=(H_in-1)xstride-2xpadding+kernel
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #特征大小：(ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #特征大小：(ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #特征大小：(ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #特征大小：(nc) x 64 x 64，nc是通道数
        )

    def forward(self, input):
        return self.main(input)


#实例化生成器
netG = Generator(ngpu).to(device)
#处理多gpu
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
#权重初始化
netG.apply(weights_init)
#打印模型
print(netG)



class Discriminator(nn.Module):
     #鉴别器
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #输入是3 x 64 x 64的标准图像
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #特征大小：(ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #特征大小：(ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #特征大小：(ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #特征大小：(ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


#实例化判别器
netD = Discriminator(ngpu).to(device)
#处理多gpu
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
#权重初始化
netD.apply(weights_init)
#打印模型
print(netD)



#使用交叉熵损失：BCELoss
criterion = nn.BCELoss()
#随机噪音
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
#真实与生成图像的label
real_label = 1
fake_label = 0
#Adam优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


#开始训练
img_list = []#保存生成的图片
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    #批次载入数据
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1)更新判别器网络: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ##先用全部为真实图像的批次训练
        netD.zero_grad()#清零梯度
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)#label全为1
        #得到D的预测结果
        output = netD(real_cpu).view(-1)
        #计算交叉熵损失
        errD_real = criterion(output, label)
        #反向传播
        errD_real.backward()
        D_x = output.mean().item()

        ##再用全部为生成假图片的批次训练
        #生成随机噪声
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        #生成假图片
        fake = netG(noise)
        label.fill_(fake_label)
        #得到D的预测结果
        output = netD(fake.detach()).view(-1)
        #计算交叉熵损失
        errD_fake = criterion(output, label)
        #反向传播
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        #将两部分加起来
        errD = errD_real + errD_fake
        #更新判别器参数
        optimizerD.step()



        ############################
        # (2)更新生成器参数: maximize log(D(G(z)))
        ###########################
        netG.zero_grad() #梯度清零
        label.fill_(real_label)  #对于判别器来说，生成的假图的label是‘真实’的
        #得到判别器分数
        output = netD(fake).view(-1)
        #计算交叉熵损失
        errG = criterion(output, label)
        #反向传播
        errG.backward()
        D_G_z2 = output.mean().item()
        #更新生成器参数
        optimizerG.step()

        #打印训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        #存储loss结果便于画图
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

#看到生成的图片结果
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
