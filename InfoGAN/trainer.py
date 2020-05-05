'''
Created on May 5, 2020
@author: nakaizura
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

class log_gaussian:
  #log高斯，根据均值mu和方差val计算
  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

#训练类
class Trainer:

  def __init__(self, G, FE, D, Q):
    #三个主要部分，FE是Q和D的前端
    self.G = G
    self.FE = FE#Q通过与D共享卷积层，可以减少计算花销。
    self.D = D
    self.Q = Q

    self.batch_size = 100 #批次大小是100

  def _noise_sample(self, dis_c, con_c, noise, bs):
    #采样噪声
    idx = np.random.randint(10, size=bs)#bs是批次大小
    c = np.zeros((bs, 10))#10维的噪声以控制10个数字的生成
    c[range(bs),idx] = 1.0 #one-hot标记类别

    dis_c.data.copy_(torch.Tensor(c))#控制类别
    con_c.data.uniform_(-1.0, 1.0)#控制旋转角度和宽度
    noise.data.uniform_(-1.0, 1.0)#初试噪音
    #62+2+10=74
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

    return z, idx

  def train(self):
    #开始训练
    real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()
    label = torch.FloatTensor(self.batch_size, 1).cuda()
    dis_c = torch.FloatTensor(self.batch_size, 10).cuda()#控制类别
    con_c = torch.FloatTensor(self.batch_size, 2).cuda()#控制旋转角度和宽度
    noise = torch.FloatTensor(self.batch_size, 62).cuda()#噪音

    #变量化
    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    dis_c = Variable(dis_c)
    con_c = Variable(con_c)
    noise = Variable(noise)

    #各个部分的评估标准
    criterionD = nn.BCELoss().cuda() #交叉熵，二分类
    criterionQ_dis = nn.CrossEntropyLoss().cuda()#交叉熵，多分类
    criterionQ_con = log_gaussian()#log高斯

    #优化器是Adam
    optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

    #导入mnist数据集
    dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    #固定随机变量
    c = np.linspace(-1, 1, 10).reshape(1, -1)#调整控制生成的info
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    idx = np.arange(10).repeat(10)
    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1
    fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)#62的固定噪音大小

    #开始训练
    for epoch in range(100):
      for num_iters, batch_data in enumerate(dataloader, 0):

        # real part
        optimD.zero_grad()#D梯度清零
        
        x, _ = batch_data

        bs = x.size(0)
        real_x.data.resize_(x.size())
        label.data.resize_(bs, 1) #标签
        dis_c.data.resize_(bs, 10) #74的噪音向量
        con_c.data.resize_(bs, 2)
        noise.data.resize_(bs, 62)
        
        real_x.data.copy_(x)
        fe_out1 = self.FE(real_x)#前端部分（和Q共享）
        probs_real = self.D(fe_out1)#得到对真实数据的预测结果
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)#评估
        loss_real.backward()#反向传播

        # fake part
        z, idx = self._noise_sample(dis_c, con_c, noise, bs)#再输入噪音数据
        fake_x = self.G(z)
        fe_out2 = self.FE(fake_x.detach())#前端部分（和Q共享）
        probs_fake = self.D(fe_out2)#得到对假数据的预测结果
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)#评估
        loss_fake.backward()#反向传播

        D_loss = loss_real + loss_fake #D的损失是对真+假的判别结果

        optimD.step() #D的梯度更新
        
        # G and Q part
        optimG.zero_grad()#G梯度清零

        fe_out = self.FE(fake_x)#D的前端
        probs_fake = self.D(fe_out)#得到D对生成结果的预测
        label.data.fill_(1.0)

        #就可以判断生成的质量是否骗过鉴别器
        reconstruct_loss = criterionD(probs_fake, label)
        
        q_logits, q_mu, q_var = self.Q(fe_out)#用前端（共享D）的结果得到Q的结果
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_logits, target)
        con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
        
        G_loss = reconstruct_loss + dis_loss + con_loss #互信息下界
        G_loss.backward()#反向传播
        optimG.step()#梯度更新

        if num_iters % 100 == 0:#定期打印

          print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy())
          )
          #生成可视化的结果
          noise.data.copy_(fix_noise)
          dis_c.data.copy_(torch.Tensor(one_hot))

          con_c.data.copy_(torch.from_numpy(c1))#在c1即旋转角度
          z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
          x_save = self.G(z) #保存G的生成图
          save_image(x_save.data, './tmp/c1.png', nrow=10)

          con_c.data.copy_(torch.from_numpy(c2))#在c2宽度
          z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
          x_save = self.G(z)
          save_image(x_save.data, './tmp/c2.png', nrow=10)
