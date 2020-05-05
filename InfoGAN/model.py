'''
Created on May 5, 2020
@author: nakaizura
'''

import torch.nn as nn


class FrontEnd(nn.Module):
  '''discriminator and Q 的前端部分，Q通过与D共享卷积层，可以减少计算花销。'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


#判别器，判断输入是真实的还是生成的
class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()
    
    self.main = nn.Sequential(
      nn.Conv2d(1024, 1, 1), 
      nn.Sigmoid() #直接得到1维的预测结果
    )
    

  def forward(self, x):
    output = self.main(x).view(-1, 1)#n个样本，每个样本一个维度，就是预测值
    return output


#直接优化互信息太困难了，所有用辅助分布Q(c|x)来近似
#D判断真假，Q判断类别c
class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()

    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, 10, 1) #10维，因为数据是mnist识别数字
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)

  def forward(self, x):
    #给出一个互信息的下界
    y = self.conv(x)

    disc_logits = self.conv_disc(y).squeeze()#得到10类别分数

    mu = self.conv_mu(y).squeeze() #得到高斯公式的均值mu
    var = self.conv_var(y).squeeze().exp() #得到高斯公式的方差var

    return disc_logits, mu, var 

#生成器，用噪声生成目标数据
class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    self.main = nn.Sequential(
      #噪音74维，62维初始噪音大小，10维控制数字类别的生成，2维的角度和字体宽度
      nn.ConvTranspose2d(74, 1024, 1, 1, bias=False), #反卷积
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False), #7x7反卷积
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),#步长变2加padding
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x) #得到1维的结果。
    return output

#权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) #卷积层用均值0，方差0.02随机初始化
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)#BN用均值1，方差0.02（要正则归一）
        m.bias.data.fill_(0) #偏差用0填充
