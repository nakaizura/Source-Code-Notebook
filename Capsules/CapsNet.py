'''
Created on May 3, 2020
@author: nakaizura
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms

USE_CUDA = True #gpu


#载入Mnist数据
class Mnist:
    def __init__(self, batch_size):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)
        
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


#按模型框架图，先由普通卷积层再PrimaryCaps，最后映射到DigitCaps。
#普通卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        #9x9卷积，256个通道，输出的大小是20x20x256
        #大一些的感受野能在层数较少的情况下得到更多的信息
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        return F.relu(self.conv(x))

#Primarycaps卷积
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        #32个平行的卷积，每个数据为8个分量
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) 
                          for _ in range(num_capsules)])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]#num_capsules个卷积层
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)#窗口大小是6x6
        return self.squash(u)#squash激活函数挤压向量
    
    def squash(self, input_tensor):
        #实现0-1的压缩，同时保持其方向不变
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

#DigitCaps胶囊层，最后输出为10个16分量的向量，分类结果是向量长度最大的输出
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        #先计算中间向量u
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)#输入x通过W进行空间映射，编码图像中低级和高级特征之间的空间关系

        #b的初始化为0
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        #动态路由
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij) #用b计算softmax的权重c
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)#加权和
            v_j = self.squash(s_j)#当前迭代的输出
            
            if iteration < num_iterations - 1: #更新a和b
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)#最后的输出
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


#重构函数，强制网络保留所有重建图像所需要的信息
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #从DigitCaps的16x10开始重建完整图片
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)#最后输出的x向量最长的为最后的结果
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))#one-hot
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        #3层FC做重建
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)
        
        return reconstructions, masked

#CapsNet完整的流程
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        #由四个class组成
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        
        self.mse_loss = nn.MSELoss()#均方差
        
    def forward(self, data):
        #三层的胶囊网络结构得到output
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        #用输出重建图像
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked
    
    def loss(self, data, x, target, reconstructions):
        #完整的loss由margin和reconstruction两部分组成
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)
    
    def margin_loss(self, x, labels, size_average=True):
        #margin loss强制使capsule之间（如预测1和预测2）的差别越来越大
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))#长度表示某个类别的概率

        #上边界和下边界
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        #惩罚偏离边缘（错位的分类对边缘0.1 or 0.9的距离）
        #如果预测是0.8，label是1，那么loss是0.1很小
        #如果label是0，那么loss的惩罚要算与right的距离，其中0.5是downweight
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss
    
    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005 #这个系数为0.0005


capsule_net = CapsNet()
if USE_CUDA:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters())#优化器


batch_size = 100
mnist = Mnist(batch_size)#导入数据

n_epochs = 30#周期

#开始训练
for epoch in range(n_epochs):
    capsule_net.train()#调到训练模式
    train_loss = 0
    for batch_id, (data, target) in enumerate(mnist.train_loader):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)#手写数字10维
        data, target = Variable(data), Variable(target)

        if USE_CUDA:#gpu
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad() #梯度清零
        output, reconstructions, masked = capsule_net(data) #得到模型输出
        loss = capsule_net.loss(data, output, target, reconstructions) #计算loss
        loss.backward() #反向传播
        optimizer.step() #参数更新

        train_loss += loss.data[0]#记录总loss
        
        if batch_id % 100 == 0: #定期打印结果
            print "train accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size)
        
    print train_loss / len(mnist.train_loader)
        
    capsule_net.eval()#评估模式
    test_loss = 0
    for batch_id, (data, target) in enumerate(mnist.test_loader):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:#gpu
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)#得到评估结果
        loss = capsule_net.loss(data, output, target, reconstructions)#计算loss

        test_loss += loss.data[0]
        
        if batch_id % 100 == 0: #定期打印结果
            print "test accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size)
    
    print test_loss / len(mnist.test_loader)


#可视化
import matplotlib
import matplotlib.pyplot as plt

def plot_images_separately(images):
    "Plot the six MNIST images separately."
    fig = plt.figure()
    for j in xrange(1, 7):
        ax = fig.add_subplot(1, 6, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

plot_images_separately(data[:6,0].data.cpu().numpy())#原结果
plot_images_separately(reconstructions[:6,0].data.cpu().numpy())#重建结果
