'''
Created on Apr 25, 2020
@author: nakaizura
'''

import torch

#torch_geometric这个库似乎很好用。
#这个py文件是一个简易的调库笔记，并且可以快速实验keras版本的cora任务。


from torch_geometric.data import Data
#####data类可以轻松创建Graph

#边索引是COO格式，仅仅用于创建邻接矩阵。
#下面的意思是节点0和1相连，1和0相连，1和2相连，2和1相连
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
#创建3个节点
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

#通过Data类来联系起节点边
data = Data(x=x, edge_index=edge_index)





from torch_geometric.datasets import TUDataset
####和torchvision很像，这个也有很多的数据集

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
print(len(dataset),dataset.num_classes,dataset.num_node_features)


from torch_geometric.data import DataLoader
####同样也有dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch)

#大多数都和pytorch本身的语法很像。



    

#-------------调库大法--------------
#用这个库快速实现keras版本的任务
    
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #直接调GCN

from torch_geometric.datasets import Planetoid

#cora数据集直接调用，任务是做半监督论文分类
dataset = Planetoid(root='./Cora', name='Cora')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #这个和keras的那个模型结构是一样的，16维的隐层，2个GCN
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x) #ReLU激活
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1) #最后做个7分类

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device) #gpu加速
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)#优化器

model.train() #训练模式
for epoch in range(200):
    optimizer.zero_grad() #梯度清零
    out = model(data) #用模型预测结果
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward() #反向传播
    optimizer.step() #更新参数

model.eval() #评估模式
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item() #计算acc
print('Accuracy: {:.4f}'.format(acc))


#acc结果直接0.8+
#调库大法好....




#不过还是看看GCNConv的源码吧
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]，in是node的特征维度
        # edge_index has shape [2, E]，边是两个node之间的关系，所以是2，E是边个数

        #第1步：把自环I加到邻接矩阵
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        #第2步：对节点node特征矩阵线性转换（linear嵌入投影一下）
        x = self.lin(x)

        #第3步：计算正则化矩阵norm
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)#计算度
        deg_inv_sqrt = deg.pow(-0.5) #度的-0.5
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] #然后乘到邻接矩阵，默认无向图的话，行列的度在结果上是一样的

        #第4-6步，开始消息传递
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]，out是嵌入的特征维度

        #第4步：用norm正则化节点特征
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        #第6步：返回新的node特征
        return aggr_out


#该库海量模型可调....
#https://python.ctolib.com/rusty1s-pytorch_geometric.html
