'''
Created on Feb 7, 2021
@author: nakaizura
'''

#GAE思路比较简单，大概就是用中间隐特征z来重建Graph，具体可以看博文，不赘述了。


import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

from ..inits import reset

EPS = 1e-15 #预测概率的控制值，以免求log的时候有问题
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    r"""这内积解码器，即将隐层表示Z内积之后来重建原来的Graph
    值得注意的，有两个forward可以分别hold住全部重建和只对局部采样重建"""
    def forward(self, z, edge_index, sigmoid=True):
        #计算节点对之间存在边的概率
        #edge_index分别存的邻接矩阵的行和列，所以取0和1直接可计算
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        
        #Sigmoid控制是否非线性
        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z, sigmoid=True):
        #计算所有节点，所以是按照公式直接内积
        adj = torch.matmul(z, z.t())

        #Sigmoid控制是否非线性
        return torch.sigmoid(adj) if sigmoid else adj



class GAE(torch.nn.Module):
    r"""GAE的代码"""
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder #这里encoder的设置可以就是普通的GCN或者其他模型
        self.decoder = InnerProductDecoder() if decoder is None else decoder #decoder是上面的class
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)


    def encode(self, *args, **kwargs):
        #这里可以就放入GCN来得到z
        return self.encoder(*args, **kwargs)


    def decode(self, *args, **kwargs):
        #根据z计算边概率来重建Graph
        return self.decoder(*args, **kwargs)


    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""计算重建损失，这里会使用正负例采样来计算交叉熵"""

        #求正例的重建分数，这里看decoder的输出可以知道调用的是采样版的forward
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        #负例中不含自环（self-loops），所以先添进去方便后面负采样
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        
        if neg_edge_index is None: #负采样得到负例的index
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        #计算负例的重建分数
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss #加和得到总loss


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1)) #正例的y为1
        neg_y = z.new_zeros(neg_edge_index.size(1)) #负例的y为0
        y = torch.cat([pos_y, neg_y], dim=0) #这是真实标签

        #得到GAE的预测结果
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy() #detach出来算指标

        return roc_auc_score(y, pred), average_precision_score(y, pred) #计算AUC和AP
