import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

# Graph Transformer算是GAT的升级版，注意力部分使用自注意力，有残差跳跃等类似Transformer的结构
# 这份代码提供了丰富的选择，比如是否有边信息，残差怎么做等等。

class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 ansfo: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels # 输入维度
        self.out_channels = out_channels # 输出维度
        self.me = heads # 注意力头数
        self.beta = beta and root_weight # 是否有残差跳跃，beta是合并系数
        self.root_weight = root_weight # 是否残差
        self.concat = concat # Ture代表多头拼，False是多头平均
        self.dropout = dropout # 采样邻居时的dropout
        self.edge_dim = edge_dim # 是否有边特征的维度，默认为None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            
        # 得到KQV
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None: # 如果有边特征，边特征要加入到K中进行运算
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:# 两种concat形式
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)# 直接拼时需要扩展head维度
            if self.beta:# 残差合并系数
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):# 重置参数
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        """"""

        if isinstance(x, Tensor):# 保证成对输入
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)# 横拼
        else:
            out = out.mean(dim=1)# 平均

        if self.root_weight:# 残差合并
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:# 如果有边特征，要加到key中
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        # 算注意力
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
