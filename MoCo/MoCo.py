'''
Created on Feb 7, 2021
@author: nakaizura
'''

#MoCo的主要就是以下操作：
#1 维护queue来动态更新。
#2 keys部分单独momentum以解耦batch size。
#3 一个trick：Shuffling BN

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    主要就是这个class来完成的逻辑
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: 特征维度 (default: 128)
        K: 负例序列长度 (default: 65536)
        m: k部分的moco momentum更新率 (default: 0.999)
        T: softmax平滑系数 (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # 创建解码器
        # num_classes输出的类别数量
        self.encoder_q = base_encoder(num_classes=dim)#moco模型图里的正例查询query
        self.encoder_k = base_encoder(num_classes=dim)#moco模型图里的负例key们

        if mlp:  #得到encoder的结果
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化
            param_k.requires_grad = False  # 设置k部分不更新梯度

        # 创建队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        key encoder的Momentum update
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        完成对队列的出队和入队更新
        """
        # 在更新队列前得到keys
        keys = concat_all_gather(keys)#合并所有keys

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # 出队入队完成队列的更新
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # 用来移动的指针

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        moco有一个很重要的trick就是会使用Shuffling BN，即打乱再BN
        只支持多gpu的DDP打乱 DistributedDataParallel (DDP) model. ***
        """
        # 从所有gpus中gather x
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # 随机打乱
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # 再播报回所有的gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # restoring序号
        idx_unshuffle = torch.argsort(idx_shuffle)

        # 更新到每个gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        还原batch shuffle.
        """
        # 从所有gpus中gather x
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # 还原到每个gpu的数据序号
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):

        # 计算query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # 计算key features
        with torch.no_grad():  # 对于keys是没有梯度的反向的
            self._momentum_update_key_encoder()  # 用自己的来更新key encoder

            # 执行shuffle BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # 还原shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # 计算概率
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #用爱因斯坦求和来算sum
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # 平滑softmax的分布，T越大越平
        logits /= self.T

        # labels是正例index
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # 出队入队更新
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    得到分布式设备中的所有tensor
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
