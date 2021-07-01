'''
Created on Jul 1, 2021
@author: nakaizura
'''
import torch as th


# 这个就是计算text和video的MIL-NCE了
class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())# 计算相似度
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        # 正例对们
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda() # 对角线即使对应的正例
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1) # 计算log sum exp
        # 正例对们+负例对们
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)# 所以这里多拼接了负例对们
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)# 相减得到定义的loss
