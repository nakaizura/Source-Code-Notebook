'''
Created on Mar 23, 2021
@author: nakaizura
'''

import math
import torch
import random
import numpy as np 
from torch import nn
import torch.nn.functional as F

from utils import sos_idx, eos_idx

#模型的架构是s2s的，encode是LSTM，decode也是LSTM。
#为了融合video，所以在输到decode的时候会融合当前词+源句子+视频特征的注意力。


class SoftDotAttention(nn.Module):
    def __init__(self, dim_ctx, dim_h):
        '''初始化层注意力层'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim_h, dim_ctx, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, context, h, mask=None):
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # 计算attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        weighted_ctx = torch.bmm(attn3, context) # batch x dim
        return weighted_ctx, attn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.src_embed = nn.Embedding(vocab_size, embed_size) #句子embedding
        self.src_encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size // 2, num_layers=n_layers,
                                     dropout=dropout, batch_first=True, bidirectional=True) #多层LSTM

        self.frame_embed = nn.Linear(1024, self.embed_size) #视频embedding
        self.video_encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size // 2, num_layers=n_layers,
                                     dropout=dropout, batch_first=True, bidirectional=True) #多层LSTM

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, src, vid, src_hidden=None, vid_hidden=None):
        batch_size = src.size(0)

        src_embedded = self.src_embed(src) #嵌入
        src_out, src_states = self.src_encoder(src_embedded, src_hidden) #过LSTM得到隐层
        # 2层LSTM的结果
        src_h = src_states[0].permute(1, 0, 2).contiguous().view(
            batch_size, 2, -1).permute(1, 0, 2) #前两维换一下
        src_c = src_states[1].permute(1, 0, 2).contiguous().view(
            batch_size, 2, -1).permute(1, 0, 2)

        vid_embedded = self.frame_embed(vid) #嵌入
        vid_out, vid_states = self.video_encoder(vid_embedded, vid_hidden) ##过LSTM得到隐层
        # 2层LSTM的结果
        vid_h = vid_states[0].permute(1, 0, 2).contiguous().view(
            batch_size, 2, -1).permute(1, 0, 2) 
        vid_c = vid_states[0].permute(1, 0, 2).contiguous().view(
            batch_size, 2, -1).permute(1, 0, 2)

        init_h = torch.cat((src_h, vid_h), 2) #在最后一维特征维cat视频和文本，之后会用它做attention
        init_c = torch.cat((src_c, vid_c), 2)

        return src_out, (init_h, init_c), vid_out


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,
                 n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size) #embedding
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.src_attention = SoftDotAttention(embed_size, hidden_size) #src的attention
        self.vid_attention = SoftDotAttention(embed_size, hidden_size) #video的attention

        self.decoder = nn.LSTM(embed_size*3, hidden_size,
                          n_layers, dropout=dropout, batch_first=True) #3倍：当前词+源句子+视频

        self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.embed_size),
                                   nn.Tanh(),
                                   nn.Dropout(p=dropout),
                                   nn.Linear(embed_size, vocab_size))

    def onestep(self, input, last_hidden, src_out, vid_out, src_mask):
        '''
        input: (B,)
        '''
        # 得到当前词的嵌入 (即前一个的输出词)
        embedded = self.embed(input).unsqueeze(1)  # (B,1, N)
        embedded = self.dropout(embedded)
        # 计算上下文注意力
        src_ctx, src_attn = self.src_attention(src_out, last_hidden[0][0], mask=src_mask) # src_ctx: (mb, 1, dim) attn: (mb, 1, seqlen)
        vid_ctx, vid_attn = self.vid_attention(vid_out, last_hidden[0][0])
        # 拼接三者
        rnn_input = torch.cat([embedded, src_ctx, vid_ctx], 2) # (mb, 1, input_size)

        output, hidden = self.decoder(rnn_input, last_hidden) #再LSTM解码
        output = output.squeeze(1)  # (B, 1, N) -> (B,N)
        output = self.fc(output)
        return output, hidden, (src_attn, vid_attn)

    def forward(self, src, trg, init_hidden, src_out, vid_out, max_len, teacher_forcing_ratio):
        #普通搜索
        batch_size = trg.size(0)
        src_mask = (src == 0) # mask paddings.

        outputs = torch.zeros(batch_size, max_len, self.vocab_size).cuda()

        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())
        output = trg.data[:, 0] # <sos>
        for t in range(1, max_len):
            output, hidden, attn_weights = self.onestep(output, hidden, src_out, vid_out, src_mask) # (mb, vocab) (1, mb, N) (mb, 1, seqlen)
            outputs[:, t, :] = output
            is_teacher = random.random() < teacher_forcing_ratio #一定比例top1
            top1 = output.data.max(1)[1]
            output = (trg.data[:, t] if is_teacher else top1).cuda() # 选择输出作为下一次的当前词嵌入
        return outputs

    def inference(self, src, trg, init_hidden, src_out, vid_out, max_len, teacher_forcing_ratio=0):
        #贪心搜索
        batch_size = trg.size(0)
        src_mask = (src == 0) # mask paddings.

        outputs = torch.zeros(batch_size, max_len, self.vocab_size).cuda()

        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())
        output = trg.data[:, 0] # <sos>
        pred_lengths = [0]*batch_size
        for t in range(1, max_len):
            output, hidden, attn_weights = self.onestep(output, hidden, src_out, vid_out, src_mask) # (mb, vocab) (1, mb, N) (mb, 1, seqlen)
            outputs[:, t, :] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]

            output = (trg.data[:, t] if is_teacher else top1).cuda()

            for i in range(batch_size):
                if output[i]==3 and pred_lengths[i]==0:
                    pred_lengths[i] = t
        for i in range(batch_size):
            if pred_lengths[i]==0:
                pred_lengths[i] = max_len
        return outputs, pred_lengths

    def beam_decoding(self, src, init_hidden, src_out, vid_out, max_len, beam_size=5):
        #束搜素
        batch_size = src.size(0)
        src_mask = (src == 0) # mask padding
        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())

        seq = torch.LongTensor(max_len, batch_size).zero_()
        seq_log_probs = torch.FloatTensor(max_len, batch_size)

        for i in range(batch_size):
            # treat the problem as having a batch size of beam_size
            src_out_i = src_out[i].unsqueeze(0).expand(beam_size, src_out.size(1), src_out.size(2)).contiguous() # (bs, seq_len, N)
            vid_out_i = vid_out[i].unsqueeze(0).expand(beam_size, vid_out.size(1), vid_out.size(2)).contiguous()
            src_mask_i = src_mask[i].unsqueeze(0).expand(beam_size, src_mask.size(1)).contiguous()
            hidden_i = [_[:, i, :].unsqueeze(1).expand(_.size(0), beam_size, _.size(2)).contiguous() for _ in
                            hidden] # (n_layers, bs, 1024)
            
            output = torch.LongTensor([sos_idx] * beam_size).cuda()
            
            output, hidden_i, attn_weights = self.onestep(output, hidden_i, src_out_i, vid_out_i, src_mask_i) # (mb, vocab) (1, mb, N) (mb, 1, seqlen)
            log_probs = F.log_softmax(output, dim=1)
            log_probs[:, -1] = log_probs[:, -1] - 1000
            neg_log_probs = -log_probs

            all_outputs = np.ones((1, beam_size), dtype='int32')
            all_masks = np.ones_like(all_outputs, dtype="float32")
            all_costs = np.zeros_like(all_outputs, dtype="float32")
            
            for j in range(max_len):
                if all_masks[-1].sum() == 0:
                    break

                next_costs = (
                    all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None])
                (finished,) = np.where(all_masks[-1] == 0)
                next_costs[finished, 1:] = np.inf

                (indexes, outputs), chosen_costs = self._smallest(
                    next_costs, beam_size, only_first_row=j == 0)
                

                new_state_d = [_.data.cpu().numpy()[:, indexes, :]
                               for _ in hidden_i]

                all_outputs = all_outputs[:, indexes]
                all_masks = all_masks[:, indexes]
                all_costs = all_costs[:, indexes]

                output = torch.from_numpy(outputs).cuda()
                hidden_i = self.from_numpy(new_state_d)
                output, hidden_i, attn_weights = self.onestep(output, hidden_i, src_out_i, vid_out_i, src_mask_i)
                log_probs = F.log_softmax(output, dim=1)

                log_probs[:, -1] = log_probs[:, -1] - 1000
                neg_log_probs = -log_probs

                all_outputs = np.vstack([all_outputs, outputs[None, :]])
                all_costs = np.vstack([all_costs, chosen_costs[None, :]])
                mask = outputs != 0
                all_masks = np.vstack([all_masks, mask[None, :]])

            all_outputs = all_outputs[1:]
            all_costs = all_costs[1:] - all_costs[:-1]
            all_masks = all_masks[:-1]
            costs = all_costs.sum(axis=0)
            lengths = all_masks.sum(axis=0)
            normalized_cost = costs / lengths
            best_idx = np.argmin(normalized_cost)
            seq[:all_outputs.shape[0], i] = torch.from_numpy(
                all_outputs[:, best_idx])
            seq_log_probs[:all_costs.shape[0], i] = torch.from_numpy(
                all_costs[:, best_idx])

        seq, seq_log_probs = seq.transpose(0, 1), seq_log_probs.transpose(0, 1)

        pred_lengths = [0]*batch_size
        for i in range(batch_size):
            if sum(seq[i] == eos_idx) == 0:
                pred_lengths[i] = max_len
            else:
                pred_lengths[i] = (seq[i] == eos_idx).nonzero()[0][0]
        # return the samples and their log likelihoods
        return seq, pred_lengths # seq_log_probs 

    def from_numpy(self, states):
        return [torch.from_numpy(state).cuda() for state in states]

    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        return np.unravel_index(args, matrix.shape), flatten[args]
