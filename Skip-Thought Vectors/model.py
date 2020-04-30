'''
Created on Apr 30, 2020
@author: nakaizura
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *


class Encoder(nn.Module):
    #编码器
    thought_size = 1200
    word_size = 620

    @staticmethod
    def reverse_variable(var):#翻转读取
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if USE_CUDA:
            idx = idx.cuda(CUDA_DEVICE)#gpu

        inverted_var = var.index_select(0, idx)
        return inverted_var

    def __init__(self):
        #本质上就是一个词嵌入后再lstm
        super().__init__()
        self.word2embd = nn.Embedding(VOCAB_SIZE, self.word_size)
        self.lstm = nn.LSTM(self.word_size, self.thought_size)

    def forward(self, sentences):
        # sentences = (batch_size, maxlen), with padding on the right.

        sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)，输入句子

        word_embeddings = F.tanh(self.word2embd(sentences))  # (maxlen, batch_size, word_size)，得到嵌入

        # The following is a hack: We read embeddings in reverse. This is required to move padding to the left.
        # If reversing is not done then the RNN sees a lot a garbage values right before its final state.
        # This reversing also means that the words will be read in reverse. But this is not a big problem since
        # several sequence to sequence models for Machine Translation do similar hacks.
        #反向读取句子是个trick（句子长度没有maxlen的会被填充，这样rnn会有很多的garbage）
        rev = self.reverse_variable(word_embeddings)

        _, (thoughts, _) = self.lstm(rev)
        thoughts = thoughts[-1]  # (batch, thought_size)

        return thoughts, word_embeddings


class DuoDecoder(nn.Module):

    word_size = Encoder.word_size

    def __init__(self):
        #解码器。模型预测某句子的前一个句子和后一个句子
        super().__init__()
        self.prev_lstm = nn.LSTM(Encoder.thought_size + self.word_size, self.word_size)#前
        self.next_lstm = nn.LSTM(Encoder.thought_size + self.word_size, self.word_size)#后
        self.worder = nn.Linear(self.word_size, VOCAB_SIZE)

    def forward(self, thoughts, word_embeddings):
        # thoughts = (batch_size, Encoder.thought_size)
        # word_embeddings = # (maxlen, batch, word_size)

        # We need to provide the current sentences's embedding or "thought" at every timestep.
        thoughts = thoughts.repeat(MAXLEN, 1, 1)  # (maxlen, batch, thought_size)
        #复制thoughts然后移位，得到对应的句子，前一个句子，后一个句子。
        # Prepare Thought Vectors for Prev. and Next Decoders.
        prev_thoughts = thoughts[:, :-1, :]  # (maxlen, batch-1, thought_size)
        next_thoughts = thoughts[:, 1:, :]   # (maxlen, batch-1, thought_size)

        # Teacher Forcing.
        #   1.)前后句子也嵌入
        prev_word_embeddings = word_embeddings[:, :-1, :]  # (maxlen, batch-1, word_size)
        next_word_embeddings = word_embeddings[:, 1:, :]  # (maxlen, batch-1, word_size)
        #   2.)延后一个时间步，拼接
        delayed_prev_word_embeddings = torch.cat([0 * prev_word_embeddings[-1:, :, :], prev_word_embeddings[:-1, :, :]])
        delayed_next_word_embeddings = torch.cat([0 * next_word_embeddings[-1:, :, :], next_word_embeddings[:-1, :, :]])

        #开始lstm编码句子语义
        prev_pred_embds, _ = self.prev_lstm(torch.cat([next_thoughts, delayed_prev_word_embeddings], dim=2))  # (maxlen, batch-1, embd_size)
        next_pred_embds, _ = self.next_lstm(torch.cat([prev_thoughts, delayed_next_word_embeddings], dim=2))  # (maxlen, batch-1, embd_size)

        #开始预测实际的词
        a, b, c = prev_pred_embds.size()
        prev_pred = self.worder(prev_pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-1, VOCAB_SIZE)
        a, b, c = next_pred_embds.size()
        next_pred = self.worder(next_pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-1, VOCAB_SIZE)

        prev_pred = prev_pred.transpose(0, 1).contiguous()  # (batch-1, maxlen, VOCAB_SIZE)
        next_pred = next_pred.transpose(0, 1).contiguous()  # (batch-1, maxlen, VOCAB_SIZE)

        return prev_pred, next_pred


class UniSkip(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()#编码器
        self.decoders = DuoDecoder()#解码器

    def create_mask(self, var, lengths):
        mask = var.data.new().resize_as_(var.data).fill_(0)
#         print("lengths", lengths)
        for i, l in enumerate(lengths):
            for j in range(l):
                mask[i, j] = 1
        
        mask = Variable(mask)
        if USE_CUDA:
            mask = mask.cuda(var.get_device())
            
        return mask

    def forward(self, sentences, lengths):
        # sentences = (B, maxlen)
        # lengths = (B)

        #编码每个句子的向量
        thoughts, word_embeddings = self.encoder(sentences)  # thoughts = (B, thought_size), word_embeddings = (B, maxlen, word_size)

        #预测前一个和下一个句子
        prev_pred, next_pred = self.decoders(thoughts, word_embeddings)  # both = (batch-1, maxlen, VOCAB_SIZE)

        # mask the predictions, so that loss for beyond-EOS word predictions is cancelled.
        #在预测时，还没出现的词的loss不能计算，所以要通过mask。
        prev_mask = self.create_mask(prev_pred, lengths[:-1])
        next_mask = self.create_mask(next_pred, lengths[1:])

        #mask为0的不用计算
        masked_prev_pred = prev_pred * prev_mask
        masked_next_pred = next_pred * next_mask

        #计算交叉熵
        prev_loss = F.cross_entropy(masked_prev_pred.view(-1, VOCAB_SIZE), sentences[:-1, :].view(-1))
        next_loss = F.cross_entropy(masked_next_pred.view(-1, VOCAB_SIZE), sentences[1:, :].view(-1))

        #总loss
        loss = prev_loss + next_loss
        
        _, prev_pred_ids = prev_pred[0].max(1)
        _, next_pred_ids = next_pred[0].max(1)

        return loss, sentences[0], sentences[1], prev_pred_ids, next_pred_ids
