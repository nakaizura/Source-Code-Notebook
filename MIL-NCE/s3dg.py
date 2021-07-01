'''
Created on Jul 1, 2021
@author: nakaizura
'''

# 抽特征Gated Separable 3D network (S3D-G).


import torch as th
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import re

class InceptionBlock(nn.Module):# 基于Inception的架构，如果不分离就退化成I3D。

    def __init__(self, input_dim,
                num_outputs_0_0a,
                num_outputs_1_0a,
                num_outputs_1_0b,
                num_outputs_2_0a,
                num_outputs_2_0b,
                num_outputs_3_0b,
                gating=True):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3],
                                    padding=1, separable=True)
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3],
                                    padding=1, separable=True)
        self.maxpool_b3 = th.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = num_outputs_0_0a + num_outputs_1_0b +\
                    num_outputs_2_0b + num_outputs_3_0b
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, input):
      """Inception block
      """
      b0 = self.conv_b0(input)
      b1 = self.conv_b1_a(input)
      b1 = self.conv_b1_b(b1)
      b2 = self.conv_b2_a(input)
      b2 = self.conv_b2_b(b2)
      b3 = self.maxpool_b3(input)
      b3 = self.conv_b3_b(b3)
      if self.gating:
          b0 = self.gating_b0(b0)
          b1 = self.gating_b1(b1)
          b2 = self.gating_b2(b2)
          b3 = self.gating_b3(b3)
      return th.cat((b0, b1, b2, b3), dim=1)

class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
      """Feature gating as used in S3D-G.
      """
      spatiotemporal_average = th.mean(input_tensor, dim=[2, 3, 4])
      weights = self.fc(spatiotemporal_average)
      weights = th.sigmoid(weights)
      return weights[:, :, None, None, None] * input_tensor

class STConv3D(nn.Module):

    def __init__(self,
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=0,
                separable=False):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
              spatial_stride = [1, stride[1], stride[2]]
              temporal_stride = [stride[0], 1, 1]
            else:
              spatial_stride = [1, stride, stride]
              temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
              spatial_padding = [0, padding[1], padding[2]]
              temporal_padding = [padding[0], 0, 0]
            else:
              spatial_padding = [0, padding, padding]
              temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(input_dim, output_dim,
                                   kernel_size=spatial_kernel_size,
                                   stride=spatial_stride,
                                   padding=spatial_padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(output_dim, output_dim,
                                   kernel_size=temporal_kernel_size,
                                   stride=temporal_stride,
                                   padding=temporal_padding, bias=False)
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(input_dim, output_dim,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)


    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


class MaxPool3dTFPadding(th.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = th.nn.ConstantPad3d(padding_shape, 0)
        self.pool = th.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out

class Sentence_Embedding(nn.Module):# 抽取句子特征
    def __init__(self,
                 embd_dim,
                 token_to_word_path,
                 num_embeddings=66250,
                 word_embedding_dim=300,
                 word2vec_path='',
                 max_words=16,
                 output_dim=2048):
        super(Sentence_Embedding, self).__init__()
        if word2vec_path:# 得到词向量
            self.word_embd = nn.Embedding.from_pretrained(th.load(word2vec_path)) 
        else:
            self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):# 补0
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

    def _split_text(self, sentence):# 分词
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):# token化
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def words_to_ids(self, x):# 词到id
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return th.stack(split_x, dim=0)

    def forward(self, x, raw_text=False):
        if raw_text:
            x = self.words_to_ids(x)
        with th.no_grad():
            x = self.word_embd(x)
        x = F.relu(self.fc1(x), inplace=True)# fc之后接max得到句子的特征
        x = th.max(x, dim=1)[0]
        x = self.fc2(x)
        return x


class S3D(nn.Module):

    def __init__(self, num_classes=512, gating=True, space_to_depth=False,
                  word2vec_path='', init='uniform', token_to_word_path='data/dict.npy'):
        super(S3D, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        if space_to_depth:
            self.conv1 = STConv3D(
                        24, 64, [2, 4, 4], stride=1, padding=(1, 2, 2), separable=False)
        else:
            self.conv1 = STConv3D(
                        3, 64, [3, 7, 7], stride=2, padding=(1, 3, 3), separable=False)
        self.conv_2b = STConv3D(64, 64, [1, 1, 1], separable=False)
        self.conv_2c = STConv3D(64, 192, [3, 3, 3], padding=1, separable=True)
        self.gating = SelfGating(192)
        self.maxpool_2a = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        self.maxpool_3a = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionBlock(self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64)
        self.maxpool_4a = MaxPool3dTFPadding(
                        kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
        self.mixed_4b = InceptionBlock(self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64)
        self.mixed_4c = InceptionBlock(self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64)
        self.mixed_4d = InceptionBlock(self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64)
        self.mixed_4e = InceptionBlock(self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64)
        self.mixed_4f = InceptionBlock(self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128)
        self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
                        kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')
        self.mixed_5b = InceptionBlock(self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = InceptionBlock(self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128)
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
        self.text_module = Sentence_Embedding(
                               num_classes,
                               os.path.join(os.path.dirname(__file__), token_to_word_path),
                               word2vec_path=os.path.join(os.path.dirname(__file__), word2vec_path))

        if init == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_in',
                                            nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _space_to_depth(self, input):
      B, C, T, H, W = input.shape
      input = input.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
      input = input.permute(0, 3, 5, 7, 1,  2, 4, 6)
      input = input.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
      return input

    def forward(self, video, text, mode='all', mixed5c=False):
      if mode == 'all':
          return self.forward_video(video), self.text_module(text)
      elif mode == 'video':
          return self.forward_video(video, mixed5c=mixed5c)
      elif mode == 'text':
          return self.text_module(text)
      else:
          raise NotImplementedError

    def forward_video(self, inputs, mixed5c=False):# S3D的在I3D的基础上多了mix部分。
      #out = {}
      if self.space_to_depth:
        inputs = self._space_to_depth(inputs)
      # 'Conv2d_1a_7x7'
      net = self.conv1(inputs)
      if self.space_to_depth:
        net = net[:, :, 1:, 1:, 1:]
      #out['Conv2d_1a_7x7'] = net
      # 'MaxPool_2a_3x3'
      net = self.maxpool_2a(net)
      #out['MaxPool_2a_3x3'] = net
      #'Conv2d_2b_1x1'
      net = self.conv_2b(net)
      #out['Conv2d_2b_1x1'] = net
      # 'Conv2d_2c_3x3'
      net = self.conv_2c(net)
      #out['Conv2d_2c_3x3'] = net
      if self.gating:
          net = self.gating(net)
          #out['gating_1'] = net
      # 'MaxPool_3a_3x3'
      net = self.maxpool_3a(net)
      #out['MaxPool_3a_3x3'] = net
      # end_point = 'Mixed_3b'
      net = self.mixed_3b(net)
      #out['Mixed_3b'] = net
      # end_point = 'Mixed_3c'
      net = self.mixed_3c(net)
      #out['Mixed_3c'] = net
      # end_point = 'MaxPool_4a_3x3'
      net = self.maxpool_4a(net)
      #out['MaxPool_4a_3x3'] = net
      # end_point = 'Mixed_4b'
      net = self.mixed_4b(net)
      #out['Mixed_4b'] = net
      # end_point = 'Mixed_4c'
      net = self.mixed_4c(net)
      #out['Mixed_4c'] = net
      # end_point = 'Mixed_4d'
      net = self.mixed_4d(net)
      #out['Mixed_4d'] = net
      # end_point = 'Mixed_4e'
      net = self.mixed_4e(net)
      #out['Mixed_4e'] = net
      # end_point = 'Mixed_4f'
      net = self.mixed_4f(net)
      #out['Mixed_4f'] = net
      #end_point = 'MaxPool_5a_2x2'
      net = self.maxpool_5a(net)
      #out['MaxPool_5a_2x2'] = net
      # end_point = 'Mixed_5b'
      net = self.mixed_5b(net)
      #out['Mixed_5b'] = net
      # end_point = 'Mixed_5c'
      net = self.mixed_5c(net)
      #out['Mixed_5c'] = net
      #out['Avgpool'] = net
      net = th.mean(net, dim=[2, 3, 4])
      if mixed5c:
          return net
      net = self.fc(net)
      #out['final'] = net
      return net
