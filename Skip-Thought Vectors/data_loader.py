'''
Created on Apr 30, 2020
@author: nakaizura
'''

import torch
from torch.autograd import Variable
from vocab import *
from config import *
import numpy as np
import random

np.random.seed(0)#可复现随机种子

#载入数据

class DataLoader:
    EOS = 0  #end of sentence，句子的结尾
    UNK = 1  #unknown token，未知的词

    maxlen = MAXLEN #最大长度

    def __init__(self, text_file=None, sentences=None, word_dict=None):

        if text_file:#读句子文件生成单词字典
            print("Loading text file at {}".format(text_file))
            with open(text_file, "rt") as f:
                sentences = f.readlines()
            print("Making dictionary for these words")
            word_dict = build_and_save_dictionary(sentences, source=text_file)#单词字典

        assert sentences and word_dict, "Please provide the file to extract from or give sentences and word_dict"

        self.sentences = sentences
        self.word_dict = word_dict
        print("Making reverse dictionary")
        self.revmap = list(self.word_dict.items())

        self.lengths = [len(sent) for sent in self.sentences]

    def convert_sentence_to_indices(self, sentence):
        #句子编码
        indices = [
                      #编码int给每个词，如果词太稀疏了就设置为UNK(unknown token)
                      self.word_dict.get(w) if self.word_dict.get(w, VOCAB_SIZE + 1) < VOCAB_SIZE else self.UNK

                      for w in sentence.split()  #用空格分出所有词，然后得到词的数量
                  ][: self.maxlen - 1]  #最多maxlen的长度

        #最后一个单词设置为EOS
        indices += [self.EOS] * (self.maxlen - len(indices))

        indices = np.array(indices)
        indices = Variable(torch.from_numpy(indices))
        if USE_CUDA:
            indices = indices.cuda(CUDA_DEVICE) #gpu

        return indices

    def convert_indices_to_sentences(self, indices):
        #由编码到句子
        def convert_index_to_word(idx):

            idx = idx.data[0]
            if idx == 0:#根据idx返回对应的词
                return "EOS"
            elif idx == 1:
                return "UNK"
            
            search_idx = idx - 2 #然后按revmap出word
            if search_idx >= len(self.revmap):
                return "NA"
            
            word, idx_ = self.revmap[search_idx]

            assert idx_ == idx
            return word

        words = [convert_index_to_word(idx) for idx in indices]

        return " ".join(words)#用空格拼接所有的词成句子

    def fetch_batch(self, batch_size):
        #得到batch。先随机选开头
        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):#再挑够数目
            sent = self.sentences[i]#一整个句子
            ind = self.convert_sentence_to_indices(sent)
            batch.append(ind)
            lengths.append(min(len(sent.split()), MAXLEN))

        batch = torch.stack(batch)#shape一下变堆叠
        lengths = np.array(lengths)

        return batch, lengths
