'''
Created on Mar 23, 2021
@author: nakaizura
'''

import os
import sys
import re
import string
import json
import time
from collections import Counter
import numpy as np
import logging

#这个文件主要存一些要用的func。
#比较重要的就是构建词表和完成一些映射操作的Tokenizer类了



# 一些特殊词padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
# 放到词表中的前几个
padding_idx = base_vocab.index('<PAD>')
sos_idx = base_vocab.index('<SOS>')
eos_idx = base_vocab.index('<EOS>')


def set_logger(log_path):
    """
    处理log部分
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 存到文件中
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # 载入到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪防止梯度爆炸
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks某些参数的学习率
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor #乘这个shrink参数就行
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))



### 构建词表，编码句子的class
class Tokenizer(object):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # 不需要非字母数字的东西

    def __init__(self, language, vocab=None, encoding_length=30):
        self.language = language #中文 or 英文
        self.encoding_length = encoding_length #嵌入长度
        self.vocab = vocab #词表
        self.word_to_index = {} #词到索引
        if vocab:
            for i,word in enumerate(vocab): #根据vocab得到词到索引的字典
                self.word_to_index[word] = i

    def split_sentence(self, sentence): #切分句子
        if self.language=='en':
            return self.split_sentence_en(sentence)
        elif self.language=='zh':
            return self.split_sentence_zh(sentence)

    def split_sentence_en(self, sentence):
        ''' 英文按照词语和标定来切分'''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def split_sentence_zh(self, sentence):
        ''' 中文按照字符 '''
        toks = []
        for char in sentence.strip():
            toks.append(char)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence): # 把word变成index里面的id
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>']) #不在表内就<UNK>
        ## <EOS>
        if len(encoding) > self.encoding_length-2:
            encoding = encoding[:self.encoding_length-2]
        ## add <SOS> and <EOS>
        encoding = [self.word_to_index['<SOS>'], *encoding, self.word_to_index['<EOS>']] 
        length = min(self.encoding_length, len(encoding))
        if len(encoding) < self.encoding_length: #不够长度就补<PAD>
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))
        return np.array(encoding[:self.encoding_length]), length


    def encode_sentence_nopad_2str(self, sentence):
        '''编码句子，其不包括<SOS> and padding，这样是真实caption的表示 '''
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence): # 把word变成index里面的id
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(999999) #没有就设个无穷大

        string = ' '.join([str(i) for i in np.array(encoding)])
        return string # exclude <SOS>


    def decode_sentence(self, encoding): #从id回到句子，这个可以方便转录的预测输出
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                if ix >= len(self.vocab):
                    sentence.append('<UNK>')
                else:
                    sentence.append(self.vocab[ix])
        return " ".join(sentence) # 连起来变成句子


def build_vocab(data_dir, language, min_count=5, start_vocab=base_vocab):
    ''' 在有几个特殊词的base上面建立词表 '''
    assert language in ['en', 'zh']
    count = Counter() #计数
    t = Tokenizer(language)#初始化Tokenizer类

    with open(data_dir+'vatex_training_v1.0.json', 'r') as file: #读训练的文件建立词表
        data = json.load(file)
    lan2cap={'en':'enCap', 'zh':'chCap'}
    for d in data:
        for cap in d[lan2cap[language]]:
            count.update(t.split_sentence(cap)) #切分句子然后计数
    vocab = list(start_vocab)
    for word,num in count.most_common(): #只保留一定的表大小，长尾的不看
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path): #写vocab到本地上
    print ('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)

def read_vocab(path): #从本地上读vocab
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab
