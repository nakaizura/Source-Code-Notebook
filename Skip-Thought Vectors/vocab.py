'''
Created on Apr 30, 2020
@author: nakaizura
'''
import _pickle as pkl
from collections import OrderedDict
import argparse


def build_dictionary(text):
    """建立一个词典
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:#不在词典就新建
                wordcount[w] = 0
            wordcount[w] += 1#加就计数

    #按词出现的频率排序
    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    worddict = OrderedDict()
    for idx, word in enumerate(sorted_words):#必须加2，因为0和1两个词已经余弦定义了。
        worddict[word] = idx+2 # 0: 句子结尾<eos>, 1: 未知词<unk>

    return worddict, wordcount


def load_dictionary(loc='./data/book_dictionary_large.pkl'):
    """载入字典
    Load a dictionary
    """
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
    return worddict


def save_dictionary(worddict, wordcount, loc='./data/book_dictionary_large.pkl'):
    """保存字典
    Save a dictionary to the specified location
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


def build_and_save_dictionary(text, source):
    save_loc = source+".pkl"
    try:
        cached = load_dictionary(save_loc)
        print("Using cached dictionary at {}".format(save_loc))
        return cached
    except:
        pass
    # build again and save
    print("unable to load from cached, building fresh")
    worddict, wordcount = build_dictionary(text)
    print("Got {} unique words".format(len(worddict)))
    print("Saveing dictionary at {}".format(save_loc))
    save_dictionary(worddict, wordcount, save_loc)
    return worddict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=str)
    args = parser.parse_args()

    print("Extracting text from {}".format(args.text_file))
    text = open(args.text_file, "rt").readlines()
    print("Extracting dictionary..")
    worddict, wordcount = build_dictionary(text)

    out_file = args.text_file+".pkl"
    print("Got {} unique words. Saving to file {}".format(len(worddict), out_file))
    save_dictionary(worddict, wordcount, out_file)
    print("Done.")
