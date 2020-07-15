'''
Created on Jul 15, 2020
@author: nakaizura
'''
from ..walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd

#逻辑为先随机游走得到一个“句子”，然后直接拿句子，gensim训练向量就行了。

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        #设定一些关于gensim的参数
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0) #词频阈值，这里句子量很少设为0
        kwargs["size"] = embed_size #最后得到128维的节点向量
        kwargs["sg"] = 1  # skip gram的模式来训练
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs) #直接用gensim的模型
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        #得到训练好后的向量
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():#建立一个所有节点的向量索引表
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
