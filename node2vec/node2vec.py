'''
Created on Jul 15, 2020
@author: nakaizura
'''
from ..walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd

#Node2vec 可以看作是对 DeepWalk 的广义抽象，主要是改进DeepWalk的随机游走策略。
#逻辑也为先随机游走得到一个“句子”（P和Q控制），然后直接拿句子，gensim训练向量。

#参数Q控制选择其他的新顶点的概率，偏广度优先，重视局部，即节点重要性
#参数P控制返回原来顶点的概率，偏深度优先，重视全局，即群体重要性。

class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0):

        self.graph = graph
        self._embeddings = {}
        #由p，q控制的游走
        self.walker = RandomWalker(
            graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        #设定一些关于gensim的参数
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0) #词频阈值，这里句子量很少设为0
        kwargs["size"] = embed_size #最后得到128维的节点向量
        kwargs["sg"] = 1  # skip gram的模式来训练
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
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
