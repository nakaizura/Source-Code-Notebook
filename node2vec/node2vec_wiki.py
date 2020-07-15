'''
Created on Jul 15, 2020
@author: nakaizura
'''
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

#networkx是专门用来存储图，构建图和分析图的库，操作真的超级方便。

def evaluate_embeddings(embeddings):
    #读入真实的分类label
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8 #80%的节点用于训练分类器，其余的用于测试
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    #应用分类器对节点进行分类以评估向量的质量
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)#用TSNE进行降维
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)): 
        color_idx.setdefault(Y[i][0], []) #类别
        color_idx[Y[i][0]].append(i) #id

    for c, idx in color_idx.items(): #不同类别不同颜色
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #读入边列表，文件中的每一行有两个节点，表示连接这两个节点的边。
    #直接用networkx读入就行，很方便的操作。
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    #实例化模型，“句子”长度为10，80次游走等。重要的参数是p=0.25, q=4
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size=5, iter=3) #训练模型，关于gensim w2v的参数都默认在train里面
    embeddings = model.get_embeddings() #得到Embedding向量

    evaluate_embeddings(embeddings) #应用节点分类来评估嵌入向量的质量
    plot_embeddings(embeddings) #降成二维画在图中可视化
