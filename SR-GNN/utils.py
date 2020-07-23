'''
Created on Jul 22, 2020
@author: nakaizura
'''
import networkx as nx
import numpy as np

#networkx是专门用来存储图，构建图和分析图的库，操作真的超级方便。

def build_graph(train_data):
    #构图
    graph = nx.DiGraph()#Digraph是有向图的基类
    for seq in train_data: #对于一个session序列
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None: #遍历相邻的节点
                weight = 1 #如果两个节点之间没有边，那么设置为1
            else: #如果存在，那么给边权重增加1
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)#添加修改后的边
    for node in graph.nodes: #遍历所有的节点
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i): 
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    #统一user的session的长度为最长的那个，其他的地方补item_tail，但是同时需要mask来标记0
    us_lens = [len(upois) for upois in all_usr_pois]#所有的长度
    len_max = max(us_lens) #选最大
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    #切分数据集
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32') #item编号
    np.random.shuffle(sidx) #打乱
    n_train = int(np.round(n_samples * (1. - valid_portion)))#采样比率
    #切分为验证集和训练集
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data(): #处理数据，主要是得到batch和建立邻接矩阵
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)#编号
            np.random.shuffle(shuffled_arg)#打乱顺序
            self.inputs = self.inputs[shuffled_arg]#input
            self.mask = self.mask[shuffled_arg]#mask
            self.targets = self.targets[shuffled_arg]#target
        n_batch = int(self.length / batch_size)#切分batch
        if self.length % batch_size != 0:
            n_batch += 1 #不能完全除尽的其余部分也要算作一个batch
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)#最大的session的item数目
        for u_input in inputs:
            node = np.unique(u_input)#unique的item
            items.append(node.tolist() + (max_n_node - len(node)) * [0])#不够的补0
            u_A = np.zeros((max_n_node, max_n_node))#user的邻接矩阵
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0: #为0说明这个session已经结束了
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            #最终想要的邻接矩阵A是入度和出度A(in)和A(out)矩阵拼接而成的(n, 2n)的矩阵
            u_sum_in = np.sum(u_A, 0) #按0维度sum，即入度总数
            u_sum_in[np.where(u_sum_in == 0)] = 1 #防止没有某节点没有入度而除了0
            u_A_in = np.divide(u_A, u_sum_in) #平均一下
            u_sum_out = np.sum(u_A, 1) #同理按1sum，算一下出度
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out) #需要转置一下
            u_A = np.concatenate([u_A_in, u_A_out]).transpose() #最后拼接两者
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
