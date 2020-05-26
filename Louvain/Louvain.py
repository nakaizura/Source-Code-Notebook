'''
Created on May 26, 2020
@author: nakaizura
'''
import pandas as pd
import numpy as np
import collections
import string
import random

def load_graph(path):
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()#按空格分列
            v_i = int(vertices[0])
            v_j = int(vertices[1])
            #无向图所以两边都要设置
            G[v_i][v_j] = 1.0
            G[v_j][v_i] = 1.0
    return G

class Vertex():
    #顶点实例
    def __init__(self, vid, cid, nodes, k_in=0):
        self._vid = vid #顶点
        self._cid = cid #社区
        self._nodes = nodes #邻域
        self._kin = k_in  #结点内部的边的权重

class Louvain():
    
    def __init__(self, G):
        self._G = G
        self._m = 0 #边数量
        self._cid_vertices = {} #需维护的关于社区的信息(社区编号,其中包含的结点编号的集合)
        self._vid_vertex = {}   #需维护的关于结点的信息(结点编号，相应的Vertex实例)
        for vid in self._G.keys():
            self._cid_vertices[vid] = set([vid])
            self._vid_vertex[vid] = Vertex(vid, vid, set([vid]))#初始化各自都是一个社区
            self._m += sum([1 for neighbor in self._G[vid].keys() if neighbor>vid])

    #在步骤一它不断地遍历网络中的结点，尝试将单个结点加入能够使modularity提升最大的社区中，直到所有结点都不再变化。
    def first_stage(self):
        mod_inc = False  #用于判断算法是否可终止
        visit_sequence = self._G.keys()
        #random.shuffle(visit_sequence)
        while True:
            can_stop = True #第一阶段是否可终止
            for v_vid in visit_sequence:#遍历所有节点
                v_cid = self._vid_vertex[v_vid]._cid
                k_v = sum(self._G[v_vid].values()) + self._vid_vertex[v_vid]._kin
                cid_Q = {}
                for w_vid in self._G[v_vid].keys():
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:#说明已经计算过了
                        continue
                    else:
                        #计算添加到社区前后的Q值
                        tot = sum([sum(self._G[k].values())+self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        k_v_in = sum([v for k,v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        delta_Q = k_v_in - k_v * tot / self._m  #由于只需要知道delta_Q的正负，所以少乘了1/(2*self._m)
                        cid_Q[w_cid] = delta_Q

                #取值的topk
                cid,max_delta_Q = sorted(cid_Q.items(),key=lambda item:item[1],reverse=True)[0]
                if max_delta_Q > 0.0 and cid!=v_cid: #max_delta_Q不为正且自己已经被归纳就不停止
                    self._vid_vertex[v_vid]._cid = cid
                    self._cid_vertices[cid].add(v_vid)
                    self._cid_vertices[v_cid].remove(v_vid)
                    can_stop = False
                    mod_inc = True
            if can_stop:
                break
        return mod_inc

    #在步骤二，它处理第一阶段的结果，将一个个小的社区归并为一个超结点来重新构造网络，这时边的权重为两个结点内所有原始结点的边权重之和。迭代这两个步骤直至算法稳定。
    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        for cid,vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices: #更新节点
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                for k,v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v/2.0
            cid_vertices[cid] = set([cid])
            vid_vertex[cid] = new_vertex
        
        G = collections.defaultdict(dict)   
        for cid1,vertices1 in self._cid_vertices.items():#对所有的社区
            if len(vertices1) == 0:
                continue
            for cid2,vertices2 in self._cid_vertices.items():
                if cid2<=cid1 or len(vertices2)==0:
                    continue
                edge_weight = 0.0
                for vid in vertices1:
                    for k,v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v #权重为两个结点内所有原始结点的边权重之和
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G

    
    def get_communities(self):#得到社区的结果
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(c)
        return communities
    
    def execute(self):#class的主函数
        iter_time = 1
        while True:
            iter_time += 1
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()

#NMI(Normalized Mutual Information)标准化互信息，常用在聚类中，度量两个聚类结果的相近程度。
#是社区发现(community detection)的重要衡量指标，基本可以比较客观地评价出一个社区划分与标准划分之间相比的准确度。NMI的值域是0到1，越高代表划分得越准。
def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat



if __name__ == '__main__':
    G = load_graph('./email-Eu-core/email-Eu-core.txt')#载入数据集，这里是使用的snap官网的数据集
    algorithm = Louvain(G) #构图
    communities = algorithm.execute()#出结果
    print(communities)

    #导入真实的label
    lab = pd.read_table('./email-Eu-core/email-Eu-core-department-labels.txt',sep = ' ',names = ['node','com'])
    lable={}
    for _,data in lab.iterrows():
        node,com=data['node'],data['com']
        if com not in lable:
            lable[com]=set({node})
        else:
            lable[com].add(node)
    print(lable)

    #计算准确率
    count=0
    for c in communities:
        ma=0
        for l in lable:
            y=c&lable[l]
            if len(y)>ma:
                ma=len(y)
        count+=ma
    acc=count/len(lab)
    print(acc)
