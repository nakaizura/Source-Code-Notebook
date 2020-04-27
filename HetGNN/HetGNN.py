'''
Created on Apr 27, 2020
@author: nakaizura
'''

import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
torch.set_num_threads(2) #设置线程
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


class model_class(object):
	def __init__(self, args):
		super(model_class, self).__init__()
		self.args = args
		self.gpu = args.cuda

                #导入各种数据
		input_data = data_generator.input_data(args = self.args)
		#input_data.gen_het_rand_walk()

		self.input_data = input_data

		if self.args.train_test_label == 2: #为每个node生成邻居
                        #重启策略的随机游走，为每个节点采样固定数量的强相关的异构邻居，然后按类型分组
                        #任意节点开始随机游走，以p概率返回。采样到固定数量后就停止。
                        #为了采样采样的邻居包含所有类型的节点，不同类型节点的数量是受限的。
                        #对每个类型都选出按频率的topk邻居
			input_data.het_walk_restart()
			print ("neighbor set generation finish")
			exit(0)

                #p是论文，a是作者，v是地点，然后可以组成一堆特征
		feature_list = [input_data.p_abstract_embed, input_data.p_title_embed,\
		input_data.p_v_net_embed, input_data.p_a_net_embed, input_data.p_ref_net_embed,\
		input_data.p_net_embed, input_data.a_net_embed, input_data.a_text_embed,\
		input_data.v_net_embed, input_data.v_text_embed]

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()
		#self.feature_list = feature_list

                #各自的邻居列表
		a_neigh_list_train = input_data.a_neigh_list_train
		p_neigh_list_train = input_data.p_neigh_list_train
		v_neigh_list_train = input_data.v_neigh_list_train

		a_train_id_list = input_data.a_train_id_list
		p_train_id_list = input_data.p_train_id_list
		v_train_id_list = input_data.v_train_id_list

		self.model = tools.HetAgg(args, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train,\
		 a_train_id_list, p_train_id_list, v_train_id_list)#实例化model，tools会对异构的信息进行聚合

		if self.gpu:
			self.model.cuda()
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay = 0)#Adam优化器
		self.model.init_weights()


	def model_train(self):
                #开始训练
		print ('model training ...')
		if self.args.checkpoint != '':
			self.model.load_state_dict(torch.load(self.args.checkpoint))
		
		self.model.train() #模型调到训练模式
		mini_batch_s = self.args.mini_batch_s #batch
		embed_d = self.args.embed_d #嵌入维度

		for iter_i in range(self.args.train_iter_n): #迭代次数
			print ('iteration ' + str(iter_i) + ' ...')
			triple_list = self.input_data.sample_het_walk_triple()#异构三元组（含正例 负例）采样
			min_len = 1e10
			for ii in range(len(triple_list)):
				if len(triple_list[ii]) < min_len:
					min_len = len(triple_list[ii])
			batch_n = int(min_len / mini_batch_s)
			print (batch_n)
			for k in range(batch_n):
				c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])#pos，正例
				n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])#neg，负例

				for triple_index in range(len(triple_list)):
					triple_list_temp = triple_list[triple_index]
					triple_list_batch = triple_list_temp[k * mini_batch_s : (k + 1) * mini_batch_s]
					#得到模型的预测结果
					c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)

					c_out[triple_index] = c_out_temp
					p_out[triple_index] = p_out_temp
					n_out[triple_index] = n_out_temp

				loss = tools.cross_entropy_loss(c_out, p_out, n_out, embed_d)#计算三元组交叉熵

				self.optim.zero_grad()#梯度清零
				loss.backward()#反向传播
				self.optim.step() #参数更新

				if k % 100 == 0: #打印结果
					print ("loss: " + str(loss))

			if iter_i % self.args.save_model_freq == 0:
				torch.save(self.model.state_dict(), self.args.model_path + "HetGNN_" + str(iter_i) + ".pt")
				#存储参数用于评估
				triple_index = 9 #一共有9种case，在tools文件中定义
				a_out, p_out, v_out = self.model([], triple_index)
			print ('iteration ' + str(iter_i) + ' finish.')



if __name__ == '__main__':
	args = read_args()
	print("------arguments-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))

	#可复现随机种子
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	#实例化模型
	model_object = model_class(args)

	if args.train_test_label == 0:
		model_object.model_train()
