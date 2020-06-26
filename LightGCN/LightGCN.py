'''
Created on Jun 26, 2020
@author: nakaizura
'''

import os
import sys
import threading
import tensorflow as tf
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']

class LightGCN(object):
    def __init__(self, data_config, pretrain_data):
        # 参数设置
        self.model_type = 'LightGCN'
        self.adj_type = args.adj_type #adj邻接矩阵的处理类型，如norm等
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data #预训练集
        self.n_users = data_config['n_users'] #user总数
        self.n_items = data_config['n_items'] #item总数
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj'] #正则化邻接矩阵
        self.n_nonzero_elems = self.norm_adj.count_nonzero() #norm_adj的非零个数
        self.lr = args.lr #学习率
        self.emb_dim = args.embed_size #嵌入大小
        self.batch_size = args.batch_size #批次大小
        self.weight_size = eval(args.layer_size) #每一层权重的大小
        self.n_layers = len(self.weight_size) #层数
        self.regs = eval(args.regs)
        self.decay = self.regs[0] #衰减系数
        self.log_dir=self.create_model_str()
        self.verbose = args.verbose #定期打印结果
        self.Ks = eval(args.Ks) #K


        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,)) #用户
        self.pos_items = tf.placeholder(tf.int32, shape=(None,)) #正例的item
        self.neg_items = tf.placeholder(tf.int32, shape=(None,)) #负例的item，正负例是因为最后做的是bpr loss。
        
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))
        
        
        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            #record for top(Ks[0])
            tf.summary.scalar('train_rec_first', self.train_rec_first)
            self.train_rec_last = tf.placeholder(tf.float32)
            #record for top(Ks[-1])
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # 初始化模型权重
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            
        elif self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        #经过模型处理之后可以得到user和正负例的lookup最后的嵌入
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        #测试阶段直接算内积
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        #计算bpr的loss
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss
        #Adam优化器
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    
    def create_model_str(self):#创造路径
        log_dir = '/' + self.alg_type+'/layers_'+str(self.n_layers)+'/dim_'+str(self.emb_dim)
        log_dir+='/'+args.dataset+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir


    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()#初始化权重使用Xavie
        if self.pretrain_data is None:#如果没有预训练的数据就直接用总数根据ID得到普通的嵌入
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')
            
        self.weight_size_list = [self.emb_dim] + self.weight_size
        
        for k in range(self.n_layers):#对每一层gcn的权重都进行初始化
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights
    def _split_A_hat(self, X):
        #得到子矩阵adjacency sub-matrix.
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold): #一共n_fold=100
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        #带dropout的版本
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold): #分成n个fold
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        # 使用矩阵的解法，所以先得到邻接矩阵
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        #自嵌入形式
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        #执行k层的消息传播
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):#所有的fold
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            #相比gncf，这里直接加和就完了
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        # 使用矩阵的解法，所以先得到邻接矩阵
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        #自嵌入形式
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]
        #执行k层的消息传播
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):#所有的fold
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # u到u，聚合u所有邻居的消息
            side_embeddings = tf.concat(temp_embed, 0)
            # 特征变换矩阵
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # i到u，合并自嵌入，邻居的嵌入
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # 再次变换特征
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            #非线性激活函数，u到u和i到u的两部分
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            #正则化
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)#拼一起
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    
    def _create_gcn_embed(self):
        #普通gcn得到的嵌入
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            #gcn的处理方式就是特征变化之后再激活，而ngcf有两种
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)#拼一起
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        #gcmc得到的嵌入
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # 也是gcn
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # 加mlp以重建链路
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        #用正例负例计算bpr
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        #使两者差距尽可能的大
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        

        emb_loss = self.decay * regularizer#正则化

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        #矩阵变稀疏矩阵，用coo的存储形式
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
        
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    #载入预训练数据
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

# parallelized sampling on CPU 
class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample()

class sample_thread_test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample_test()
            
# training on GPU
class train_thread(threading.Thread):
    def __init__(self,model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
    def run(self):
        if len(gpus):
            with tf.device(gpus[-1]):
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run([self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss],
                                       feed_dict={model.users: users, model.pos_items: pos_items,
                                                  model.node_dropout: eval(args.node_dropout),
                                                  model.mess_dropout: eval(args.mess_dropout),
                                                  model.neg_items: neg_items})
        else:
            users, pos_items, neg_items = self.sample.data
            self.data = sess.run([self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss],
                                       feed_dict={model.users: users, model.pos_items: pos_items,
                                                  model.node_dropout: eval(args.node_dropout),
                                                  model.mess_dropout: eval(args.mess_dropout),
                                                  model.neg_items: neg_items})
class train_thread_test(threading.Thread):
    def __init__(self,model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
    def run(self):
        if len(gpus):
            with tf.device(gpus[-1]):
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss],
                                     feed_dict={model.users: users, model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.node_dropout: eval(args.node_dropout),
                                                model.mess_dropout: eval(args.mess_dropout)})
        else:
            users, pos_items, neg_items = self.sample.data
            self.data = sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss],
                                     feed_dict={model.users: users, model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.node_dropout: eval(args.node_dropout),
                                                model.mess_dropout: eval(args.mess_dropout)})            

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    f0 = time()
    
    config = dict()
    config['n_users'] = data_generator.n_users#得到总数
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj,pre_adj = data_generator.get_adj_mat()#生成拉普拉斯矩阵
    if args.adj_type == 'plain':#选择对adj的处理
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type=='pre':
        config['norm_adj']=pre_adj
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
    model = LightGCN(data_config=config, pretrain_data=pretrain_data)#实例化模型
    
    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()#用于保存模型的参数

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:#如果有预训练的参数就直接载入
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]
                
                pretrain_ret = 'pretrained model recall=[%s], precision=[%s], '\
                               'ndcg=[%s]' % \
                               (', '.join(['%.5f' % r for r in ret['recall']]),
                                ', '.join(['%.5f' % r for r in ret['precision']]),
                                ', '.join(['%.5f' % r for r in ret['ndcg']]))
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')
         
        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], ndcg=[%s]" % \
                         (', '.join(['%.5f' % r for r in ret['recall']]),
                          ', '.join(['%.5f' % r for r in ret['precision']]),
                          ', '.join(['%.5f' % r for r in ret['ndcg']]))

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    #开始训练
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path + model.log_dir +'/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path +model.log_dir+ '/run_' + str(run_time), sess.graph)
    
    
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    
    
    for epoch in range(1, args.epoch + 1):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.
        '''
        *********************************************************
        parallelized sampling
        '''
        sample_last = sample_thread()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread(model, sess, sample_last)
            sample_next = sample_thread()
            
            train_cur.start()
            sample_next.start()
            
            sample_next.join()
            train_cur.join()
            
            users, pos_items, neg_items = sample_last.data
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = train_cur.data
            sample_last = sample_next
        
            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            emb_loss += batch_emb_loss/n_batch
            
        summary_train_loss= sess.run(model.merged_train_loss,
                                      feed_dict={model.train_loss: loss, model.train_mf_loss: mf_loss,
                                                 model.train_emb_loss: emb_loss, model.train_reg_loss: reg_loss})
        train_writer.add_summary(summary_train_loss, epoch)
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()
        
        if (epoch % 20) != 0:#每20个周期就打印一次结果
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        users_to_test = list(data_generator.train_items.keys())
        ret = test(sess, model, users_to_test ,drop_flag=True,train_set_flag=1)
        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s]' % \
                   (epoch, loss, mf_loss, emb_loss, reg_loss, 
                    ', '.join(['%.5f' % r for r in ret['recall']]),
                    ', '.join(['%.5f' % r for r in ret['precision']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]))
        print(perf_str)
        summary_train_acc = sess.run(model.merged_train_acc, feed_dict={model.train_rec_first: ret['recall'][0],
                                                                        model.train_rec_last: ret['recall'][-1],
                                                                        model.train_ndcg_first: ret['ndcg'][0],
                                                                        model.train_ndcg_last: ret['ndcg'][-1]})
        train_writer.add_summary(summary_train_acc, epoch // 20)
        
        '''
        *********************************************************
        parallelized sampling
        '''
        sample_last= sample_thread_test()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread_test(model, sess, sample_last)
            sample_next = sample_thread_test()
            
            train_cur.start()
            sample_next.start()
            
            sample_next.join()
            train_cur.join()
            
            users, pos_items, neg_items = sample_last.data
            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test = train_cur.data
            sample_last = sample_next
            
            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            emb_loss_test += batch_emb_loss_test / n_batch
            
        summary_test_loss = sess.run(model.merged_test_loss,
                                     feed_dict={model.test_loss: loss_test, model.test_mf_loss: mf_loss_test,
                                                model.test_emb_loss: emb_loss_test, model.test_reg_loss: reg_loss_test})
        train_writer.add_summary(summary_test_loss, epoch // 20)
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        summary_test_acc = sess.run(model.merged_test_acc,
                                    feed_dict={model.test_rec_first: ret['recall'][0], model.test_rec_last: ret['recall'][-1],
                                               model.test_ndcg_first: ret['ndcg'][0], model.test_ndcg_last: ret['ndcg'][-1]})
        train_writer.add_summary(summary_test_acc, epoch // 20)
                                                                                                 
                                                                                                 
        t3 = time()
        
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%s], ' \
                       'precision=[%s], ndcg=[%s]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test, 
                        ', '.join(['%.5f' % r for r in ret['recall']]),
                        ', '.join(['%.5f' % r for r in ret['precision']]),
                        ', '.join(['%.5f' % r for r in ret['ndcg']]))
            print(perf_str)
            
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
