'''
Created on Nov 18, 2020
@author: nakaizura
'''

from __future__ import print_function
import os, time, cPickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
import sklearn.preprocessing
from base_model import BaseModel, BaseModelParams, BaseDataIter
import utils
from flip_gradient import flip_gradient #自定义反向传播


class DataIter(BaseDataIter): #读取数据集
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open('./data/nuswide/img_train_id_feats.pkl', 'rb') as f:
            self.train_img_feats = cPickle.load(f) #载入训练图片特征
        with open('./data/nuswide/train_id_bow.pkl', 'rb') as f:
            self.train_txt_vecs = cPickle.load(f) #载入训练文本特征
        with open('./data/nuswide/train_id_label_map.pkl', 'rb') as f:
            self.train_labels = cPickle.load(f) #载入训练标签
        with open('./data/nuswide/img_test_id_feats.pkl', 'rb') as f:
            self.test_img_feats = cPickle.load(f) #载入测试图片特征
        with open('./data/nuswide/test_id_bow.pkl', 'rb') as f:
            self.test_txt_vecs = cPickle.load(f) #载入测试文本特征
        with open('./data/nuswide/test_id_label_map.pkl', 'rb') as f:
            self.test_labels = cPickle.load(f) #载入测试标签
        with open('data/nuswide/train_id_label_single.pkl', 'rb') as f:
            self.train_labels_single = cPickle.load(f) 
        with open('data/nuswide/test_id_label_single.pkl', 'rb') as f:
            self.test_labels_single = cPickle.load(f)              
                
        
        self.num_train_batch = len(self.train_img_feats) / self.batch_size #计算所需批次数
        self.num_test_batch = len(self.test_img_feats) / self.batch_size


    def train_data(self): #训练集
        for i in range(self.num_train_batch): #得到每批次的各数据
            batch_img_feats = [self.train_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [self.train_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [self.train_labels[n] for n in batch_img_ids]
            yield batch_img_feats, batch_txt_vecs, batch_labels, batch_labels_single, i #以迭代的模式返回

    def test_data(self): #测试集
        for i in range(self.num_test_batch): #得到每批次的各数据
            batch_img_feats = [self.test_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [self.test_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [self.test_labels[n] for n in batch_img_ids]
            yield batch_img_feats, batch_txt_vecs, batch_labels, batch_labels_single, i


class ModelParams(BaseModelParams):
    def __init__(self): #模型的各个参数
        BaseModelParams.__init__(self)

        self.epoch = 50 #周期
        self.margin = .1 #margin大小
        self.alpha = 5 #平衡参数
        self.batch_size = 64 #批次规模
        self.visual_feat_dim = 4096 #视觉特征维度
        #self.word_vec_dim = 300
        self.word_vec_dim = 5000 #文本词特征维度
        self.lr_emb = 0.0001 #学习率有2种，因为emb和adv(domain)的优化方向不一样（对抗）
        self.lr_domain = 0.0001
        self.top_k = 50 #topk
        self.semantic_emb_dim = 40 #语义嵌入维度
        self.dataset_name = 'wikipedia_datasete' #使用数据集名称
        self.model_name = 'adv_semantic_zsl' #模型名，用于存储
        self.model_dir = 'adv_semantic_zsl_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        #各种储存型文件的路径
        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    def update(self): #拼成完成路径并更新到self里面
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        #Triple版本会有正例和负例
        self.tar_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.tar_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.pos_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.neg_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.pos_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.neg_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size,10])
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size,1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.tar_img) #语义嵌入维度
        self.emb_w = self.label_embed(self.tar_txt)
        self.emb_v_pos = self.visual_feature_embed(self.pos_img,reuse=True)
        self.emb_v_neg = self.visual_feature_embed(self.neg_img,reuse=True)
        self.emb_w_pos = self.label_embed(self.pos_txt,reuse=True)
        self.emb_w_neg = self.label_embed(self.neg_txt,reuse=True)

        # 按照论文一共会计算2种loss，emb和adv loss，其中emb由两部分组成。以下一一注释。
        # triplet loss形式的emb loss
        margin = self.model_params.margin
        alpha = self.model_params.alpha
        v_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_v-self.emb_w_pos))
        v_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_v-self.emb_w_neg))
        w_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_w-self.emb_v_pos))
        w_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_w-self.emb_v_neg))
        self.triplet_loss = tf.maximum(0.,margin+alpha*v_loss_pos-v_loss_neg) + tf.maximum(0.,margin+alpha*w_loss_pos-w_loss_neg)

        # lable loss是多分类交叉熵损失
        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.emb_loss = 100*self.label_loss + self.triplet_loss # emb loss由以上2种得到

        # 模态分类对应论文的adv loss，使模型无法分清文本or图像
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)#先预测lable
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                                   tf.zeros([self.model_params.batch_size, 1])], 1)#拼接lable方便一起计算
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                                   tf.ones([self.model_params.batch_size, 1])], 1)
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)#得到adv loss

        self.t_vars = tf.trainable_variables()#因为adv和emb的计算方向不一样，所以需要分别优化
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name] #vf和le是训练emb的，对应投影器
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name] #dc和lc是训练adv的，对应分类器
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]

    def visual_feature_embed(self, X, is_training=True, reuse=False):#即论文中的视觉特征投影器
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X, 512, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    def label_embed(self, L, is_training=True, reuse=False):#即论文中的文本特征投影器
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, self.model_params.semantic_emb_dim, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net 
    def label_classifier(self, X, reuse=False):#标签预测
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 10, scope='lc_fc_0')
        return net         
    def domain_classifier(self, E, l, is_training=True, reuse=False):#二分类是文本/图像
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, self.model_params.semantic_emb_dim/2, scope='dc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim/4, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def train(self, sess):
        #self.check_dirs()
        #emb和adv(domain)分别优化，GAN的思想进行训练
        total_loss = self.emb_loss + self.domain_class_loss
        total_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_total,
            beta1=0.5).minimize(total_loss)
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars+self.vf_vars)
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

        start_time = time.time()
        map_avg_ti = []
        map_avg_it = []
        adv_loss = []
        emb_loss = []
        for epoch in range(self.model_params.epoch):
                        
            p = float(epoch) / self.model_params.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1 #学习率按周期衰减
            for batch_feat, batch_vec, batch_labels, idx in self.data_iter.train_data():
                # create one-hot labels
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(max(batch_labels_)+1))
                b = label_binarizer.transform(batch_labels_)
                adj_mat = np.dot(b,np.transpose(b))
                mask_mat = np.ones_like(adj_mat) - adj_mat
                img_sim_mat = mask_mat*cosine_similarity(batch_feat,batch_feat)
                txt_sim_mat = mask_mat*cosine_similarity(batch_vec,batch_vec)
                img_neg_txt_idx = np.argmax(img_sim_mat,axis=1).astype(int)
                txt_neg_img_idx = np.argmax(txt_sim_mat,axis=1).astype(int)
                #print('{0}'.format(img_neg_txt_idx.shape)
                batch_vec_ = np.array(batch_vec)
                batch_feat_ = np.array(batch_feat)                
                img_neg_txt = batch_vec_[img_neg_txt_idx,:]
                txt_neg_img = batch_feat_[txt_neg_img_idx,:]
                #_, label_loss_val, dissimilar_loss_val, similar_loss_val = sess.run([total_train_op, self.label_loss, self.dissimilar_loss, self.similar_loss], feed_dict={self.tar_img: batch_feat, self.tar_txt: batch_vec, self.y: b, self.y_single: np.transpose([batch_labels]),self.l: l})
                sess.run([emb_train_op, domain_train_op],
                          feed_dict={self.tar_img: batch_feat,
                          self.tar_txt: batch_vec,
                          self.pos_txt: batch_vec,
                          self.neg_txt: img_neg_txt,
                          self.pos_img: batch_feat,
                          self.neg_img: txt_neg_img,
                          self.y: b,
                          self.y_single: np.transpose([batch_labels]),
                          self.l: l})
                label_loss_val, triplet_loss_val, emb_loss_val, domain_loss_val= sess.run([self.label_loss, self.triplet_loss, self.emb_loss, self.domain_class_loss],
                          feed_dict={self.tar_img: batch_feat,
                          self.tar_txt: batch_vec,
                          self.pos_txt: batch_vec,
                          self.neg_txt: img_neg_txt,
                          self.pos_img: batch_feat,
                          self.neg_img: txt_neg_img,
                          self.y: b,
                          self.y_single: np.transpose([batch_labels]),
                          self.l: l})
                print('Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, triplet_loss: %.8f' %(
                    epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, emb_loss_val, domain_loss_val, label_loss_val, triplet_loss_val
                ))
            #if epoch == (self.model_params.epoch - 1): 
            #    self.emb_v_eval, self.emb_w_eval = sess.run([self.emb_v, self.emb_w],     
            #             feed_dict={
            #                 self.tar_img: batch_feat,
            #                 self.tar_txt: batch_vec,
            #                 self.y: b,
            #                 self.y_single: np.transpose([batch_labels]),
            #                 self.l: l})
            #    with open('./data/wikipedia_dataset/train_img_emb.pkl', 'wb') as f:
            #        cPickle.dump(self.emb_v_eval, f, cPickle.HIGHEST_PROTOCOL)
            #    with open('./data/wikipedia_dataset/train_txt_emb.pkl', 'wb') as f:
            #        cPickle.dump(self.emb_w_eval, f, cPickle.HIGHEST_PROTOCOL)                    
    def eval_random_rank(self): #测试随机排序的性能
        start = time.time()
        #with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as fpkl:
        #    test_labels = cPickle.load(fpkl)
        with open('./data/wiki_shallow/L_te.pkl', 'rb') as fpkl:
            test_labels = cPickle.load(fpkl)
        k = self.model_params.top_k
        avg_precs = []
        for i in range(len(test_labels)):
            query_label = test_labels[i]

            # distances and sort by distances
            sorted_idx = range(len(test_labels))
            shuffle(sorted_idx) #直接打乱

            # for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0 : topk]
                if query_label != test_labels[top_k[-1]]:
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if query_label != retrieved_label:
                        hits += 1
                precs.append(float(hits) / float(topk))
            avg_precs.append(np.sum(precs) / float(k))
        mean_avg_prec = np.mean(avg_precs)
        print('[Eval - random] mAP: %f in %4.4fs' % (mean_avg_prec, (time.time() - start)))
        

    def eval(self, sess): #评价训练后的性能
        start = time.time()

        test_img_feats_trans = []
        test_txt_vecs_trans = []
        test_labels = []
        for feats, vecs, labels, i in self.data_iter.test_data():
            feats_trans = sess.run(self.emb_v, feed_dict={self.tar_img: feats})
            vecs_trans = sess.run(self.emb_w, feed_dict={self.tar_txt: vecs})
            test_labels += labels      
            for ii in range(len(feats)):
                test_img_feats_trans.append(feats_trans[ii])
                test_txt_vecs_trans.append(vecs_trans[ii])
        test_img_feats_trans = np.asarray(test_img_feats_trans)
        test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)
        test_feats_trans = np.concatenate((test_img_feats_trans[0:1000], test_txt_vecs_trans[-1000:]))
        #with open('./data/wikipedia_dataset/test_feats_transformed.pkl', 'wb') as f:
        #    cPickle.dump(test_feats_trans, f, cPickle.HIGHEST_PROTOCOL)        
        with open('./data/wiki_shallow/test_feats_transformed.pkl', 'wb') as f:
            cPickle.dump(test_feats_trans, f, cPickle.HIGHEST_PROTOCOL)                   
        print('[Eval] transformed test features in %4.4f' % (time.time() - start))
        top_k = self.model_params.top_k
        avg_precs = []
        all_precs = []
        for k in top_k:
            for i in range(len(test_txt_vecs_trans)):#计算文本检索的topk
                query_label = test_labels[i]

                # distances and sort by distances
                wv = test_txt_vecs_trans[i]
                diffs = test_img_feats_trans - wv
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                #for each k do top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if np.sum(query_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(retrieved_label) == query_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        print('[Eval - txt2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))
        t2i = all_precs[0]
        #with open('./data/wikipedia_dataset/txt2img_all_precision.pkl', 'wb') as f:
        #    cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)     
        with open('./data/wiki_shallow/txt2img_all_precision.pkl', 'wb') as f:
            cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)                  

        avg_precs = []
        all_precs = []

        for k in top_k:        
            for i in range(len(test_img_feats_trans)):#计算文图像检索的topk
                query_img_feat = test_img_feats_trans[i]
                ground_truth_label = test_labels[i]

                # calculate distance and sort
                diffs = test_txt_vecs_trans - query_img_feat
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k in top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if np.sum(ground_truth_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(ground_truth_label) == retrieved_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)            
        print('[Eval - img2txt] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))

        
        
        #with open('./data/wikipedia_dataset/text_words_map.pkl', 'wb') as f:
        #    cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)
        with open('./data/wiki_shallow/text_words_map.pkl', 'wb') as f:
            cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)        
        #Text query    

        #with open('./data/wikipedia_dataset/text_words_map.pkl', 'rb') as f:
        #    txt_words = cPickle.load(f)
        #with open('./data/wikipedia_dataset/test_img_words.pkl', 'rb') as f:
        #    img_words = cPickle.load(f)
        #with open('./data/wikipedia_dataset/test_txt_files.pkl', 'rb') as f:
        #    test_txt_names = cPickle.load(f)
        #with open('./data/wikipedia_dataset/test_img_files.pkl', 'rb') as f:
        #    test_img_names = cPickle.load(f)   
        with open('./data/wikipedia_dataset/text_words_map.pkl', 'rb') as f:
            txt_words = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_img_words.pkl', 'rb') as f:
            img_words = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_txt_files.pkl', 'rb') as f:
            test_txt_names = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_img_files.pkl', 'rb') as f:
            test_img_names = cPickle.load(f)                                                    
         # Precision-scope for text query
        scope = 100
        retrieval_results = []
        precisions = np.zeros(scope)
        for i in range(len(test_txt_vecs_trans)):
            query_words = img_words[test_img_names[i]]

            # distances and sort by distances
            wv = test_txt_vecs_trans[i]
            diffs = test_img_feats_trans - wv
            dists = np.linalg.norm(diffs, axis=1)
            sorted_idx = np.argsort(dists)

            hits = np.zeros(scope)
            p = np.zeros(scope)
            for k in range(scope):
                retrieved = img_words[test_img_names[sorted_idx[k]]]
                if utils.is_text_relevant(query_words, retrieved, None):
                    hits[k] = 1.0
            for k in range(scope):
                p[k] = np.sum(hits[0:k]) / float(k + 1)
            precisions += p

            if i in sorted_idx[0:5] and np.sum(hits[0:5]) >= 4:
                result = {
                    'query': test_txt_names[i],
                    'retrieval': [test_img_names[hh] for hh in sorted_idx[0:5]]
                }
                retrieval_results.append(result)


        with open('./data/wikipedia_dataset/txt2img-retrievals.pkl', 'wb') as f:
            cPickle.dump(retrieval_results, f, cPickle.HIGHEST_PROTOCOL)
        print('[Eval - txt2img] finished precision-scope in %4.4fs' % (time.time() - start))

        # Precision-scope for image query
        retrieval_results = []
        precisions = np.zeros(scope)
        for i in range(len(test_img_feats_trans)):
            query_img_feat = test_img_feats_trans[i]
            query_img_name = test_img_names[i]
            ground_truth_words = img_words[query_img_name]

            # calculate distance and sort
            diffs = test_txt_vecs_trans - query_img_feat
            dists = np.linalg.norm(diffs, axis=1)
            sorted_idx = np.argsort(dists)

            hits = np.zeros(scope)
            p = np.zeros(scope)
            for k in range(scope):
                retrieved = img_words[test_img_names[sorted_idx[k]]]
                if utils.is_text_relevant(retrieved, ground_truth_words, None):
                    hits[k] = 1.0
            for k in range(scope):
                p[k] = np.sum(hits[0:k]) / float(k + 1)
            precisions += p

            if i in sorted_idx[0:5] and np.sum(hits[0:5]) >= 4 :
                result = {
                    'query': test_img_names[i],
                    'retrieval': [test_txt_names[hh] for hh in sorted_idx[0:5]]
                }
                retrieval_results.append(result)

        with open('./data/wikipedia_dataset/img2txt-retrievals.pkl', 'wb') as f:
            cPickle.dump(retrieval_results, f, cPickle.HIGHEST_PROTOCOL)
        print('[Eval - img2txt] finished precision-scope in %4.4fs' % (time.time() - start))
