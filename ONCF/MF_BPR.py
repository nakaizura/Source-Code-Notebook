'''
Created on Apr 19, 2020
@author: nakaizura
'''

from __future__ import absolute_import
from __future__ import division
import os
import math
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset
from saver import GMFSaver

#MF_BPR用于预训练embedding。

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽通知信息和警告
#0是默认值，输出所有信息
#1屏蔽通知信息
#2屏蔽通知信息和警告
#3屏蔽通知信息，警告和报错信息。但此时仍然会输出FATAL信息。

_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_sess = None
_dataset = None
_K = None
_feed_dict = None
_output = None

#################### 参数设置 ####################
def parse_args():
    '''
    python自带的参数解析包argparse，方便读取命令行参数。
    参数主要有：数据集路径、数据集名称、预训练模型、批次大小、周期、隐层大小、负采样数目、正则化系数、任务名、学习率、是否预训练、是否保存模型、是否计算训练时的auc。
    '''
    parser = argparse.ArgumentParser(description="Run MF-BPR.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yelp',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='MF',
                        help='Choose model: MF')
    parser.add_argument('--verbose', type=int, default=50,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dns', type=int, default=1,
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--regs', nargs='?', default='[0.01,0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Use the pretraining weights or not')
    parser.add_argument('--ckpt', type=int, default=0,
                        help='Save the pretraining weights or not')
    parser.add_argument('--train_auc', type=int, default=0,
                        help='Calculate train_auc or not')
    return parser.parse_args()


#---------- 数据处理 -------
#数据采样和打乱

# input: dataset(Mat, List, Rating, Negatives), batch_choice
# output: [_user_input_list, _item_input_pos_list]
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    for (u, i) in dataset.trainMatrix.keys():
        #有交互的正例
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos

def shuffle(samples, batch_size, dataset, model):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _index
    global _model
    global _dataset
    _user_input, _item_input_pos = samples
    _batch_size = batch_size
    _index = range(len(_user_input))
    _model = model
    _dataset = dataset
    np.random.shuffle(_index)#随机打乱
    num_batch = len(_user_input) // _batch_size #计算batch数量
    pool = Pool(cpu_count()) #多线程处理batch的切分
    res = pool.map(_get_train_batch, range(num_batch))#_get_train_batch函数生成batch
    pool.close() #关闭pool，不再接受新进程
    pool.join() #子进程运行完后，再把主进程全部关掉。
    user_list = [r[0] for r in res]
    item_pos_list = [r[1] for r in res]
    user_dns_list = [r[2] for r in res]
    item_dns_list = [r[3] for r in res]
    return user_list, item_pos_list, user_dns_list, item_dns_list

def _get_train_batch(i):
    '''生成batch，i是batch的序号，顺序生成batch'''
    user_batch, item_batch = [], []
    user_neg_batch, item_neg_batch = [], []
    begin = i * _batch_size #起始点
    for idx in range(begin, begin + _batch_size):#顺序采样到一个batch大小
        #先append有交互的正例
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input_pos[_index[idx]])
        #再负采样无交互的负例
        for dns in range(_model.dns): #dns是负采样数目
            user = _user_input[_index[idx]]
            user_neg_batch.append(user)#先append用户
            #在所有items中随机采样
            gtItem = _dataset.testRatings[user][1]
            j = np.random.randint(_dataset.num_items)
            while j in _dataset.trainList[_user_input[_index[idx]]]:#如果采到正例了就再采一次
                j = np.random.randint(_dataset.num_items)
            item_neg_batch.append(j)
    #返回正例user-item和负例user-item
    return np.array(user_batch)[:,None], np.array(item_batch)[:,None], \
           np.array(user_neg_batch)[:,None], np.array(item_neg_batch)[:,None]

#---------- GMF模型 -------
class GMF:
    def __init__(self, num_users, num_items, args):
        #导入所有参数
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)#恢复成列表
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.dns = args.dns
        self.train_auc = args.train_auc

    def _create_placeholders(self):
        #常量，主要是输入数据集
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_neg")
    def _create_variables(self):
        #变量，主要是嵌入参数
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)

            self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name = "h")


    #----------MF-BPR模型的核心内容---------------
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            #对user和item进行嵌入再相乘，即MF预测。
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1) #(b, embedding_size)

            return self.embedding_p, self.embedding_q,\
                tf.matmul(self.embedding_p*self.embedding_q, self.h)  #(b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            #得到损失函数
            self.p1, self.q1, self.output = self._create_inference(self.item_input_pos)#user和正例的MF分数
            self.p2, self.q2, self.output_neg = self._create_inference(self.item_input_neg)#user和负例的MF分数
            #BPR loss
            self.result = self.output - self.output_neg #正负例差值
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) #sigmoid log 再求和

            #把得分都加起来。
            self.opt_loss = self.loss + self.lambda_bilinear * ( tf.reduce_sum(tf.square(self.p1)) \
                                    + tf.reduce_sum(tf.square(self.q2)) + tf.reduce_sum(tf.square(self.q1)))

    def _create_optimizer(self):
        #梯度下降优化器
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)

    def build_graph(self):
        #构建计算图
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    def saveParams(self, sess, fpath):
        #存储参数
        np.save(fpath, sess.run([self.embedding_P,self.embedding_Q,self.h]))

    def load_parameter_MF(self, sess, path):
        #载入MF的参数
        ps = np.load(path)
        ap = tf.assign(self.embedding_P, ps[0])
        aq = tf.assign(self.embedding_Q, ps[1])
        #ah = tf.assign(self.h, np.diag(ps[2][:, 0]).reshape(4096, 1))
        sess.run([ap, aq])
        print ("parameter loaded")



#---------- 训练过程 -------
def training(model, dataset, args, saver = None):
    with tf.Session() as sess:
        #初始化图，没有路径就准备创建保存
        ckpt_save_path = "Pretrain/MF_BPR/embed_%d/" %  args.embed_size
        ckpt_restore_path = "Pretrain/MF_BPR/embed_%d/" %  args.embed_size
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        if not os.path.exists(ckpt_restore_path):
            os.makedirs(ckpt_restore_path)

        #初始化saver，主要是嵌入参数P和Q
        saver_ckpt = tf.train.Saver({'embedding_P':model.embedding_P,'embedding_Q':model.embedding_Q})

        #初始化变量
        sess.run(tf.global_variables_initializer())

        #如果有预训练模型就恢复权重
        if args.pretrain:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
                logging.info("using pretrained variables")
                print ("using pretrained variables")

        #没有预训练就初始化
        else:
            logging.info("initialized")
            print ("initialized")

        #初始化Evaluate模块，构建测试集等
        eval_feed_dicts = init_eval_model(model, dataset)

        #从数据集中采样
        samples = sampling(dataset)

        #保存最好的结果
        max_ndcg = 0
        max_res = " "

        #开始训练
        for epoch_count in range(args.epochs):

            #打乱训练集，生成batch
            batch_begin = time()
            batches = shuffle(samples, args.batch_size, dataset, model)
            batch_time = time() - batch_begin

            #先计算一下在训练前的acc
            prev_batch = batches[0], batches[1], batches[3]
            _, prev_acc = training_loss_acc(model, sess, prev_batch)

            #开始训练
            train_begin = time()
            train_batches = training_batch(model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0 or epoch_count == args.epochs-1:
                _, ndcg, cur_res = output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts,
                                epoch_count, batch_time, train_time, prev_acc)

                #定期打印结果，存储最优值
                if max_ndcg < ndcg:
                    max_ndcg = ndcg
                    max_res = cur_res
                    model.saveParams(sess, "best_%s_MF.npy" % args.dataset)

            #保存嵌入权重参数
            if args.ckpt and epoch_count%100 == 0:
                saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch_count)

        model.saveParams(sess, "final_%s_MF_%f.npy" % (args.dataset, time()))
        print ("best:" + max_res)
        logging.info("best:" + max_res)


# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches):
    user_input, item_input_pos, user_dns_list, item_dns_list = batches
    # 如果负采样数dns = 1, 就直接计算BPR
    if model.dns == 1:
        item_input_neg = item_dns_list
        #对BPR loss进行训练
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_input_neg[i]}
            sess.run(model.optimizer, feed_dict)
    # 如果负采样数dns > 1, 就mini-batch BPR
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            #得到负采样样本的得分output_neg
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i]}
            output_neg = sess.run(model.output_neg, feed_dict)
            #根据得分从负采样样本中选择最好的样本
            item_neg_batch = []
            for j in range(0, len(output_neg), model.dns):
                item_index = np.argmax(output_neg[j : j + model.dns])#分数最高，即对模型的干扰越强
                item_neg_batch.append(item_dns_list[i][j : j + model.dns][item_index][0])
            item_neg_batch = np.array(item_neg_batch)[:,None]
            #再进行mini-batch
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_neg_batch}
            sess.run(model.optimizer, feed_dict)
            item_input_neg.append(item_neg_batch)
    return user_input, item_input_pos, item_input_neg

# input: model, sess, batches
# output: training_loss
def training_loss_acc(model, sess, train_batches):
    #计算acc
    train_loss = 0.0
    acc = 0
    num_batch = len(train_batches[1])
    user_input, item_input_pos, item_input_neg = train_batches
    for i in range(len(user_input)):#对每个用户
        # print (user_input[i][0]. item_input_pos[i][0], item_input_neg[i][0])
        feed_dict = {model.user_input: user_input[i],
                     model.item_input_pos: item_input_pos[i],
                     model.item_input_neg: item_input_neg[i]}

        loss, output_pos, output_neg = sess.run([model.loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)#计算acc
    return train_loss / num_batch, acc / num_batch



#---------- evaluation -------
def output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts, epoch_count, batch_time, train_time, prev_acc):
    #对模型output的评估
    loss_begin = time()
    train_loss, post_acc = 0,0 #training_loss_acc(model, sess, train_batches)
    loss_time = time() - loss_begin

    eval_begin = time()
    hr, ndcg, auc, train_auc = evaluate(model, sess, dataset, eval_feed_dicts)#计算hr, ndcg, auc, train_auc
    eval_time = time() - eval_begin

    res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f train_AUC = %.4f [%.1fs]" \
        " ACC = %.4f train_loss = %.4f ACC = %.4f [%.1fs]" % ( epoch_count, batch_time, train_time,
                    hr, ndcg, auc, train_auc, eval_time, prev_acc, train_loss, post_acc, loss_time)
    # res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f [%.1fs]" % (epoch_count, batch_time, train_time, hr, ndcg, auc, eval_time)

    logging.info(res)
    print res

    return post_acc, ndcg, res

def init_eval_model(model, dataset):
    #测试初始化时候input的评估
    global _dataset
    global _model
    _dataset = dataset
    _model = model

    pool = Pool(cpu_count())
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))#载入负样本
    pool.close()
    pool.join()

    print("already load the evaluate model...")
    return feed_dicts

def _evaluate_input(user):
    #生成items_list
    item_input = dataset.testNegatives[user] #读负样本
    test_item = _dataset.testRatings[user][1]
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:,None]
    return user_input, item_input


def evaluate(model, sess, dataset, feed_dicts):
    #评估模型，计算hr, ndcg, auc, train_auc
    global _model
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    global _output
    _dataset = dataset
    _model = model
    _sess = sess
    _K = 10
    _feed_dicts = feed_dicts

    res = []
    for user in range(_dataset.num_users):
        res.append(_eval_by_user(user))
    res = np.array(res)
    hr, ndcg, auc, train_auc = (res.mean(axis = 0)).tolist()

    return hr, ndcg, auc, train_auc

def _eval_by_user(user):

    if _model.train_auc:
        #训练集中对正例的预测
        train_item_input = _dataset.trainList[user]
        train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
        train_item_input = np.array(train_item_input)[:, None]
        feed_dict = {_model.user_input: train_user_input, _model.item_input_pos: train_item_input}

        train_predict = _sess.run(_model.output, feed_dict)

    #得到测试集的预测
    user_input, item_input = _feed_dicts[user]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input}

    predictions = _sess.run(_model.output, feed_dict)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum() #负例分数大于正例

    #计算训练集上的AUC
    train_auc = 0
    if _model.train_auc:
        for train_res in train_predict:
            train_auc += (train_res > neg_predict).sum() / len(neg_predict)
        train_auc /= len(train_predict)

    #计算HR@K, NDCG@K, AUC
    hr = position < _K
    if hr:
        ndcg = math.log(2) / math.log(position+2)
    else:
        ndcg = 0
    auc = 1 - (position * 1. / len(neg_predict))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
    return hr, ndcg, auc, train_auc

def init_logging(args):
    #初始化日志
    regs = eval(args.regs)
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):
        os.makedirs(path)

    fpath = path + "%s_embed_size%.4f_lambda1%.7f_reg2%.7f%s" % (
        args.dataset, args.embed_size, regs[0], regs[1],strftime('%Y_%m_%d_%H_%M_%S', localtime()))
    logging.basicConfig(filename=fpath,
                        level=logging.INFO)
    print "log to", fpath
    logging.info("begin training %s model ......" % args.model)
    logging.info("dataset:%s  embedding_size:%d   dns:%d    batch_size:%d"
                 % (args.dataset, args.embed_size, args.dns, args.batch_size))
    print "dataset:%s  embedding_size:%d   dns:%d   batch_szie:%d" \
                 % (args.dataset, args.embed_size, args.dns, args.batch_size)
    logging.info("regs:%.8f, %.8f  learning_rate:%.4f"
                 % (regs[0], regs[1], args.lr))
    print "regs:%.8f, %.8f  learning_rate:%.4f" \
                 % (regs[0], regs[1], args.lr)
    print str(args)
    logging.info(str(args))

if __name__ == '__main__':

    #初始化logging
    args = parse_args()
    init_logging(args)

    #初始化dataset
    dataset = Dataset(args.path + args.dataset)

    #初始化models
    model = GMF(dataset.num_users, dataset.num_items, args)
    model.build_graph()

    #开始训练
    #start trainging
    #saver = GMFSaver()
    #saver.setPrefix("./param")
    training(model, dataset, args)
