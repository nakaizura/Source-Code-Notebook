'''
Created on Apr 28, 2020
@author: nakaizura
'''

import os
import argparse
import numpy as np
import tensorflow as tf
import pprint as pp
from replay_buffer import RelayBuffer
from simulator import Simulator
from pre_process_data import recall_data
from util.logger import logger

#AC网络（实际上是用的DDPG）和Training函数都放一起了...

#状态空间：用户的历史浏览行为
#动作空间：要推荐给用户的商品列表，是多个item组成的list。注意show是用户没注意的
#reward：对这个列表的反馈（忽略、点击或购买）的平均分数

#另外注意由于是交互的，所以一轮的用户点击之后，历史的更新是去掉以前的，然后补上新的，作为新状态

class Actor(object):
    """动作家估计策略分布"""
    def __init__(self, sess, s_dim, a_dim, batch_size, output_size, weights_len, tau, learning_rate, scope="actor"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.output_size = output_size
        self.weights_len = weights_len
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator actor network 新网络，估计动作
            self.state, self.action_weights, self.len_seq = self._build_net("estimator_actor")
            self.network_params = tf.trainable_variables() #可训练参数
            # self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_actor')

            # target actor network，旧网络，定期保存估计网络中的参数
            self.target_state, self.target_action_weights, self.target_len_seq = self._build_net("target_actor")
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]
            # self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

            #用新网络定期更新target的参数。两种Update方式，soft更新会多个tau来平衡
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assign(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.a_gradient = tf.placeholder(tf.float32, [None, self.a_dim])#Actor的梯度
            #梯度前半部分从Critic新网络来的, 用于评价Actor的动作要怎么移动, 才能获得更大的 Q
            #而后半部分是从 Actor 来的, 用于说明Actor 要怎么样修改自身参数, 使得 Actor 更有可能做这个动作.
            self.params_gradients = list(
                map(
                    lambda x: tf.div(x, self.batch_size * self.a_dim),
                    tf.gradients(tf.reshape(self.action_weights, [self.batch_size, self.a_dim]),
                                 self.network_params, -self.a_gradient)
                )
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(self.params_gradients, self.network_params)
            )
            self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    @staticmethod
    def cli_value(x, v):
        #截断0
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        #greater则是比较A是否大于B,是的话返回true。where会先判断第一项是否为true,如果为true则返回a，否则b。
        return tf.where(tf.greater(x, y), x, y)

    def _gather_last_output(self, data, seq_lens):
        #得到倒数seq_lens的输出
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
        tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
        indices = tf.stack([this_range, tmp_end], axis=1)
        return tf.gather_nd(data, indices)

    def _build_net(self, scope):
        """build the tensorflow graph"""
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            len_seq = tf.placeholder(tf.int32, [None])
            #动态rnn处理序列的状态
            cell = tf.nn.rnn_cell.GRUCell(self.output_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer())
            outputs, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            outputs = self._gather_last_output(outputs, len_seq)
        return state, outputs, len_seq

    def train(self, state, a_gradient, len_seq):#喂值训练
        self.sess.run(self.optimizer, feed_dict={self.state: state, self.a_gradient: a_gradient, self.len_seq: len_seq})

    def predict(self, state, len_seq):#预测
        return self.sess.run(self.action_weights, feed_dict={self.state: state, self.len_seq: len_seq})

    def predict_target(self, state, len_seq):
        return self.sess.run(self.target_action_weights, feed_dict={self.target_state: state,
                                                                    self.target_len_seq: len_seq})

    def update_target_network(self):#两者更新target的方式
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    """评论家估计价值函数"""
    def __init__(self, sess, s_dim, a_dim, num_actor_vars, weights_len, gamma, tau, learning_rate, scope="critic"):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_actor_vars = num_actor_vars
        self.weights_len = weights_len
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.variable_scope(self.scope):
            # estimator critic network 新网络，评价动作的值
            self.state, self.action, self.q_value, self.len_seq = self._build_net("estimator_critic")
            # self.network_params = tf.trainable_variables()[self.num_actor_vars:]
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator_critic")

            # target critic network，旧网络，定期保存估计网络中的参数
            self.target_state, self.target_action, self.target_q_value, self.target_len_seq = self._build_net("target_critic")
            # self.target_network_params = tf.trainable_variables()[(len(self.network_params) + self.num_actor_vars):]
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_critic")

            #用新网络定期更新target的参数。两种Update方式，soft更新会多个tau来平衡
            self.update_target_network_params = [
                self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1 - self.tau)
                ) for i in range(len(self.target_network_params))
            ]
            self.hard_update_target_network_params = [
                self.target_network_params[i].assgin(
                    self.network_params[i]
                ) for i in range(len(self.target_network_params))
            ]

            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            #Critic的估计值进行均方误差就可以直接更新Critic网络
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.a_gradient = tf.gradients(self.q_value, self.action)
     #功能函数和Actor基本一致
    @staticmethod
    def cli_value(x, v):
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.where(tf.greater(x, y), x, y)

    def _gather_last_output(self, data, seq_lens):
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype=tf.int64)
        tmp_end = tf.map_fn(lambda x: self.cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
        indices = tf.stack([this_range, tmp_end], axis=1)
        return tf.gather_nd(data, indices)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.s_dim], "state")
            state_ = tf.reshape(state, [-1, self.weights_len, int(self.s_dim / self.weights_len)])
            action = tf.placeholder(tf.float32, [None, self.a_dim], "action")
            len_seq = tf.placeholder(tf.int64, [None], name="critic_len_seq")
            cell = tf.nn.rnn_cell.GRUCell(self.weights_len,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.initializers.random_normal(),
                                          bias_initializer=tf.zeros_initializer()
                                          )
            out_state, _ = tf.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=len_seq)
            out_state = self._gather_last_output(out_state, len_seq)

            inputs = tf.concat([out_state, action], axis=-1)
            layer1 = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
            layer2 = tf.layers.Dense(16, activation=tf.nn.relu)(layer1)
            q_value = tf.layers.Dense(1)(layer2)
            return state, action, q_value, len_seq

    def train(self, state, action, predicted_q_value, len_seq):
        return self.sess.run([self.q_value, self.loss, self.optimizer], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.len_seq: len_seq
        })

    def predict(self, state, action, len_seq):
        return self.sess.run(self.q_value, feed_dict={self.state: state, self.action: action, self.len_seq: len_seq})

    def predict_target(self, state, action, len_seq):
        return self.sess.run(self.target_q_value, feed_dict={self.target_state: state, self.target_action: action,
                                                             self.len_seq: len_seq})

    def action_gradients(self, state, action, len_seq):
        return self.sess.run(self.a_gradient, feed_dict={self.state: state, self.action: action, self.len_seq: len_seq})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def hard_update_target_network(self):
        self.sess.run(self.hard_update_target_network_params)


class OUNoise:
    """动作中加入噪声以增加随机性，探索性"""
    def __init__(self, a_dim, mu=0, theta=0.5, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.a_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.a_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state


def gene_actions(item_space, weight_batch):
    """根据动作的输出计算动作列表
    Args:
        item_space: recall items, dict: id: embedding
        weight_batch: actor network outputs
    Returns:
        recommendation list
    """
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    max_ids = list()
    for weight in weight_batch:
        score = np.dot(item_weights, weight)
        idx = np.argmax(score)#取得分最高的动作作为动作
        max_ids.append(item_ids[idx])#在商品空间里面找相应的item
    return max_ids


def gene_action(item_space, weight):
    #得到不是一个列表list，而是最高分数的item作为推荐
    item_ids = list(item_space.keys())
    item_weights = list(item_space.values())
    score = np.dot(item_weights, weight)
    idx = np.argmax(score)
    return item_ids[idx]


def build_summaries():
    #写summary
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("reward", episode_reward)
    episode_max_q = tf.Variable(0.)
    tf.summary.scalar("max_q_value", episode_max_q)
    critic_loss = tf.Variable(0.)
    tf.summary.scalar("critic_loss", critic_loss)

    summary_vars = [episode_reward, episode_max_q, critic_loss]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars


def learn_from_batch(replay_buffer, batch_size, actor, critic, item_space, action_len, s_dim, a_dim):
    #训练DDPG
    if replay_buffer.size() < batch_size:
        pass
    samples = replay_buffer.sample_batch(batch_size)
    state_batch = np.asarray([_[0] for _ in samples])
    action_batch = np.asarray([_[1] for _ in samples])
    reward_batch = np.asarray([_[2] for _ in samples])
    n_state_batch = np.asarray([_[3] for _ in samples])

    #计算预测的q值
    action_weights = actor.predict_target(state_batch)#从状态预测动作
    n_action_batch = gene_actions(item_space, action_weights, action_len)#根据动作得分生成具体的n个列表
    target_q_batch = critic.predict_target(n_state_batch.reshape((-1, s_dim)), n_action_batch.reshape((-1, a_dim)))#计算动作的Q值
    y_batch = []
    for i in range(batch_size):
        y_batch.append(reward_batch[i] + critic.gamma * target_q_batch[i])

    #有了Q值直接训练Critic
    q_value, critic_loss, _ = critic.train(state_batch, action_batch, np.reshape(y_batch, (batch_size, 1)))
    #再利用Critic训练Actor
    action_weight_batch_for_gradients = actor.predict(state_batch) #从状态预测动作
    action_batch_for_gradients = gene_actions(item_space, action_weight_batch_for_gradients, action_len)#根据动作得分生成具体的n个列表
    a_gradient_batch = critic.action_gradients(state_batch, action_batch_for_gradients.reshape((-1, a_dim)))#得到Critic的评估
    actor.train(state_batch, a_gradient_batch[0])

    #再更新target网络
    actor.update_target_network()
    critic.update_target_network()

    return np.amax(q_value), critic_loss


def train(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args):
    #启动summary
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    #初始化target network权重
    actor.hard_update_target_network()
    critic.hard_update_target_network()

    #初始化 replay memory
    replay_buffer = RelayBuffer(int(args['buffer_size']))

    for i in range(int(args['max_episodes'])):#迭代周期
        ep_reward = 0.
        ep_q_value = 0.
        loss = 0.
        item_space = recall_data #这个空间是得到一个最大得分action，可对应在该空间选择一个item
        state = env.reset()
        # update average parameters every 1000 episodes
        if (i + 1) % 10 == 0:
            env.rewards, env.group_sizes, env.avg_states, env.avg_actions = env.avg_group()
        for j in range(args['max_episodes_len']):
            weight = actor.predict(np.reshape(state, [1, s_dim])) + exploration_noise.noise().reshape(
                (1, int(args['action_item_num']), int(a_dim / int(args['action_item_num'])))
            )#观察环境得到动作得分
            action = gene_actions(item_space, weight, int(args['action_item_num']))#根据得分在item space选item
            reward, n_state = env.step(action[0])#得到reward和下一个状态
            replay_buffer.add(list(state.reshape((s_dim,))),
                              list(action.reshape((a_dim,))),
                              [reward],
                              list(n_state.reshape((s_dim,))))#写入记忆库M
            ep_reward += reward
            ep_q_value_, critic_loss = learn_from_batch(replay_buffer, args['batch_size'], actor, critic, item_space,
                                                        args['action_item_num'], s_dim, a_dim)
            ep_q_value += ep_q_value_#统计回合Q值
            loss += critic_loss #loss
            state = n_state #更新当前状态
            if (j + 1) % 50 == 0:
                logger.info("=========={0} episode of {1} round: {2} reward=========".format(i, j, ep_reward))
            summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward,
                                                           summary_vars[1]: ep_q_value,
                                                           summary_vars[2]: loss})
            writer.add_summary(summary_str, i)

    writer.close()
    saver = tf.train.Saver()
    ckpt_path = os.path.join(os.path.dirname(__file__), "models")
    saver.save(sess, ckpt_path, write_meta_graph=False)


def main(args):
    # init memory data
    # data = load_data()
    with tf.Session() as sess:
        #先模拟环境，生成一堆数据
        env = Simulator()
        s_dim = int(args['embedding']) * int(args['state_item_num'])
        a_dim = int(args['embedding']) * int(args['action_item_num'])

        #实例化DDPG
        actor = Actor(sess, s_dim, a_dim,
                      int(args['batch_size']), int(args['embedding']),
                      int(args['action_item_num']), float(args['tau']),
                      float(args['actor_lr']))

        critic = Critic(sess, s_dim, a_dim,
                        actor.get_num_trainable_vars(), float(args['gamma']),
                        float(args['tau']), float(args['critic_lr']))

        exploration_noise = OUNoise(a_dim)#加探索噪音

        #开始训练
        train(sess, env, actor, critic, exploration_noise, s_dim, a_dim, args)


if __name__ == '__main__':
    #设置一堆参数
    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--embedding", help="dimension of item embedding", default=30)
    parser.add_argument("--state_item_num", help="click history list length for user", default=12)
    parser.add_argument("--action_item_num", help="length of the recommendation item list", default=4)
    parser.add_argument("--actor_lr", help="actor network learning rate", default=0.0001)
    parser.add_argument("--critic_lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer_size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--batch_size", help="size of minibatch for minbatch-SGD", default=64)

    # run parameters
    parser.add_argument("--max_episodes", help="max num of episodes to do while training", default=50000)
    parser.add_argument("--max_episodes_len", help="max length of 1 episode", default=100)
    parser.add_argument("--summary_dir", help="directory for storing tensorboard info", default='./results')

    args_ = vars(parser.parse_args())
    logger.info(pp.pformat(args_))

    main(args_) #参数导入，开始训练
