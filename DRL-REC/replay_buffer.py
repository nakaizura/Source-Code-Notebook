'''
Created on Apr 28, 2020
@author: nakaizura
'''

from collections import deque
from simulator import data
import random

#经验重放的记忆 M

class RelayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        for idx, row in data.iterrows():
            sample = list()
            sample.append(row['state_float'])#当前状态
            sample.append(row['action_float'])#动作
            sample.append(row['reward_float'])#奖励
            sample.append(row['n_state_float'])#接下来的状态
            self.buffer.append(sample)

    def add(self, state, action, reward, next_reward):
        #存入（状态，动作，动作的奖励，下一个状态）
        experience = (state, action, reward, next_reward)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else: #如果M满了
            self.buffer.popleft()#出队最早的记忆
            self.buffer.append(experience)#再把自己加进去

    def size(self):
        return self.count #存储M当前容量

    def sample_batch(self, batch_size):#随机采样
        return random.sample(self.buffer, batch_size)

    def clear(self):#M清零
        self.buffer.clear()
        self.count = 0
