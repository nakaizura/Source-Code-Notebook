'''
Created on Apr 29, 2020
@author: nakaizura
'''

import tensorflow as tf


#Dice是DIN自己独特的激活函数。认为ReLU系列的分割点都是0，这个分割点应该由数据决定。
#主要是通过改造Parametric ReLU，将alpha根据数据分布（期望和方差）来调整。
#优点：根据数据分布灵活调整阶跃变化点，具有BN的优点
#缺点：BN复杂度，比较耗时

def dice(_x, axis=-1, epsilon=0.000000001, name=''):
  #Data Adaptive Activation Function
  with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
    alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],                                  
                         initializer=tf.constant_initializer(0.0),                         
                         dtype=tf.float32)
    beta = tf.get_variable('beta'+name, _x.get_shape()[-1],                                  
                         initializer=tf.constant_initializer(0.0),                         
                         dtype=tf.float32)
  input_shape = list(_x.get_shape())

  reduction_axes = list(range(len(input_shape)))
  del reduction_axes[axis]
  broadcast_shape = [1] * len(input_shape)
  broadcast_shape[axis] = input_shape[axis]
                                                                                                                                                                            
  # case: train mode (uses stats of the current batch)
  #计算batch的均值和方差
  mean = tf.reduce_mean(_x, axis=reduction_axes)
  brodcast_mean = tf.reshape(mean, broadcast_shape)
  std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
  std = tf.sqrt(std)
  brodcast_std = tf.reshape(std, broadcast_shape)
  x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
  # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
  x_p = tf.sigmoid(beta * x_normed)
 
  
  return alphas * (1.0 - x_p) * _x + x_p * _x #根据原文中给的公式计算

def parametric_relu(_x):
  #PRELU激活函数，形式上和leakReLU很像，只是它的alpha可学习
  #alpha=0，退化成ReLU。alpha不更新，退化成Leak
  with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                         initializer=tf.constant_initializer(0.0),
                         dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5 #用alpha控制

  return pos + neg
