'''
Created on Nov 18, 2020
@author: nakaizura
'''

import tensorflow as tf
from models.adv_crossmodal_simple_nuswide import AdvCrossModalSimple, ModelParams

def main(_):
    graph = tf.Graph()
    model_params = ModelParams() #模型所有的参数
    model_params.update() #更新文件夹路径。具体实现在adv_crossmodal_simple_nuswide

    with graph.as_default(): #默认图
        model = AdvCrossModalSimple(model_params)
    with tf.Session(graph=graph) as sess:
        model.train(sess) #开始训练
        #model.eval_random_rank()
        model.eval(sess) #开始测试


if __name__ == '__main__':
    tf.app.run() #启动图
