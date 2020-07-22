'''
Created on Jul 22, 2020
@author: nakaizura
'''

import numpy as np

import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D


def get_sport_clip(clip_name, verbose=True):
    """
    载入视频片段给C3D做分类。
    """
    #载入并中心crop一下
    clip = sorted(glob(join('data', clip_name, '*.png')))#视频由多个图片帧组成的
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose: #如果为True就显示视频
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # 变换维度为ch, fr, h, w，即通道 帧 高 宽
    clip = np.expand_dims(clip, axis=0)  # 增加一维batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip) #(n, ch, fr, h, w)


def read_labels_from_file(filepath):
    """
    读入真实的标签，这里用的是Sport1M，所以返回的都是动作标签。
    """
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():
    """
    主函数
    """

    # 载入视频片段做预测
    X = get_sport_clip('roger') #roger视频
    X = Variable(X)
    X = X.cuda() #GPU

    # 载入预训练好了的模型权重
    net = C3D() #模型实例化
    net.load_state_dict(torch.load('c3d.pickle')) #填入权重
    net.cuda()
    net.eval()# 调到测试模式

    # 然后直接拿网络预测就好
    prediction = net(X)
    prediction = prediction.data.cpu().numpy()

    # 读入真实标签
    labels = read_labels_from_file('labels.txt')

    # 得到topN的预测类别
    top_inds = prediction[0].argsort()[::-1][:5]
    print('\nTop 5:')
    for i in top_inds:
        print('{:.5f} {}'.format(prediction[0][i], labels[i]))


if __name__ == '__main__':
    main()
