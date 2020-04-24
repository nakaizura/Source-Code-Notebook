'''
Created on Apr 24, 2020
@author: nakaizura
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math


#生成模型
def generator_model():
    #对于图像，生成模型不断上采样上采样，反卷积反卷积就可以了。
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))#上采样，14*14*128
    model.add(Conv2D(64, (5, 5), padding='same'))#14*14*64
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))#上采样，28*28*64
    model.add(Conv2D(1, (5, 5), padding='same'))#28*28*1
    model.add(Activation('tanh'))
    return model

#判别模型
def discriminator_model():
    #判别器就是正常的图像分类模型。
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            ) #28*28*64
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) #展平了方便做全连接
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))#Sigmoid输出一个概率
    return model


def generator_containing_discriminator(g, d):
    #只训练生成器g
    model = Sequential()
    model.add(g)
    d.trainable = False #判别器d设置为不可以训练
    model.add(d)
    return model


def combine_images(generated_images):
    #把最后生成的一些图片合成一张大图
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]#看橫纵一共有多少张小图
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)#得到整张大图的大小
    for index, img in enumerate(generated_images):#往大图里面拼图
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    #开始训练
    #拿最简单的mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    #得到判别器和生成器模型
    d = discriminator_model()
    g = generator_model()

    #冻结判别器，只训练生成器的模型
    d_on_g = generator_containing_discriminator(g, d)
    #带nesterov的momentum优化器
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(1):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        #训练集批次
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))#100维的随机噪声
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]#批次
            generated_images = g.predict(noise, verbose=0)#用g生成图片
            if index % 20 == 0:#每20个批次就保存合并一张图
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")

            #训练鉴别器d
            X = np.concatenate((image_batch, generated_images))#将真实的和生成的合到一起
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE #真实与生成的label
            d_loss = d.train_on_batch(X, y)#鉴别器进行鉴别得到鉴别器的loss
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))

            #训练生成器g
            d.trainable = False #d不训练，只做判别
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)#得到的loss就是g的loss
            d.trainable = True #恢复成True，进行下一次训练
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    #测试过程。
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')#载入生成器的参数
    if nice:
        #生成“好”的图。即会用d判别一次得到分数，按分数高低筛选一些生成的图。
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')#载入判别器参数
        
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)#得到d的分数
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)#逆排，d分数低的是g生成的“好”图。
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)#合并成大图
    else:
        #直接通过噪声用训练好的模型生成一些图片。
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    #Image.imshow(image)
    #Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


def get_args():
    #设置模型参数。
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,default="train")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)#可调整为True，显示的效果更好
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
