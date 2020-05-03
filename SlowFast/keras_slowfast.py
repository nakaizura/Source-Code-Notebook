'''
Created on May 3, 2020
@author: nakaizura
'''

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv3D, BatchNormalization, ReLU, Add, MaxPool3D, GlobalAveragePooling3D, Concatenate, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


#一个Conv3D--BN--ReLU的模块。
def Conv_BN_ReLU(planes, kernel_size, strides=(1, 1, 1), padding='same', use_bias=False):
    return Sequential([
        Conv3D(planes, kernel_size, strides=strides, padding=padding, use_bias=use_bias),
        BatchNormalization(),
        ReLU()
    ])


#残差模块
def bottleneck(x, planes, stride=1, downsample=None, head_conv=1, use_bias=False):
    residual = x #保存原数据x
    if head_conv == 1:#只支持1和3
        x = Conv_BN_ReLU(planes, kernel_size=1, use_bias=use_bias)(x)
    elif head_conv == 3:
        x = Conv_BN_ReLU(planes, kernel_size=(3, 1, 1), use_bias=use_bias)(x)
    else:
        raise ValueError('Unsupported head_conv!!!')
    x = Conv_BN_ReLU(planes, kernel_size=(1, 3, 3), strides=(1, stride, stride), use_bias=use_bias)(x)
    x = Conv3D(planes*4, kernel_size=1, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    if downsample is not None:
        #把残差的shortcut的channel的维度统一
        residual = downsample(residual)
    x = Add()([x, residual])#加入残差
    x = ReLU()(x)#再激活一下
    return x

def datalayer(x, stride):
    return x[:, ::stride, :, :, :]


#整个SlowFast框架
def SlowFast_body(inputs, layers, block, num_classes, dropout=0.5):
    #fast和slow的输入
    #不同时序方向步长（16和2，那么平均1s采两帧和15帧左右），构造了两个有不同帧率的视频片段
    inputs_fast = Lambda(datalayer, name='data_fast', arguments={'stride':2})(inputs)
    inputs_slow = Lambda(datalayer, name='data_slow', arguments={'stride':16})(inputs)
    #输到Fast_body和Slow_body两部分
    fast, lateral = Fast_body(inputs_fast, layers, block)
    slow = Slow_body(inputs_slow, lateral, layers, block)
    x = Concatenate()([slow, fast]) #拼接结果
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation='softmax')(x) #全连接输出类别
    return Model(inputs, out)


#Fast pathway
def Fast_body(x, layers, block):
    #inplanes其实就是channel,叫法不同
    fast_inplanes = 8
    lateral = [] #这个侧向会提供给Slow
    #按论文中给的结构图，先conv1，pool1
    x = Conv_BN_ReLU(8, kernel_size=(5, 7, 7), strides=(1, 2, 2))(x)
    x = MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    #然后接4个res模块
    lateral_p1 = Conv3D(8*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_p1)
    #帧率差8倍，所以通道更小来保持轻量化
    x, fast_inplanes = make_layer_fast(x, block, 8, layers[0], head_conv=3, fast_inplanes=fast_inplanes)
    #记录lateral，因为slow和fast的输出维度不同，需要做变换之后才能在slow融合。经过实验作者认为用5x1x1卷效果最好
    lateral_res2 = Conv3D(32*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_res2)
    x, fast_inplanes = make_layer_fast(x, block, 16, layers[1], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
    lateral_res3 = Conv3D(64*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_res3)
    x, fast_inplanes = make_layer_fast(x, block, 32, layers[2], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
    lateral_res4 = Conv3D(128*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_res4)
    x, fast_inplanes = make_layer_fast(x, block, 64, layers[3], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
    x = GlobalAveragePooling3D()(x)
    return x, lateral
#Slow pathway
def Slow_body(x, lateral, layers, block):
    #lateral会提供给slow，让slow可以知道fast的处理结果
    slow_inplanes = 64 + 64//8*2
    x = Conv_BN_ReLU(64, kernel_size=(1, 7, 7), strides=(1, 2, 2))(x)
    x = MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = Concatenate()([x, lateral[0]])
    #帧率差8倍，所以通道比fast多乘8
    x, slow_inplanes = make_layer_slow(x, block, 64, layers[0], head_conv=1, slow_inplanes=slow_inplanes)
    x = Concatenate()([x, lateral[1]])
    x, slow_inplanes = make_layer_slow(x, block, 128, layers[1], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
    x = Concatenate()([x, lateral[2]])
    x, slow_inplanes = make_layer_slow(x, block, 256, layers[2], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
    x = Concatenate()([x, lateral[3]])
    x, slow_inplanes = make_layer_slow(x, block, 512, layers[3], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
    x = GlobalAveragePooling3D()(x)
    return x


def make_layer_fast(x, block, planes, blocks, stride=1, head_conv=1, fast_inplanes=8, block_expansion=4):
    #downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
    downsample = None
    #inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
    if stride != 1 or fast_inplanes != planes * block_expansion:
        downsample = Sequential([
            Conv3D(planes*block_expansion, kernel_size=1, strides=(1, stride, stride), use_bias=False),
            BatchNormalization()
        ])
    fast_inplanes = planes * block_expansion #此时通道不一致，需要扩维
    x = block(x, planes, stride, downsample=downsample, head_conv=head_conv)
    for _ in range(1, blocks):
        x = block(x, planes, head_conv=head_conv)
    return x, fast_inplanes

def make_layer_slow(x, block, planes, blocks, stride=1, head_conv=1, slow_inplanes=80, block_expansion=4):
    #和上面函数功能一致
    downsample = None
    if stride != 1 or slow_inplanes != planes * block_expansion:
        downsample = Sequential([
            Conv3D(planes*block_expansion, kernel_size=1, strides = (1, stride, stride), use_bias=False),
            BatchNormalization()
        ])
    x = block(x, planes, stride, downsample, head_conv=head_conv)
    for _ in range(1, blocks):
        x = block(x, planes, head_conv=head_conv)
    slow_inplanes = planes * block_expansion + planes * block_expansion//8*2
    return x, slow_inplanes





if __name__=="__main__":
    tf.enable_eager_execution()
    conv = Conv_BN_ReLU(8, (5, 7, 7), strides=(1, 2, 2), padding='same')
    x = tf.random_uniform([1, 32, 224, 224, 3])#随机初始化
    out = conv(x)
    out = MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(out)
    print(out.get_shape())
