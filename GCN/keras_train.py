'''
Created on Apr 25, 2020
@author: nakaizura
'''

from __future__ import print_function
#future处理新功能版本不兼容问题，加上这句话，所有的print函数将是3.x的模式（即便环境是2.x）

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import time

#定义超参数
DATASET = 'cora' #数据集，整个模型的目的是对其进行有引用关系的一堆论文做文本分类
FILTER = 'localpool'  # 过滤器是局部池化localpool或者'chebyshev'
MAX_DEGREE = 2  # 最大多项式的度
SYM_NORM = True  # 是否对称正则化
NB_EPOCH = 200 #迭代次数
PATIENCE = 10  # 早停次数（10次不变就早停）

#得到训练集，验证集和测试集。
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

#特征归一化
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':#如果是局部池化
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM) #处理有自环的邻接矩阵
    support = 1
    graph = [X, A_] #特征矩阵和邻接矩阵
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':#如果是切比雪夫多项式
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM) #归一化后的拉普拉斯矩阵
    L_scaled = rescale_laplacian(L) #重调整以简化
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE) #计算到max_degree阶的切比雪夫
    support = MAX_DEGREE + 1 #support是邻接矩阵的归一化形式
    graph = [X]+T_k 
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')



#定义模型架构。用GCN的参数作为张量列表传递。
#两层GCN
X_in = Input(shape=(X.shape[1],))

H = Dropout(0.5)(X_in)#输入维度是1433
H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = Dropout(0.5)(H)#隐层是16
#最后是要预测y的维度，这个任务是论文分类7个类别，所以是7个维度的softmax
Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

#编译模型
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

#辅助变量
wait = 0
preds = None
best_val_loss = 99999

#开始训练
for epoch in range(1, NB_EPOCH+1):
    t = time.time()

    #用被mask设为0了的node计算loss训练模型
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    #预测是在所有数据上的结果
    preds = model.predict(graph, batch_size=A.shape[0])

    #模型在验证集上的表现
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    #早停设置
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

#模型在测试集上的表现
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
