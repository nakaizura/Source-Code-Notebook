'''
Created on May 5, 2020
@author: nakaizura
'''

from model import *
from trainer import Trainer


fe = FrontEnd()#D和Q的前端部分
#三个part
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
  i.cuda()
  i.apply(weights_init) #权重初始化

trainer = Trainer(g, fe, d, q)#开始训练
trainer.train()
