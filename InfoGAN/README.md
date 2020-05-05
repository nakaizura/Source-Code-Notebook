# Mutual Information Generative Adversarial Networks (InfoGAN)

逐行源码阅读中文笔记。InfoGAN比起GAN的随机噪声中多了十来维的控制噪音c（如Mnist的数字类别，粗细等），然后优化互信息用了一个Q网络来得到“下界”（实际上这个网络是预测了类别c）。

blog解读：https://blog.csdn.net/qq_39388410/article/details/96306813

建议阅读顺序：model.py-->trainer.py-->main.py

#

原paper： Mutual Information Generative Adversarial Networks

原code： https://github.com/JonathanRaiman/tensorflow-infogan

#
pytorch版本：https://github.com/pianomania/infoGAN-pytorch
