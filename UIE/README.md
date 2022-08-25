# Unified Structure Generation for Universal Information Extraction (UIE)
* 逐行源码阅读中文笔记。
* paddle转pytorch过程。

## 逐行源码阅读中文笔记
模型结构上UIE并没有特别的地方，论文中UIE使用T5做backbone，再百度开源的模型中使用ERNIE3.0。

关键在于使用prompt改装各个抽取任务，做预训练。详细细节请参考原文。

整份代码最重要的是以下：

```[CLS]+ prompt + [SEP] + Content + [SEP]```

改写完成后输入到T5/ERNIE3.0即可。

## paddle转pytorch
由于百度开源的模型是基于paddlepaddle的，所以博主自己写了个程序来转换。

见convert.py



# 
原paper：
Title：
Author：Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, Hua Wu. Unified Structure Generation for Universal Information Extraction. ACL 2022.


原Demo：
https://universal-ie.github.io/
