# Attention is All You Need(Transformer)

逐行源码阅读中文笔记。

blog解读与复现：https://blog.csdn.net/qq_39388410/article/details/102081253

建议阅读顺序：Transformer.py-->nn.Transformer.py

# 

原paper：
```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

原code（harvardnlp的pytorch版）： http://nlp.seas.harvard.edu/2018/04/03/attention.html

# 

nn.Transformer已经已经出框架了...

所以nn.Transformer.py这个是pytorch官方实例的代码，搭起来很方便。
