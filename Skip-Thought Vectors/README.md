# Skip-Thought Vectors

源码笔记：思路和word2vec的skip类似，一个句子预测它的前一个句子和后一个句子，句子之间用lstm编码和解码就行了。

建议阅读顺序：data_loader-->vocab-->model-->train-->evaluate

#

原paper：
```
@article{kiros2015skip,
  title={Skip-Thought Vectors},
  author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
  journal={arXiv preprint arXiv:1506.06726},
  year={2015}
}
```

原code： https://github.com/ryankiros/skip-thoughts

#
原code是Tensorflow，按照作者的教程很容易训练，这是Pytorch版的实现：

http://sanyam5.github.io/my-thoughts-on-skip-thoughts/
