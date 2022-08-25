# extractive_summarization
这个中文文本摘要工具特别好用，为大家介绍一下。

主要基于规则做抽取式摘要，
* 词权重和句权重。1）tfidf词权重、词性权重（如虚词打压）；2）LDA主题权重；3）标题加权。
* 摘要句选择。1）MMR：综合分数排序，语义差异最大化，直到满足限制条件（字数达到或分数过低）。

十分简单好用。

# 
原code地址：https://github.com/dongrixinyu/extractive_summary
