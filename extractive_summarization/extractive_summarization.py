from os.path import join, dirname, basename
import re
import pdb
import json
import math
import pkuseg

import numpy as np

# 以下注释大部分是原作者写的，特别详细，但居然才个数位star？

class ChineseSummaryExtractor(object):
    """
    ChineseSummaryExtractor 解决如下问题：
    从中文文本中抽取关键句子，作为摘要结果展示。样例如下：
    e.g.
    >>> text = '朝鲜官方媒体12月29日确认，最高司令官金正恩将出访俄罗斯...'
    >>> summary = '朝鲜金正恩出访俄罗斯会见普京'
    
    原理简述：为每个文本中的句子分配权重，权重计算包括 tfidf 方法的权重，以及
    LDA 主题权重，以及 lead-3 得到位置权重，并在最后结合 MMR 模型对句子做筛选，
    得到抽取式摘要。（默认使用 pkuseg 的分词工具效果好）
    
    使用方法为：
    >>> import cse
    >>> cse_obj = cse.cse()
    >>> summary = cse.extract_summary(text)
    
    """
    def __init__(self, ):
        # 词性预处理
        # 词性参考 https://github.com/lancopku/pkuseg-python/blob/master/tags.txt
        self.pos_name = ['t', 's', 'f', 'm', 'n', 'v', 'a', 'z', 'd', 'l', 'j', 
                         'nr', 'ns', 'nt', 'nx', 'nz', 'vd', 'vx', 'ad', 'an']
        self.stricted_pos_name = ['a', 'n', 'j', 'nr', 'ns', 'nt', 'nx', 'nz', 
                                  'ad', 'an', 'vn', 'vd', 'vx']
        
        # 正则表达式
        self.quote_pattern = re.compile('[“"”‘’]')
        self.chinese_character_pattern = re.compile('[\u4e00-\u9fa5]')
        self.exception_char_ptn = re.compile(
            '[^ ¥～「」％－＋\n\u3000\u4e00-\u9fa5\u0021-\u007e'
            '\u00b7\u00d7\u2014\u2018\u2019\u201c\u201d\u2026'
            '\u3001\u3002\u300a\u300b\u300e\u300f\u3010\u3011'
            '\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f]')
        self._update_parentheses_ptn('{}「」[]【】()（）<>')
        self._gen_redundant_char_ptn(' -\t\n啊哈呀嘻=~\u3000')
        
        # 一、 二、等等
        self.puncs_fine_ptn = re.compile('([。！!?？…])')
        self.idf_file_path = join(dirname(__file__), "idf.txt")
        self._load_idf()
        self.seg = pkuseg.pkuseg(postag=True)  # 北大分词器
        
        # 读取停用词文件
        with open(join(dirname(__file__), 'stop_word.txt'), 
                  'r', encoding='utf8') as f:
            self.stop_words = list(set(f.read().split()))
            if '' in self.stop_words:
                self.stop_words.remove('')
            if '' not in self.stop_words:
                self.stop_words.append('\n')
                
        self._lda_prob_matrix()
        
    def _lda_prob_matrix(self):
        ''' 读取 lda 模型有关概率分布文件，并计算 unk 词的概率分布 '''
        # 读取 p(topic|word) 概率分布文件，由于 lda 模型过大，不方便加载并计算
        # 概率 p(topic|word)，所以未考虑 p(topic|doc) 概率，可能会导致不准
        # 但是，由于默认的 lda 模型 topic_num == 100，事实上，lda 模型是否在
        # 预测的文档上收敛对结果影响不大（topic_num 越多，越不影响）。
        with open(join(dirname(__file__), 'topic_word_weight.json'), 
                  'r', encoding='utf8') as f:
            self.topic_word_weight = json.load(f)
        self.word_num = len(self.topic_word_weight)
        
        # 读取 p(word|topic) 概率分布文件
        with open(join(dirname(__file__), 'word_topic_weight.json'),
                  'r', encoding='utf8') as f:
            self.word_topic_weight = json.load(f)
        self.topic_num = len(self.word_topic_weight)
        
        self._topic_prominence()  # 预计算主题突出度

    def _load_idf(self):
        with open(self.idf_file_path, 'r', encoding='utf-8') as f:
            idf_list = [line.strip().split(' ') for line in f.readlines()]
        self.idf_dict = dict()
        for item in idf_list:
            self.idf_dict.update({item[0]: float(item[1])})
        self.median_idf = sorted(self.idf_dict.values())[len(self.idf_dict) // 2]
    
    def _update_parentheses_ptn(self, parentheses):
        ''' 更新括号权重 '''
        length = len(parentheses)
        assert len(parentheses) % 2 == 0

        remove_ptn_list = []
        remove_ptn_format = '{left}[^{left}{right}]*{right}'
        for i in range(0, length, 2):
            left = re.escape(parentheses[i])
            right = re.escape(parentheses[i + 1])
            remove_ptn_list.append(remove_ptn_format.format(left=left, right=right))
        remove_ptn = '|'.join(remove_ptn_list)
        
        self.remove_parentheses_ptn = re.compile(remove_ptn)
        self.parentheses = parentheses
        
    def _gen_redundant_char_ptn(self, redundant_char):
        """ 生成 redundant_char 的正则 pattern """
        pattern_list = []
        for char in redundant_char:
            pattern_tmp = '(?<={char}){char}+'.format(char=re.escape(char))
            # pattern_tmp = re.escape(char) + '{2,}'
            pattern_list.append(pattern_tmp)
        pattern = '|'.join(pattern_list)
        self.redundant_char_ptn = re.compile(pattern)
        
    def _preprocessing_text(self, text):
        ''' 使用预处理函数去除文本中的各种杂质 '''
        # 去除中文的异常字符
        text = self.exception_char_ptn.sub('', text)
        # 去除中文的冗余字符
        text = self.redundant_char_ptn.sub('', text)
        # 去除文本中的各种括号
        length = len(text)
        while True:
            text = self.remove_parentheses_ptn.sub('', text)
            if len(text) == length:
                break
            length = len(text)
        
        if text[-1] not in '。！!?？…':
            text = text + '。'
        return text
    
    def _split_sentences(self, text, summary_length=200):
        """ 将文本切分为若干句子，并剪出有杂质的部分 """
        tmp_list = self.puncs_fine_ptn.split(text)
        sentences = [''.join(i) for i in zip(tmp_list[0::2], tmp_list[1::2])
                     if ''.join(i) != '']
        # 对每句话使用规则做处理
        new_sentences = list()
        for sen in sentences:
            sen = sen.strip('\t\n\r \u3000“"”‘’')
            # 规则 1：单句汉字字数必须大于 8 个字，小于 300 字符
            chinese_char_list = list()
            for item in self.chinese_character_pattern.finditer(sen):
                chinese_char_list.append([item.group(), item.start()])
                
            chinese_character = ''.join([item[0] for item in chinese_char_list])
            if len(chinese_character) < 8 or len(chinese_character) > summary_length:
                continue
            
            # 规则 2：把落单的、边缘的、多余的引号去除或替换
            res = self.quote_pattern.findall(sen)
            if len(res) == 1:
                # 若落单引号周围都是汉字，则替换为逗号，否则去除
                quote_idx = sen.index(res[0])
                before_char = sen[quote_idx - 1]
                matched_res = self.chinese_character_pattern.match(before_char)
                if matched_res:
                    sen = sen.replace(res[0], '，')
                else:
                    sen = sen.replace(res[0], '')
                    
            # 规则 3：把句子里的非中文字符太多的部分去除掉，任意20字符中，无汉字的去除掉
            std_sub_sen_len = 20  # 是一个标准
            sen_length = len(sen)
            if sen_length <= std_sub_sen_len:
                ratio = len(chinese_character) / sen_length
                if ratio >= 0.5:
                    new_sentences.append(sen)
            else:
                chinese_char_index_list = [item[1] for item in chinese_char_list]
                if chinese_char_index_list[0] != 0:
                    chinese_char_index_list.insert(0, 0)
                chinese_char_index_list.append(sen_length)
                
                new_sen_list = list()
                for idx in range(1, len(chinese_char_index_list)):
                    if chinese_char_index_list[idx] - chinese_char_index_list[idx - 1] >= 20:
                        matched_res = self.chinese_character_pattern.match(
                            sen[chinese_char_index_list[idx - 1]])
                        if matched_res is not None:
                            new_sen_list.append(sen[chinese_char_index_list[idx - 1]])
                    else:
                        new_sen_list.append(sen[chinese_char_index_list[idx - 1]: 
                                                chinese_char_index_list[idx]])
                
                sen = ''.join(new_sen_list)
                new_sentences.append(sen)
            
        return new_sentences
    
    def extract_summary(self, text, summary_length=200, lead_3_weight=1.2,
                        topic_theta=0.2, allow_topic_weight=True):
        """
        抽取一篇中文文本的摘要
        :param text: utf-8 编码中文文本
        :param summary_length: (int) 指定文摘的长度（软指定，有可能超出）
        :param lead_3_weight: (float) 文本的前三句的权重强调，取值必须大于1
        :param topic_theta: (float) 主题权重的权重调节因子，默认0.2，范围（0~无穷）
        :param allow_topic_weight: (bool) 考虑主题突出度，它有助于过滤与主题无关的句子
        :return: 关键短语及其权重
        """ 
        try:
            # 确保参数正确
            if lead_3_weight < 1:
                raise ValueError(
                    'the params `lead_3_weight` should not be less than 1.0')
            if len(text) <= summary_length:
                return text
            # step0: 清洗文本，去除杂质
            text = self._preprocessing_text(text)

            # step1: 分句，并逐句清理杂质
            sentences_list = self._split_sentences(
                text, summary_length=summary_length)

            # step2: 使用北大的分词器 pkuseg 做分词和词性标
            sentences_segs_dict = dict()
            counter_segs_list = list()
            for idx, sen in enumerate(sentences_list):
                sen_segs = self.seg.cut(sen)
                sentences_segs_dict.update({sen: [idx, sen_segs, list(), 0]})
                counter_segs_list.extend(sen_segs)

            # step3: 计算词频，用以 tfidf
            total_length = len(counter_segs_list)
            freq_dict = dict()
            for word_pos in counter_segs_list:
                word, pos = word_pos
                if word in freq_dict:
                    freq_dict[word][1] += 1
                else:
                    freq_dict.update({word: [pos, 1]})

            # step4: 计算每一个词的权重
            for sen, sen_segs in sentences_segs_dict.items():
                sen_segs_weights = list()
                for word_pos in sen_segs[1]:
                    word, pos = word_pos
                    if pos not in self.pos_name and word in self.stop_words:  # 虚词权重为 0
                        weight = 0.0
                    else:
                        weight = freq_dict[word][1] * self.idf_dict.get(
                            word, self.median_idf) / total_length
                    sen_segs_weights.append(weight)

                sen_segs[2] = sen_segs_weights
                sen_segs[3] = len([w for w in sen_segs_weights if w != 0]) / len(sen_segs_weights)

            # step5: 得到每个句子的权重
            for sen, sen_segs in sentences_segs_dict.items():
                # tfidf 权重
                tfidf_weight = sum(sen_segs[2]) / len(sen_segs[2])

                # 主题模型权重
                if allow_topic_weight:
                    topic_weight = 0.0
                    for item in sen_segs[1]:
                        topic_weight += self.topic_prominence_dict.get(
                            item[0], self.unk_topic_prominence_value)
                    topic_weight = topic_weight / len(sen_segs[1])
                else:
                    topic_weight = 0.0

                sen_weight = topic_weight * topic_theta + tfidf_weight
                if sen_segs[0] < 3:
                    sen_weight *= lead_3_weight
                    
                sen_segs[3] = sen_weight

            # step6: 按照 MMR 算法重新计算权重，并把不想要的过滤掉
            sentences_info_list = sorted(sentences_segs_dict.items(), 
                                         key=lambda item: item[1][3], reverse=True)

            mmr_list = list()
            for sentence_info in sentences_info_list:
                # 计算与已有句子的相似度
                sim_ratio = self._mmr_similarity(sentence_info, mmr_list)
                sentence_info[1][3] = (1 - sim_ratio) * sentence_info[1][3]
                mmr_list.append(sentence_info)

            # step7: 按重要程度进行排序，选取若干个句子作为摘要
            if len(sentences_info_list) == 1:
                return sentences_info_list[0][0]
            total_length = 0
            summary_list = list()
            for idx, item in enumerate(sentences_info_list):
                if len(item[0]) + total_length > summary_length:
                    if idx == 0:
                        return item[0]
                    else:
                        # 按序号排序
                        summary_list = sorted(
                            summary_list, key=lambda item: item[1][0])
                        summary = ''.join([item[0] for item in summary_list])
                        return summary
                else:
                    summary_list.append(item)
                    total_length += len(item[0])
                    if idx == len(sentences_info_list) - 1:
                        summary_list = sorted(
                            summary_list, key=lambda item: item[1][0])
                        summary = ''.join([item[0] for item in summary_list])
                        return summary

            return text[:summary_length]
        except Exception as e:
            print('the text is not legal. \n{}'.format(e))
            return []

    def _mmr_similarity(self, sentence_info, mmr_list):
        ''' 计算出每个句子和之前的句子有多大的相似性'''
        sim_ratio = 0.0
        notional_info = set([item[0] for item in sentence_info[1][1]
                             if item[1] in self.stricted_pos_name])
        if len(notional_info) == 0:
            return 1.0
        for sen_info in mmr_list:
            no_info = set([item[0] for item in sen_info[1][1]
                           if item[1] in self.stricted_pos_name])
            common_part = notional_info & no_info
            if sim_ratio < len(common_part) / len(notional_info):
                sim_ratio = len(common_part) / len(notional_info)
        return sim_ratio
        
    def _topic_prominence(self):
        ''' 计算每个词语的主题突出度，并保存在内存 '''
        init_prob_distribution = np.array(
            [self.topic_num for i in range(self.topic_num)])
        
        topic_prominence_dict = dict()
        for word in self.topic_word_weight:
            conditional_prob_list = list()
            for i in range(self.topic_num):
                if str(i) in self.topic_word_weight[word]:
                    conditional_prob_list.append(self.topic_word_weight[word][str(i)])
                else:
                    conditional_prob_list.append(1e-5)
            conditional_prob = np.array(conditional_prob_list)
            
            tmp_dot_log_res = np.log2(np.multiply(conditional_prob, 
                                                  init_prob_distribution))
            kl_div_sum = np.dot(conditional_prob, tmp_dot_log_res)  # kl divergence
            topic_prominence_dict.update({word: float(kl_div_sum)})
            
        tmp_list = [i[1] for i in tuple(topic_prominence_dict.items())]
        max_prominence = max(tmp_list)
        min_prominence = min(tmp_list)
        for k, v in topic_prominence_dict.items():
            topic_prominence_dict[k] = (v - min_prominence) / (max_prominence - min_prominence)
            
        self.topic_prominence_dict = topic_prominence_dict
        
        # 计算未知词汇的主题突出度，由于停用词已经预先过滤，所以这里不需要再考停用词无突出度
        tmp_prominence_list = [item[1] for item in self.topic_prominence_dict.items()]
        self.unk_topic_prominence_value = sum(tmp_prominence_list) / (2 * len(tmp_prominence_list))


if __name__ == '__main__':
    title = '巴黎圣母院大火：保安查验火警失误 现场找到7根烟头'
    text = '法国媒体最新披露，巴黎圣母院火灾当晚，第一次消防警报响起时，负责查验的保安找错了位置，因而可能贻误了救火的最佳时机。据法国BFMTV电视台报道，4月15日晚，巴黎圣母院起火之初，教堂内的烟雾报警器两次示警。当晚18时20分，值班人员响应警报前往电脑指示地点查看，但没有发现火情。20分钟后，警报再次响起，保安赶到教堂顶部确认起火。然而为时已晚，火势已迅速蔓延开来。报道援引火因调查知情者的话说，18时20分首次报警时，监控系统侦测到的失火位置准确无误。当时没有发生电脑故障，而是负责现场查验的工作人员走错了地方，因而属于人为失误。报道称，究竟是人机沟通出错，还是电脑系统指示有误，亦或是工作人员对机器提示理解不当？事发当时的具体情形尚待调查确认，以厘清责任归属。该台还证实了此前法媒的另一项爆料：调查人员在巴黎圣母院顶部施工工地上找到了7个烟头，但并未得出乱扔烟头引发火灾的结论。截至目前，警方尚未排除其它可能性。大火发生当天（15日）晚上，巴黎检察机关便以“因火灾导致过失损毁”为由展开司法调查。目前，巴黎司法警察共抽调50名警力参与调查工作。参与圣母院顶部翻修施工的工人、施工方企业负责人以及圣母院保安等30余人相继接受警方问话。此前，巴黎市共和国检察官海伊茨曾表示，目前情况下，并无任何针对故意纵火行为的调查，因此优先考虑的调查方向是意外失火。调查将是一个“漫长而复杂”的过程。现阶段，调查人员尚未排除任何追溯火源的线索。因此，烟头、短路、喷焊等一切可能引发火灾的因素都有待核实，尤其是圣母院顶部的电路布线情况将成为调查的对象。负责巴黎圣母院顶部翻修工程的施工企业负责人在接受法国电视一台新闻频道采访时表示，该公司部分员工向警方承认曾在脚手架上抽烟，此举违反了工地禁烟的规定。他对此感到遗憾，但同时否认工人吸烟与火灾存在任何直接关联。该企业负责人此前还曾在新闻发布会上否认检方关于起火时尚有工人在场的说法。他声称，火灾发生前所有在现场施工的工人都已经按点下班，因此事发时无人在场。《鸭鸣报》在其报道中称，警方还将调查教堂电梯、电子钟或霓虹灯短路的可能性。但由于教堂内的供电系统在大火中遭严重破坏，有些电路配件已成灰烬，几乎丧失了分析价值。此外，目前尚难以判定究竟是短路引发大火还是火灾造成短路。25日，即巴黎圣母院发生震惊全球的严重火灾10天后，法国司法警察刑事鉴定专家进入失火现场展开勘查取证工作，标志着火因调查的技术程序正式启动。此前，由于灾后建筑结构仍不稳定和现场积水过多，调查人员一直没有真正开始采集取样。'
    
    cse_obj = ChineseSummaryExtractor()
    summary = cse_obj.extract_summary(text, topic_theta=0.2)
    print('summary_0.2topic: ', summary)
    summary = cse_obj.extract_summary(text, topic_theta=0)
    print('summary_no_topic: ', summary)
    summary = cse_obj.extract_summary(text, topic_theta=0.5)
    print('summary_0.5topic: ', summary)
