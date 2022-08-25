# nakaizuta 25.AUG.2022

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel

#这是UIE的模型文件，较为简单是大模型跟两个mlp来预测span。

class UIE(ErniePretrainedModel):
    def __init__(self, encoding_model):
        super(UIE, self).__init__()
        self.encoder = encoding_model #此处制定模型为ERNIE或T5都可以
        hidden_size = self.encoder.config["hidden_size"]
        self.linear_start = paddle.nn.Linear(hidden_size, 1) #预测抽取词的开头位置
        self.linear_end = paddle.nn.Linear(hidden_size, 1) #预测抽取词的结尾位置 
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, pos_ids, att_mask):
        sequence_output, pooled_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids, #用于标记是prompt还是sentence，其他参数同bert一致
            position_ids=pos_ids,
            attention_mask=att_mask)
        start_logits = self.linear_start(sequence_output) #模型的结果通过linear预测位置概率
        start_logits = paddle.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output) #预测结尾概率
        end_logits = paddle.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob #返回位置概率
        
        
