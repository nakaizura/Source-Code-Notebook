from asyncio import run_coroutine_threadsafe
from ntpath import join
import os
import json
import paddle.fluid as fluid
import numpy as np
import shutil
import pickle
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UIE(nn.Module):
    def __init__(self, encoding_model):
        super(UIE, self).__init__()
        self.encoder = encoding_model
        hidden_size =  768
        self.linear_start = nn.Linear(hidden_size,1)
        self.linear_end = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask, pos_ids):
        # pad = nn.ZeroPad2d(padding=(0, 495, 0, 0))
        # input_ids = pad(input_ids)
        # token_type_ids = pad(token_type_ids)
        # attention_mask = pad(attention_mask)
        # pos_ids = pad(pos_ids)
        # print(input_ids, token_type_ids, attention_mask, pos_ids)
        sequence_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask)['last_hidden_state']
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob


def convert_paddle_framework():
    model_path = './model_state.pdparams'
    model_save_path = './uieparams.pickle'

    assert os.path.exists(model_path)
    #assert os.path.exists(model_save_path)

    def load_state(path):
        print(path)
        if os.path.exists(path + '.pdopt'):
            # XXX another hack to ignore the optimizer state
            tmp = tempfile.mkdtemp()
            dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
            shutil.copy(path + '.pdparams', dst + '.pdparams')
            state = fluid.io.load_program_state(dst)
            shutil.rmtree(tmp)
        else:
            state = fluid.io.load_program_state(path)
        return state

    # state = load_state(model_path)
    # for i in state:
    #     print(i,state[i].shape)
    #pickle.dump(state, open(model_save_path, 'wb'))


def combine_torch_framework():
    #载入ernie并结合uie的框架
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
    ernie = BertModel.from_pretrained("nghuyong/ernie-3.0-base-zh")
    model = UIE(ernie)
    a = model.forward()

    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())

    torch.save(model.state_dict(), "uietorch.pth")


def paddle2torch():
    #参数对齐和赋值
    import pickle
    model_paddle = pickle.load(open('uieparams.pickle', 'rb'))
    model_torch = torch.load('uietorch.pth')

    # model_paddle_key_list = []
    # for name in model_paddle.keys():
    #     if name == 'StructuredToParameterName@@':continue
    #     model_paddle_key_list.append(name)

    # model_torch_key_list = []
    # for name in model_torch.keys():
    #     if name == 'encoder.embeddings.position_ids':continue
    #     model_torch_key_list.append(name)

    # for name_paddle, name_torch in zip(model_paddle_key_list, model_torch_key_list):
    #     print(name_paddle, '  ', name_torch)
    #     print(model_paddle[name_paddle].shape, "  ", model_torch[name_torch].shape)


    map_list={'encoder.embeddings.word_embeddings.weight': 'encoder.embeddings.word_embeddings.weight', 
    'encoder.embeddings.position_embeddings.weight': 'encoder.embeddings.position_embeddings.weight', 
    'encoder.embeddings.token_type_embeddings.weight': 'encoder.embeddings.token_type_embeddings.weight', 
    'encoder.embeddings.task_type_embeddings.weight': 'encoder.embeddings.task_type_embeddings.weight', 
    'encoder.embeddings.layer_norm.weight': 'encoder.embeddings.LayerNorm.weight', 
    'encoder.embeddings.layer_norm.bias': 'encoder.embeddings.LayerNorm.bias', 
    'encoder.encoder.layers.0.self_attn.q_proj.weight': 'encoder.encoder.layer.0.attention.self.query.weight', 
    'encoder.encoder.layers.0.self_attn.q_proj.bias': 'encoder.encoder.layer.0.attention.self.query.bias', 
    'encoder.encoder.layers.0.self_attn.k_proj.weight': 'encoder.encoder.layer.0.attention.self.key.weight', 
    'encoder.encoder.layers.0.self_attn.k_proj.bias': 'encoder.encoder.layer.0.attention.self.key.bias', 
    'encoder.encoder.layers.0.self_attn.v_proj.weight': 'encoder.encoder.layer.0.attention.self.value.weight', 
    'encoder.encoder.layers.0.self_attn.v_proj.bias': 'encoder.encoder.layer.0.attention.self.value.bias', 
    'encoder.encoder.layers.0.self_attn.out_proj.weight': 'encoder.encoder.layer.0.attention.output.dense.weight', 
    'encoder.encoder.layers.0.self_attn.out_proj.bias': 'encoder.encoder.layer.0.attention.output.dense.bias', 
    'encoder.encoder.layers.0.linear1.weight': 'encoder.encoder.layer.0.intermediate.dense.weight', 
    'encoder.encoder.layers.0.linear1.bias': 'encoder.encoder.layer.0.intermediate.dense.bias', 
    'encoder.encoder.layers.0.linear2.weight': 'encoder.encoder.layer.0.output.dense.weight', 
    'encoder.encoder.layers.0.linear2.bias': 'encoder.encoder.layer.0.output.dense.bias', 
    'encoder.encoder.layers.0.norm1.weight': 'encoder.encoder.layer.0.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.0.norm1.bias': 'encoder.encoder.layer.0.attention.output.LayerNorm.bias',
    'encoder.encoder.layers.0.norm2.weight': 'encoder.encoder.layer.0.output.LayerNorm.weight', 
    'encoder.encoder.layers.0.norm2.bias': 'encoder.encoder.layer.0.output.LayerNorm.bias', 
    'encoder.encoder.layers.1.self_attn.q_proj.weight': 'encoder.encoder.layer.1.attention.self.query.weight', 
    'encoder.encoder.layers.1.self_attn.q_proj.bias': 'encoder.encoder.layer.1.attention.self.query.bias', 
    'encoder.encoder.layers.1.self_attn.k_proj.weight': 'encoder.encoder.layer.1.attention.self.key.weight', 
    'encoder.encoder.layers.1.self_attn.k_proj.bias': 'encoder.encoder.layer.1.attention.self.key.bias', 
    'encoder.encoder.layers.1.self_attn.v_proj.weight': 'encoder.encoder.layer.1.attention.self.value.weight', 
    'encoder.encoder.layers.1.self_attn.v_proj.bias': 'encoder.encoder.layer.1.attention.self.value.bias', 
    'encoder.encoder.layers.1.self_attn.out_proj.weight': 'encoder.encoder.layer.1.attention.output.dense.weight', 
    'encoder.encoder.layers.1.self_attn.out_proj.bias': 'encoder.encoder.layer.1.attention.output.dense.bias', 
    'encoder.encoder.layers.1.linear1.weight': 'encoder.encoder.layer.1.intermediate.dense.weight', 
    'encoder.encoder.layers.1.linear1.bias': 'encoder.encoder.layer.1.intermediate.dense.bias', 
    'encoder.encoder.layers.1.linear2.weight': 'encoder.encoder.layer.1.output.dense.weight', 
    'encoder.encoder.layers.1.linear2.bias': 'encoder.encoder.layer.1.output.dense.bias', 
    'encoder.encoder.layers.1.norm1.weight': 'encoder.encoder.layer.1.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.1.norm1.bias': 'encoder.encoder.layer.1.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.1.norm2.weight': 'encoder.encoder.layer.1.output.LayerNorm.weight', 
    'encoder.encoder.layers.1.norm2.bias': 'encoder.encoder.layer.1.output.LayerNorm.bias', 
    'encoder.encoder.layers.2.self_attn.q_proj.weight': 'encoder.encoder.layer.2.attention.self.query.weight', 
    'encoder.encoder.layers.2.self_attn.q_proj.bias': 'encoder.encoder.layer.2.attention.self.query.bias', 
    'encoder.encoder.layers.2.self_attn.k_proj.weight': 'encoder.encoder.layer.2.attention.self.key.weight', 
    'encoder.encoder.layers.2.self_attn.k_proj.bias': 'encoder.encoder.layer.2.attention.self.key.bias', 
    'encoder.encoder.layers.2.self_attn.v_proj.weight': 'encoder.encoder.layer.2.attention.self.value.weight', 
    'encoder.encoder.layers.2.self_attn.v_proj.bias': 'encoder.encoder.layer.2.attention.self.value.bias', 
    'encoder.encoder.layers.2.self_attn.out_proj.weight': 'encoder.encoder.layer.2.attention.output.dense.weight', 
    'encoder.encoder.layers.2.self_attn.out_proj.bias': 'encoder.encoder.layer.2.attention.output.dense.bias', 
    'encoder.encoder.layers.2.linear1.weight': 'encoder.encoder.layer.2.intermediate.dense.weight', 
    'encoder.encoder.layers.2.linear1.bias': 'encoder.encoder.layer.2.intermediate.dense.bias',
    'encoder.encoder.layers.2.linear2.weight': 'encoder.encoder.layer.2.output.dense.weight', 
    'encoder.encoder.layers.2.linear2.bias': 'encoder.encoder.layer.2.output.dense.bias', 
    'encoder.encoder.layers.2.norm1.weight': 'encoder.encoder.layer.2.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.2.norm1.bias':  'encoder.encoder.layer.2.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.2.norm2.weight': 'encoder.encoder.layer.2.output.LayerNorm.weight', 
    'encoder.encoder.layers.2.norm2.bias': 'encoder.encoder.layer.2.output.LayerNorm.bias', 
    'encoder.encoder.layers.3.self_attn.q_proj.weight': 'encoder.encoder.layer.3.attention.self.query.weight', 
    'encoder.encoder.layers.3.self_attn.q_proj.bias': 'encoder.encoder.layer.3.attention.self.query.bias', 
    'encoder.encoder.layers.3.self_attn.k_proj.weight': 'encoder.encoder.layer.3.attention.self.key.weight', 
    'encoder.encoder.layers.3.self_attn.k_proj.bias': 'encoder.encoder.layer.3.attention.self.key.bias', 
    'encoder.encoder.layers.3.self_attn.v_proj.weight': 'encoder.encoder.layer.3.attention.self.value.weight', 
    'encoder.encoder.layers.3.self_attn.v_proj.bias': 'encoder.encoder.layer.3.attention.self.value.bias', 
    'encoder.encoder.layers.3.self_attn.out_proj.weight': 'encoder.encoder.layer.3.attention.output.dense.weight', 
    'encoder.encoder.layers.3.self_attn.out_proj.bias': 'encoder.encoder.layer.3.attention.output.dense.bias', 
    'encoder.encoder.layers.3.linear1.weight': 'encoder.encoder.layer.3.intermediate.dense.weight', 
    'encoder.encoder.layers.3.linear1.bias': 'encoder.encoder.layer.3.intermediate.dense.bias',
    'encoder.encoder.layers.3.linear2.weight': 'encoder.encoder.layer.3.output.dense.weight', 
    'encoder.encoder.layers.3.linear2.bias': 'encoder.encoder.layer.3.output.dense.bias', 
    'encoder.encoder.layers.3.norm1.weight': 'encoder.encoder.layer.3.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.3.norm1.bias':  'encoder.encoder.layer.3.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.3.norm2.weight': 'encoder.encoder.layer.3.output.LayerNorm.weight', 
    'encoder.encoder.layers.3.norm2.bias': 'encoder.encoder.layer.3.output.LayerNorm.bias', 
    'encoder.encoder.layers.4.self_attn.q_proj.weight': 'encoder.encoder.layer.4.attention.self.query.weight', 
    'encoder.encoder.layers.4.self_attn.q_proj.bias': 'encoder.encoder.layer.4.attention.self.query.bias', 
    'encoder.encoder.layers.4.self_attn.k_proj.weight': 'encoder.encoder.layer.4.attention.self.key.weight', 
    'encoder.encoder.layers.4.self_attn.k_proj.bias': 'encoder.encoder.layer.4.attention.self.key.bias', 
    'encoder.encoder.layers.4.self_attn.v_proj.weight': 'encoder.encoder.layer.4.attention.self.value.weight', 
    'encoder.encoder.layers.4.self_attn.v_proj.bias': 'encoder.encoder.layer.4.attention.self.value.bias', 
    'encoder.encoder.layers.4.self_attn.out_proj.weight': 'encoder.encoder.layer.4.attention.output.dense.weight', 
    'encoder.encoder.layers.4.self_attn.out_proj.bias': 'encoder.encoder.layer.4.attention.output.dense.bias', 
    'encoder.encoder.layers.4.linear1.weight': 'encoder.encoder.layer.4.intermediate.dense.weight', 
    'encoder.encoder.layers.4.linear1.bias': 'encoder.encoder.layer.4.intermediate.dense.bias',
    'encoder.encoder.layers.4.linear2.weight': 'encoder.encoder.layer.4.output.dense.weight', 
    'encoder.encoder.layers.4.linear2.bias': 'encoder.encoder.layer.4.output.dense.bias', 
    'encoder.encoder.layers.4.norm1.weight': 'encoder.encoder.layer.4.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.4.norm1.bias':  'encoder.encoder.layer.4.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.4.norm2.weight': 'encoder.encoder.layer.4.output.LayerNorm.weight', 
    'encoder.encoder.layers.4.norm2.bias': 'encoder.encoder.layer.4.output.LayerNorm.bias', 
    'encoder.encoder.layers.5.self_attn.q_proj.weight': 'encoder.encoder.layer.5.attention.self.query.weight', 
    'encoder.encoder.layers.5.self_attn.q_proj.bias': 'encoder.encoder.layer.5.attention.self.query.bias', 
    'encoder.encoder.layers.5.self_attn.k_proj.weight': 'encoder.encoder.layer.5.attention.self.key.weight', 
    'encoder.encoder.layers.5.self_attn.k_proj.bias': 'encoder.encoder.layer.5.attention.self.key.bias', 
    'encoder.encoder.layers.5.self_attn.v_proj.weight': 'encoder.encoder.layer.5.attention.self.value.weight', 
    'encoder.encoder.layers.5.self_attn.v_proj.bias': 'encoder.encoder.layer.5.attention.self.value.bias', 
    'encoder.encoder.layers.5.self_attn.out_proj.weight': 'encoder.encoder.layer.5.attention.output.dense.weight', 
    'encoder.encoder.layers.5.self_attn.out_proj.bias': 'encoder.encoder.layer.5.attention.output.dense.bias', 
    'encoder.encoder.layers.5.linear1.weight': 'encoder.encoder.layer.5.intermediate.dense.weight', 
    'encoder.encoder.layers.5.linear1.bias': 'encoder.encoder.layer.5.intermediate.dense.bias',
    'encoder.encoder.layers.5.linear2.weight': 'encoder.encoder.layer.5.output.dense.weight', 
    'encoder.encoder.layers.5.linear2.bias': 'encoder.encoder.layer.5.output.dense.bias', 
    'encoder.encoder.layers.5.norm1.weight': 'encoder.encoder.layer.5.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.5.norm1.bias':  'encoder.encoder.layer.5.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.5.norm2.weight': 'encoder.encoder.layer.5.output.LayerNorm.weight', 
    'encoder.encoder.layers.5.norm2.bias': 'encoder.encoder.layer.5.output.LayerNorm.bias', 
    'encoder.encoder.layers.6.self_attn.q_proj.weight': 'encoder.encoder.layer.6.attention.self.query.weight', 
    'encoder.encoder.layers.6.self_attn.q_proj.bias': 'encoder.encoder.layer.6.attention.self.query.bias', 
    'encoder.encoder.layers.6.self_attn.k_proj.weight': 'encoder.encoder.layer.6.attention.self.key.weight', 
    'encoder.encoder.layers.6.self_attn.k_proj.bias': 'encoder.encoder.layer.6.attention.self.key.bias', 
    'encoder.encoder.layers.6.self_attn.v_proj.weight': 'encoder.encoder.layer.6.attention.self.value.weight', 
    'encoder.encoder.layers.6.self_attn.v_proj.bias': 'encoder.encoder.layer.6.attention.self.value.bias', 
    'encoder.encoder.layers.6.self_attn.out_proj.weight': 'encoder.encoder.layer.6.attention.output.dense.weight', 
    'encoder.encoder.layers.6.self_attn.out_proj.bias': 'encoder.encoder.layer.6.attention.output.dense.bias', 
    'encoder.encoder.layers.6.linear1.weight': 'encoder.encoder.layer.6.intermediate.dense.weight', 
    'encoder.encoder.layers.6.linear1.bias': 'encoder.encoder.layer.6.intermediate.dense.bias',
    'encoder.encoder.layers.6.linear2.weight': 'encoder.encoder.layer.6.output.dense.weight', 
    'encoder.encoder.layers.6.linear2.bias': 'encoder.encoder.layer.6.output.dense.bias', 
    'encoder.encoder.layers.6.norm1.weight': 'encoder.encoder.layer.6.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.6.norm1.bias':  'encoder.encoder.layer.6.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.6.norm2.weight': 'encoder.encoder.layer.6.output.LayerNorm.weight', 
    'encoder.encoder.layers.6.norm2.bias': 'encoder.encoder.layer.6.output.LayerNorm.bias', 
    'encoder.encoder.layers.7.self_attn.q_proj.weight': 'encoder.encoder.layer.7.attention.self.query.weight', 
    'encoder.encoder.layers.7.self_attn.q_proj.bias': 'encoder.encoder.layer.7.attention.self.query.bias', 
    'encoder.encoder.layers.7.self_attn.k_proj.weight': 'encoder.encoder.layer.7.attention.self.key.weight', 
    'encoder.encoder.layers.7.self_attn.k_proj.bias': 'encoder.encoder.layer.7.attention.self.key.bias', 
    'encoder.encoder.layers.7.self_attn.v_proj.weight': 'encoder.encoder.layer.7.attention.self.value.weight', 
    'encoder.encoder.layers.7.self_attn.v_proj.bias': 'encoder.encoder.layer.7.attention.self.value.bias', 
    'encoder.encoder.layers.7.self_attn.out_proj.weight': 'encoder.encoder.layer.7.attention.output.dense.weight', 
    'encoder.encoder.layers.7.self_attn.out_proj.bias': 'encoder.encoder.layer.7.attention.output.dense.bias',
    'encoder.encoder.layers.7.linear1.weight': 'encoder.encoder.layer.7.intermediate.dense.weight', 
    'encoder.encoder.layers.7.linear1.bias': 'encoder.encoder.layer.7.intermediate.dense.bias',
    'encoder.encoder.layers.7.linear2.weight': 'encoder.encoder.layer.7.output.dense.weight', 
    'encoder.encoder.layers.7.linear2.bias': 'encoder.encoder.layer.7.output.dense.bias', 
    'encoder.encoder.layers.7.norm1.weight': 'encoder.encoder.layer.7.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.7.norm1.bias':  'encoder.encoder.layer.7.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.7.norm2.weight': 'encoder.encoder.layer.7.output.LayerNorm.weight', 
    'encoder.encoder.layers.7.norm2.bias': 'encoder.encoder.layer.7.output.LayerNorm.bias', 
    'encoder.encoder.layers.8.self_attn.q_proj.weight': 'encoder.encoder.layer.8.attention.self.query.weight', 
    'encoder.encoder.layers.8.self_attn.q_proj.bias': 'encoder.encoder.layer.8.attention.self.query.bias', 
    'encoder.encoder.layers.8.self_attn.k_proj.weight': 'encoder.encoder.layer.8.attention.self.key.weight', 
    'encoder.encoder.layers.8.self_attn.k_proj.bias': 'encoder.encoder.layer.8.attention.self.key.bias', 
    'encoder.encoder.layers.8.self_attn.v_proj.weight': 'encoder.encoder.layer.8.attention.self.value.weight', 
    'encoder.encoder.layers.8.self_attn.v_proj.bias': 'encoder.encoder.layer.8.attention.self.value.bias', 
    'encoder.encoder.layers.8.self_attn.out_proj.weight': 'encoder.encoder.layer.8.attention.output.dense.weight', 
    'encoder.encoder.layers.8.self_attn.out_proj.bias': 'encoder.encoder.layer.8.attention.output.dense.bias', 
    'encoder.encoder.layers.8.linear1.weight': 'encoder.encoder.layer.8.intermediate.dense.weight', 
    'encoder.encoder.layers.8.linear1.bias': 'encoder.encoder.layer.8.intermediate.dense.bias',
    'encoder.encoder.layers.8.linear2.weight': 'encoder.encoder.layer.8.output.dense.weight', 
    'encoder.encoder.layers.8.linear2.bias': 'encoder.encoder.layer.8.output.dense.bias', 
    'encoder.encoder.layers.8.norm1.weight': 'encoder.encoder.layer.8.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.8.norm1.bias':  'encoder.encoder.layer.8.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.8.norm2.weight': 'encoder.encoder.layer.8.output.LayerNorm.weight', 
    'encoder.encoder.layers.8.norm2.bias': 'encoder.encoder.layer.8.output.LayerNorm.bias', 
    'encoder.encoder.layers.9.self_attn.q_proj.weight': 'encoder.encoder.layer.9.attention.self.query.weight', 
    'encoder.encoder.layers.9.self_attn.q_proj.bias': 'encoder.encoder.layer.9.attention.self.query.bias', 
    'encoder.encoder.layers.9.self_attn.k_proj.weight': 'encoder.encoder.layer.9.attention.self.key.weight', 
    'encoder.encoder.layers.9.self_attn.k_proj.bias': 'encoder.encoder.layer.9.attention.self.key.bias', 
    'encoder.encoder.layers.9.self_attn.v_proj.weight': 'encoder.encoder.layer.9.attention.self.value.weight', 
    'encoder.encoder.layers.9.self_attn.v_proj.bias': 'encoder.encoder.layer.9.attention.self.value.bias', 
    'encoder.encoder.layers.9.self_attn.out_proj.weight': 'encoder.encoder.layer.9.attention.output.dense.weight', 
    'encoder.encoder.layers.9.self_attn.out_proj.bias': 'encoder.encoder.layer.9.attention.output.dense.bias', 
    'encoder.encoder.layers.9.linear1.weight': 'encoder.encoder.layer.9.intermediate.dense.weight', 
    'encoder.encoder.layers.9.linear1.bias': 'encoder.encoder.layer.9.intermediate.dense.bias',
    'encoder.encoder.layers.9.linear2.weight': 'encoder.encoder.layer.9.output.dense.weight', 
    'encoder.encoder.layers.9.linear2.bias': 'encoder.encoder.layer.9.output.dense.bias', 
    'encoder.encoder.layers.9.norm1.weight': 'encoder.encoder.layer.9.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.9.norm1.bias':  'encoder.encoder.layer.9.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.9.norm2.weight': 'encoder.encoder.layer.9.output.LayerNorm.weight', 
    'encoder.encoder.layers.9.norm2.bias': 'encoder.encoder.layer.9.output.LayerNorm.bias', 
    'encoder.encoder.layers.10.self_attn.q_proj.weight': 'encoder.encoder.layer.10.attention.self.query.weight', 
    'encoder.encoder.layers.10.self_attn.q_proj.bias': 'encoder.encoder.layer.10.attention.self.query.bias', 
    'encoder.encoder.layers.10.self_attn.k_proj.weight': 'encoder.encoder.layer.10.attention.self.key.weight', 
    'encoder.encoder.layers.10.self_attn.k_proj.bias': 'encoder.encoder.layer.10.attention.self.key.bias', 
    'encoder.encoder.layers.10.self_attn.v_proj.weight': 'encoder.encoder.layer.10.attention.self.value.weight', 
    'encoder.encoder.layers.10.self_attn.v_proj.bias': 'encoder.encoder.layer.10.attention.self.value.bias', 
    'encoder.encoder.layers.10.self_attn.out_proj.weight': 'encoder.encoder.layer.10.attention.output.dense.weight', 
    'encoder.encoder.layers.10.self_attn.out_proj.bias': 'encoder.encoder.layer.10.attention.output.dense.bias', 
    'encoder.encoder.layers.10.linear1.weight': 'encoder.encoder.layer.10.intermediate.dense.weight', 
    'encoder.encoder.layers.10.linear1.bias': 'encoder.encoder.layer.10.intermediate.dense.bias',
    'encoder.encoder.layers.10.linear2.weight': 'encoder.encoder.layer.10.output.dense.weight', 
    'encoder.encoder.layers.10.linear2.bias': 'encoder.encoder.layer.10.output.dense.bias', 
    'encoder.encoder.layers.10.norm1.weight': 'encoder.encoder.layer.10.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.10.norm1.bias':  'encoder.encoder.layer.10.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.10.norm2.weight': 'encoder.encoder.layer.10.output.LayerNorm.weight', 
    'encoder.encoder.layers.10.norm2.bias': 'encoder.encoder.layer.10.output.LayerNorm.bias', 
    'encoder.encoder.layers.11.self_attn.q_proj.weight': 'encoder.encoder.layer.11.attention.self.query.weight', 
    'encoder.encoder.layers.11.self_attn.q_proj.bias': 'encoder.encoder.layer.11.attention.self.query.bias', 
    'encoder.encoder.layers.11.self_attn.k_proj.weight': 'encoder.encoder.layer.11.attention.self.key.weight', 
    'encoder.encoder.layers.11.self_attn.k_proj.bias': 'encoder.encoder.layer.11.attention.self.key.bias', 
    'encoder.encoder.layers.11.self_attn.v_proj.weight': 'encoder.encoder.layer.11.attention.self.value.weight', 
    'encoder.encoder.layers.11.self_attn.v_proj.bias': 'encoder.encoder.layer.11.attention.self.value.bias', 
    'encoder.encoder.layers.11.self_attn.out_proj.weight': 'encoder.encoder.layer.11.attention.output.dense.weight', 
    'encoder.encoder.layers.11.self_attn.out_proj.bias': 'encoder.encoder.layer.11.attention.output.dense.bias', 
    'encoder.encoder.layers.11.linear1.weight': 'encoder.encoder.layer.11.intermediate.dense.weight', 
    'encoder.encoder.layers.11.linear1.bias': 'encoder.encoder.layer.11.intermediate.dense.bias',
    'encoder.encoder.layers.11.linear2.weight': 'encoder.encoder.layer.11.output.dense.weight', 
    'encoder.encoder.layers.11.linear2.bias': 'encoder.encoder.layer.11.output.dense.bias', 
    'encoder.encoder.layers.11.norm1.weight': 'encoder.encoder.layer.11.attention.output.LayerNorm.weight', 
    'encoder.encoder.layers.11.norm1.bias':  'encoder.encoder.layer.11.attention.output.LayerNorm.bias', 
    'encoder.encoder.layers.11.norm2.weight': 'encoder.encoder.layer.11.output.LayerNorm.weight', 
    'encoder.encoder.layers.11.norm2.bias': 'encoder.encoder.layer.11.output.LayerNorm.bias', 
    'encoder.pooler.dense.weight': 'encoder.pooler.dense.weight', 
    'encoder.pooler.dense.bias': 'encoder.pooler.dense.bias', 
    'linear_start.weight': 'linear_start.weight', 
    'linear_start.bias': 'linear_start.bias', 
    'linear_end.weight': 'linear_end.weight', 
    'linear_end.bias': 'linear_end.bias'}


    for name_paddle in map_list:
        #if 'linear' in name_paddle and 'weight' in name_paddle: 
        if 'encoder.encoder' in name_paddle or 'encoder.pooler' in name_paddle or 'start' in name_paddle or 'end' in name_paddle:
            model_paddle[name_paddle]=model_paddle[name_paddle].T #两者的维度Linear相反！

        print(name_paddle, '  ', map_list[name_paddle])
        print(model_paddle[name_paddle].shape, "  ", model_torch[map_list[name_paddle]].shape)
        
        model_torch[map_list[name_paddle]] = torch.from_numpy(model_paddle[name_paddle])
        assert model_paddle[name_paddle].shape == model_torch[map_list[name_paddle]].shape

    print(model_torch)
    torch.save(model_torch, "uie_paddle2torch.pth")


def get_eval(Content_input,Prompt):
    #这里要变成Content="[CLS]"+ "时间" + "[SEP]" + Content + "[SEP]"
    #Content_input="6月10日加班打车回家25元"
    #Prompt="时间"

    Content= Prompt + "[SEP]" + Content_input
    inputs = tokenizer(Content, return_tensors="pt")
    inputs['token_type_ids'][:,len(Prompt)+2:]=1
    pos_ids=torch.tensor([[i for i in range(len(inputs['input_ids'][0]))]])
    inputs['pos_ids']=pos_ids

    output_sp, output_ep = model(**inputs)
    point_s = np.argmax(output_sp[0].detach().numpy())
    point_e = np.argmax(output_ep[0].detach().numpy())
    #print(point_s,point_e)
    #print(output_sp[0][point_s],output_ep[0][point_e])
    if output_sp[0][point_s]>0.5 or output_ep[0][point_e]>0.5:
        tokens = tokenizer.tokenize(Content)
        outputs = tokens[point_s-1:point_e]
        return "".join(outputs)
    return ""


# 试用
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("./ernie-3.0-base-zh")
ernie = BertModel.from_pretrained("./ernie-3.0-base-zh",output_hidden_states = True)
#tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
#ernie = BertModel.from_pretrained("nghuyong/ernie-3.0-base-zh",output_hidden_states = True)
model = UIE(ernie).eval()
model.load_state_dict(torch.load('uie_paddle2torch.pth'))
print('---------torch-version uie loaded---------')


sentence="6月10日加班打车回家25元"
schema={'加班触发词':['时间','花费','目的地']}

res_dic={}
for trigger in schema:
    for prompt in schema[trigger]:
        res=get_eval(sentence,prompt)
        if len(res)>0:
            res_dic[prompt]=res

print(res_dic)



