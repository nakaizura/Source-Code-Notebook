'''
Created on Apr 30, 2020
@author: nakaizura
'''
from model import UniSkip, Encoder
from data_loader import DataLoader
from vocab import load_dictionary
from config import *
from torch import nn

from torch.autograd import Variable
import torch

class UsableEncoder:
    
    def __init__(self, loc="./saved_models/skip-best"):#导入之前训练得到的最好模型
        print("Preparing the DataLoader. Loading the word dictionary")
        self.d = DataLoader(sentences=[''], word_dict=load_dictionary('./data/dummy_corpus.txt.pkl'))
        self.encoder = None
        
        print("Loading encoder from the saved model at {}".format(loc))
        model = UniSkip()#载入模型
        model.load_state_dict(torch.load(loc, map_location=lambda storage, loc: storage))
        self.encoder = model.encoder
        if USE_CUDA:
            self.encoder.cuda(CUDA_DEVICE)#gpu
    
    def encode(self, text):
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]
        
        ret = []
        
        for chunk in chunks(text, 100):#每次往后增加词
            print("encoding chunk of size {}".format(len(chunk)))
            indices = [self.d.convert_sentence_to_indices(sentence) for sentence in chunk]
            indices = torch.stack(indices)
            indices, _ = self.encoder(indices)#编码
            indices = indices.view(-1, self.encoder.thought_size)
            indices = indices.data.cpu().numpy()
            
            ret.extend(indices)
        ret = np.array(ret)
        
        return ret

usable_encoder = UsableEncoder()

from tasks.eval_classification import *
#载入数据进行模型评估
eval_nested_kfold(usable_encoder, "MR", loc='./tasks/mr_data/', k=3, seed=1234, use_nb=False)
