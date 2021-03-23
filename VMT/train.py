'''
Created on Mar 23, 2021
@author: nakaizura
'''

import sys 
import os
import argparse
import time
import datetime
import logging
import numpy as np 
import json

import torch
import torch.nn as nn

from model import Encoder, Decoder
from utils import set_logger,read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,clip_gradient,adjust_learning_rate
from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
cc = SmoothingFunction()


#这个文件是整个工程的流程，各个细节函数需要从其他py中调用。
#大致需要：读参数--建词表--读数据--过模型--训练验证测试--存各种数据和文件。


class Arguments(): #读参数，都是从configs.yaml读出来的
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

def save_checkpoint(state, cp_file): #存cp
    torch.save(state, cp_file)


def count_paras(encoder, decoder, logging=None):
    '''
    计算模型的总参数，跑起来大概是encoder 12m，decoder 23m，一共35m的参数量
    '''
    nparas_enc = sum(p.numel() for p in encoder.parameters()) #numel函数取p的个数
    nparas_dec = sum(p.numel() for p in decoder.parameters())
    nparas_sum = nparas_enc + nparas_dec
    if logging is None:  #打印结果
        print ('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc/1e6, nparas_dec/1e6, nparas_sum/1e6))
    else:
        logging.info('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc/1e6, nparas_dec/1e6, nparas_sum/1e6))

def setup(args, clear=False):
    '''
    主要就是构建词表vocabs.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH #中文词表和英文词表的路径
    if clear: ## 删除已经有的词表
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # 构建English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'),  TRAIN_VOCAB_EN)
    # 构建Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # 设定随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id) #模型前缀名

    # 各个路径的参数
    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"
    init_epoch = 0

    #创建对应的文件夹
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## 初始化log
    set_logger(os.path.join(log_path, 'train.log'))

    ## 保存参数，即copy一份cogfigs.yaml方便复现
    with open(log_path+'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix)) #写log

    ## 构建相应的词表
    setup(args, clear=True)
    print(args.__dict__)

    # 设置src and tgt的语言，模型可以同时做中翻英或者英翻中，需要在此处设置
    src, tgt = 'en', 'zh'

    maps = {'en':args.TRAIN_VOCAB_EN, 'zh':args.TRAIN_VOCAB_ZH} #这个maps字典存的是对应词表地址
    vocab_src = read_vocab(maps[src]) #按照地址读vocab进去
    tok_src = Tokenizer(language=src, vocab=vocab_src, encoding_length=args.MAX_INPUT_LENGTH) #然后初始化tokenizer类，这个类在utils函数中，可以完成词表，encode等等的操作
    vocab_tgt = read_vocab(maps[tgt]) #tgt同理
    tok_tgt = Tokenizer(language=tgt, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size src/tgt:{}/{}'.format( len(vocab_src), len(vocab_tgt)) ) #写log

    ## 构造training, validation, and testing dataloaders，这个在dataloader.py中，得到对应batch的数据
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_src, tok_tgt), args.batch_size, args.MAX_VID_LENGTH, (src, tgt), num_workers=4, pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format( len(train_loader), len(val_loader), len(test_loader) )) #写log

    ## 初始化模型
    if args.model_type == 's2s': #seq2seq，不过目前似乎只有这一种type
        encoder = Encoder(vocab_size=len(vocab_src), embed_size=args.wordembed_dim, hidden_size=args.enc_hid_size).cuda()
        decoder = Decoder(embed_size=args.wordembed_dim, hidden_size=args.dec_hid_size, vocab_size=len(vocab_tgt)).cuda()

    #开始训练
    encoder.train()
    decoder.train()

    ## loss是交叉熵
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx).cuda()
    ## 优化器都是Adam
    dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr, weight_decay=args.weight_decay)
    enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr, weight_decay=args.weight_decay)

    count_paras(encoder, decoder, logging) #这里会打印一下总参数量

    ## 存loss
    total_train_loss, total_val_loss = [], []
    best_val_bleu, best_epoch = 0, 0

    ## 初始化时间
    zero_time = time.time()

    # 开始整个训练过程
    earlystop_flag = False #是否早停
    rising_count = 0

    for epoch in range(init_epoch, args.epochs):
        ## 开始按epoch迭代
        start_time = time.time()
        train_loss = train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch)#一个train周期，函数在下面

        val_loss, sentbleu, corpbleu = validate(val_loader, encoder, decoder, criterion)#一个验证周期，函数在下面
        end_time = time.time()

        #记录时间
        epoch_time = end_time - start_time
        total_time = end_time - zero_time
        
        logging.info('Total time used: %s Epoch %d time uesd: %s train loss: %.4f val loss: %.4f sentbleu: %.4f corpbleu: %.4f' % (
                str(datetime.timedelta(seconds=int(total_time))),
                epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, sentbleu, corpbleu))

        if corpbleu > best_val_bleu: #更新最好的结果
            best_val_bleu = corpbleu
            save_checkpoint({ 'epoch': epoch, 
                'enc_state_dict': encoder.state_dict(), 'dec_state_dict': decoder.state_dict(),
                'enc_optimizer': enc_optimizer.state_dict(), 'dec_optimizer': dec_optimizer.state_dict(),
                }, cp_file)
            best_epoch = epoch

        logging.info("Finished {0} epochs of training".format(epoch+1)) #写log

        #存loss
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

    logging.info('Best corpus bleu score {:.4f} at epoch {}'.format(best_val_bleu, best_epoch)) #写log

    ### 最好效果的模型会被存起来，之后test的时候可以用
    logging.info ('************ Start eval... ************')
    eval(test_loader, encoder, decoder, cp_file, tok_tgt, result_path)

def train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch):
    '''
    执行每个eopch的训练
    '''
    encoder.train()
    decoder.train()

    avg_loss = 0
    for cnt, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(train_loader, 1):
        # loader可以对应看dataloarder.py，cap是词表了，video是视频特征，caplen是长度方便索引，ref是无<PAD>的真实句子用于计算loss
        srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.cuda(), tgtcap.cuda(), video.cuda(), caplen_src.cuda(), caplen_tgt.cuda()

        src_out, init_hidden, vid_out = encoder(srccap, video) # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
        scores = decoder(srccap, tgtcap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH, teacher_forcing_ratio=args.teacher_ratio)

        targets = tgtcap[:, 1:] # 所有的句子从<start>开始，所以列表从1开始
        loss = criterion(scores[:, 1:].contiguous().view(-1, decoder.vocab_size), targets.contiguous().view(-1))#得到loss
        # 反向传播
        dec_optimizer.zero_grad() #梯度清零
        if enc_optimizer is not None:
            enc_optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.grad_clip is not None:
            clip_gradient(dec_optimizer, args.grad_clip)
            clip_gradient(enc_optimizer, args.grad_clip)

        # 应用梯度
        dec_optimizer.step()
        enc_optimizer.step()

        # 记录loss
        avg_loss += loss.item()

    return avg_loss/cnt

def validate(val_loader, encoder, decoder, criterion):
    '''
    Performs one epoch's validation.
    '''
    decoder.eval()  # eval mode (没有dropout和batchnorm)
    if encoder is not None:
        encoder.eval()

    references = list()  # references (真正的caption) 用于计算BLEU-4
    hypotheses = list()  # hypotheses (预测的caption)

    avg_loss = 0

    with torch.no_grad():
        # 逐Batches
        for cnt, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(val_loader, 1):
            srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.cuda(), tgtcap.cuda(), video.cuda(), caplen_src.cuda(), caplen_tgt.cuda()

            # 前向传播，可train类似
            src_out, init_hidden, vid_out = encoder(srccap, video)
            scores, pred_lengths = decoder.inference(srccap, tgtcap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH)

            targets = tgtcap[:, 1:]
            scores_copy = scores.clone()

            # 计算在val数据集上的loss
            loss = criterion(scores[:, 1:].contiguous().view(-1, decoder.vocab_size), targets.contiguous().view(-1))

            # 预测结果
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][1:pred_lengths[j]])  # 移除 pads and idx-0

            preds = temp_preds
            hypotheses.extend(preds) # preds= [1,2,3]

            tgtrefs = [ list(map(int, i.split())) for i in tgtrefs] # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]
            
            for r in tgtrefs: #tag会有多个句子，所以要保持一致算bleu
                references.append([r]) 

            assert len(references) == len(hypotheses) #强制长度一致，即一一对应关系

            avg_loss += loss.item()

        # 计算评估指数
        avg_loss = avg_loss/cnt
        corpbleu = corpus_bleu(references, hypotheses)#计算真实和预测的bleu
        sentbleu = 0
        for i, (r, h) in enumerate(zip(references, hypotheses), 1):
            sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
        sentbleu /= i

    return avg_loss, sentbleu, corpbleu

def eval(test_loader, encoder, decoder, cp_file, tok_tgt, result_path):
    '''
    测试模型
    '''
    ### 会用最好的模型来测试
    epoch = torch.load(cp_file)['epoch']
    logging.info ('Use epoch {0} as the best model for testing'.format(epoch)) #写log
    #载入best参数
    encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
    decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])
    
    decoder.eval()  # eval mode (同样没有dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    ids = list() # 句子id
    hypotheses = list()  # hypotheses (预测的caption)

    with torch.no_grad():
        # 按Batches
        for cnt, (srccap, video, caplen_src, sent_id) in enumerate(test_loader, 1):
            srccap, video, caplen_src = srccap.cuda(), video.cuda(), caplen_src.cuda()

            # 前向，跟前面的函数一致
            src_out, init_hidden, vid_out = encoder(srccap, video)
            preds, pred_lengths = decoder.beam_decoding(srccap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH, beam_size=5)

            # 预测caption
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:pred_lengths[j]])  # 移除 pads and idx-0

            preds = [tok_tgt.decode_sentence(t) for t in temp_preds] #把标号从词表中decode为句子

            hypotheses.extend(preds) # preds= [[1,2,3], ... ]

            ids.extend(sent_id)

    ## 存一个预测出来caption文件submission
    dc = dict(zip(ids, hypotheses))
    print (len(dc))

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path+'submission.json', 'w') as fp:
        json.dump(dc, fp)
    return dc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMT') #参数
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin: #载入参数
        import yaml
        args = Arguments(yaml.load(fin))
    main(args) #开始work
