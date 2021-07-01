'''
Created on Jul 1, 2021
@author: nakaizura
'''
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re

# 这是是对输入的video和text进行处理，video抽帧,text token化，以及多个正例对的构建。

class HT100M_DataLoader(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root='',
            caption_root='',
            min_time=4.0,
            fps=16,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            benchmark=False,
            token_to_word_path='data/dict.npy',
            max_words=20,
            num_candidates=1,
            random_left_right_flip=False,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = num_candidates
        self.random_flip = random_left_right_flip

    def __len__(self):
        return len(self.csv)

    def _get_video(self, video_path, start, end):# 对视频进行处理
        start_seek = random.randint(start, int(max(start, end - self.num_sec)))# 找到开始帧
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )# 用ffmpeg切帧
        if self.center_crop:# 中心裁剪
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)# 调整维度
        if video.shape[1] < self.num_frames:# 帧不够就补0
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _split_text(self, sentence):# 切分文本
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):# token化
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)

    def _zero_pad_tensor_token(self, tensor, size):# 补0
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))# 找到对应id

    def _find_nearest_candidates(self, caption, ind):# 找到最近的候选，即论文中的围绕某个正例前后扩充多个正例们
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1# 初始只有自己
        while n_candidate < self.num_candidates:# 扩充正例
           if start == 0:
               return 0
           elif end == len(caption) - 1:
               return start - (self.num_candidates - n_candidate)
           elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
               start -= 1
           else:
               end += 1
           n_candidate += 1
        return start

    def _get_text(self, caption):# 处理文本
        cap = pd.read_csv(caption)
        ind = random.randint(0, len(cap) - 1)
        if self.num_candidates == 1:
            words = self.words_to_ids(cap['text'].values[ind])
        else:
            words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):# 扩充多个正例
                words[i] = self.words_to_ids(cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))])
        start, end = cap['start'].values[ind], cap['end'].values[ind]
        #TODO: May need to be improved for edge cases. 
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 
        return words, int(start), int(end) 

    def __getitem__(self, idx):
        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('.')[0]
        video_path = os.path.join(self.video_root, video_file)
        text, start, end = self._get_text(os.path.join(self.caption_root, video_id + '.csv'))
        video = self._get_video(video_path, start, end)
        return {'video': video, 'text': text} # 返回视频和文本对
