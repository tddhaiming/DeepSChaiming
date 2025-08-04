# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# class EurDataset(Dataset):
#     def __init__(self, split='train'):
#         data_dir = '/content/drive/MyDrive/datasets/extracted_data/'
#         with open(data_dir + 'txt/{}_data.pkl'.format(split), 'rb') as f:
#             self.data = pickle.load(f)


#     def __getitem__(self, index):
#         sents = self.data[index]
#         return  sents

#     def __len__(self):
#         return len(self.data)

# def collate_data(batch):

#     batch_size = len(batch)
#     max_len = max(map(lambda x: len(x), batch))   # get the max length of sentence in current batch
#     sents = np.zeros((batch_size, max_len), dtype=np.int64)
#     sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

#     for i, sent in enumerate(sort_by_len):
#         length = len(sent)
#         sents[i, :length] = sent  # padding the questions

#     return  torch.from_numpy(sents)
class SST2Dataset(Dataset):
    def __init__(self, split='train'):
        # 加载SST2（train/validation，validation作为测试集）
        self.dataset = load_dataset('sst2')[split]
        # 过滤长度适中的句子（避免过短/过长）
        self.sentences = [item['sentence'] for item in self.dataset 
                          if 4 <= len(item['sentence'].split()) <= 30]

    def __getitem__(self, index):
        return self.sentences[index]  # 返回句子文本

    def __len__(self):
        return len(self.sentences)

def collate_data(batch):
    # 后续用DistilBERT的tokenizer处理，这里返回文本列表
    return batch