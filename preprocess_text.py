# # !usr/bin/env python
# # -*- coding:utf-8 _*-
# """
# @Author: Huiqiang Xie
# @File: text_preprocess.py
# @Time: 2021/3/31 22:14
# """
# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan 30 16:44:08 2020

# @author: hx301
# """
# import unicodedata
# import re
# from w3lib.html import remove_tags
# import pickle
# import argparse
# import os
# import json
# from tqdm import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument('--input-data-dir', default='txt/en', type=str)
# parser.add_argument('--output-train-dir', default='txt/train_data.pkl', type=str)
# parser.add_argument('--output-test-dir', default='txt/test_data.pkl', type=str)
# parser.add_argument('--output-vocab', default='txt/vocab.json', type=str)

# SPECIAL_TOKENS = {
#   '<PAD>': 0,
#   '<START>': 1,
#   '<END>': 2,
#   '<UNK>': 3,
# }

# def unicode_to_ascii(s):
#     return ''.join(c for c in unicodedata.normalize('NFD', s)
#                    if unicodedata.category(c) != 'Mn')

# def normalize_string(s):
#     # normalize unicode characters
#     s = unicode_to_ascii(s)
#     # remove the XML-tags
#     s = remove_tags(s)
#     # add white space before !.?
#     s = re.sub(r'([!.?])', r' \1', s)
#     s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
#     s = re.sub(r'\s+', r' ', s)
#     # change to lower letter
#     s = s.lower()
#     return s

# def cutted_data(cleaned, MIN_LENGTH=2, MAX_LENGTH=5):
#     cutted_lines = list()
#     for line in cleaned:
#         length = len(line.split())
#         if length > MIN_LENGTH and length < MAX_LENGTH:
#             line = [word for word in line.split()]
#             cutted_lines.append(' '.join(line))
#     return cutted_lines

# def save_clean_sentences(sentence, save_path):
#     pickle.dump(sentence, open(save_path, 'wb'))
#     print('Saved: %s' % save_path)

# def process(text_path):
#     fop = open(text_path, 'r', encoding='utf8')
#     raw_data = fop.read()
#     sentences = raw_data.strip().split('\n')
#     raw_data_input = [normalize_string(data) for data in sentences]
#     raw_data_input = cutted_data(raw_data_input)
#     fop.close()

#     return raw_data_input


# def tokenize(s, delim=' ',  add_start_token=True, add_end_token=True,
#              punct_to_keep=None, punct_to_remove=None):
#     """
#     Tokenize a sequence, converting a string s into a list of (string) tokens by
#     splitting on the specified delimiter. Optionally keep or remove certain
#     punctuation marks and add start and end tokens.
#     """
#     if punct_to_keep is not None:
#         for p in punct_to_keep:
#             s = s.replace(p, '%s%s' % (delim, p))

#     if punct_to_remove is not None:
#         for p in punct_to_remove:
#             s = s.replace(p, '')

#     tokens = s.split(delim)
#     if add_start_token:
#         tokens.insert(0, '<START>')
#     if add_end_token:
#         tokens.append('<END>')
#     return tokens


# def build_vocab(sequences, token_to_idx = { }, min_token_count=1, delim=' ',
#                 punct_to_keep=None, punct_to_remove=None, ):
#     token_to_count = {}

#     for seq in sequences:
#       seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
#                       punct_to_remove=punct_to_remove,
#                       add_start_token=False, add_end_token=False)
#       for token in seq_tokens:
#         if token not in token_to_count:
#           token_to_count[token] = 0
#         token_to_count[token] += 1

#     for token, count in sorted(token_to_count.items()):
#       if count >= min_token_count:
#         token_to_idx[token] = len(token_to_idx)

#     return token_to_idx


# def encode(seq_tokens, token_to_idx, allow_unk=False):
#     seq_idx = []
#     for token in seq_tokens:
#       if token not in token_to_idx:
#         if allow_unk:
#           token = '<UNK>'
#         else:
#           raise KeyError('Token "%s" not in vocab' % token)
#       seq_idx.append(token_to_idx[token])
#     return seq_idx


# def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
#     tokens = []
#     for idx in seq_idx:
#       tokens.append(idx_to_token[idx])
#       if stop_at_end and tokens[-1] == '<END>':
#         break
#     if delim is None:
#       return tokens
#     else:
#       return delim.join(tokens)


# def main(args):
#     data_dir = '/content/drive/MyDrive/datasets/extracted_data/'
#     args.input_data_dir = data_dir + args.input_data_dir
#     args.output_train_dir = data_dir + args.output_train_dir
#     args.output_test_dir = data_dir + args.output_test_dir
#     args.output_vocab = data_dir + args.output_vocab

#     print(args.input_data_dir)
#     sentences = []
#     print('Preprocess Raw Text')
#     for fn in tqdm(os.listdir(args.input_data_dir)):
#         if not fn.endswith('.txt'): continue
#         process_sentences = process(os.path.join(args.input_data_dir, fn))
#         sentences += process_sentences

#     # remove the same sentences
#     a = {}
#     for set in sentences:
#         if set not in a:
#             a[set] = 0
#         a[set] += 1
#     sentences = list(a.keys())
#     print('Number of sentences: {}'.format(len(sentences)))
    
#     print('Build Vocab')
#     token_to_idx = build_vocab(
#         sentences, SPECIAL_TOKENS,
#         punct_to_keep=[';', ','], punct_to_remove=['?', '.']
#     )

#     vocab = {'token_to_idx': token_to_idx}
#     print('Number of words in Vocab: {}'.format(len(token_to_idx)))

#     # save the vocab
#     if args.output_vocab != '':
#         with open(args.output_vocab, 'w') as f:
#             json.dump(vocab, f)

#     print('Start encoding txt')
#     results = []
#     for seq in tqdm(sentences):
#         words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
#         tokens = [token_to_idx[word] for word in words]
#         results.append(tokens)


#     print('Writing Data')
#     train_data = results[: round(len(results) * 0.9)]
#     test_data = results[round(len(results) * 0.9):]

#     with open(args.output_train_dir, 'wb') as f:
#         pickle.dump(train_data, f)
#     with open(args.output_test_dir, 'wb') as f:
#         pickle.dump(test_data, f)

# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)

# !usr/bin/env python
# -*- coding:utf-8 _*-
from transformers import DistilBertTokenizer
import pickle
import argparse
import os

# 设置缓存路径到Drive（避免重复下载）
os.environ['HF_DATASETS_CACHE'] = '/content/drive/MyDrive/DeepSChaiming/huggingface_cache/datasets'

parser = argparse.ArgumentParser()
parser.add_argument('--output-train-dir', default='sst2/train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='sst2/test_data.pkl', type=str)

def main(args):
    # 初始化DistilBERT分词器
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 加载SST2数据集
    from datasets import load_dataset
    train_dataset = load_dataset('sst2')['train']
    test_dataset = load_dataset('sst2')['validation']  # SST2的测试集实际是validation

    # 预处理函数：分词、截断/填充
    def preprocess(sentence):
        return tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=30,  # 与原项目保持一致
            return_tensors='pt'
        )['input_ids'].squeeze(0)  # 去除batch维度

    # 创建输出目录
    os.makedirs('sst2', exist_ok=True)

    # 处理并保存训练集/测试集
    train_data = [preprocess(sent) for sent in train_dataset['sentence'] 
                 if 4 <= len(sent.split()) <= 30]
    test_data = [preprocess(sent) for sent in test_dataset['sentence'] 
                if 4 <= len(sent.split()) <= 30]

    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"预处理完成，保存到 {args.output_train_dir} 和 {args.output_test_dir}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)