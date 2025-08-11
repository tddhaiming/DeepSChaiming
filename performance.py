# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2025/8/6 (modified for multi-channel)
"""
import os
import json
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import SST2Dataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from utils import BleuScore, SNR_to_noise, greedy_decode
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 创建输出文件夹（按信道分类）
def create_output_dirs(channel, drive_root="/content/drive/MyDrive/deepsc_results"):
    # 根目录设置为Google Drive下的deepsc_results文件夹
    base_dir = f"{drive_root}/{channel}"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir, drive_root  # 返回信道目录和根目录

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='sst2/', type=str)
# 修改 checkpoint 路径为你的实际路径
parser.add_argument('--checkpoint-path', default='/content/drive/MyDrive/deepsc_sst2_checkpoints', type=str)
parser.add_argument('--channel', required=True, type=str, help="指定信道类型（如Rayleigh、AWGN、Rician）")
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=768, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--num-examples', default=5, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class SST2Dataset(Dataset):
#     def __init__(self, split='train'):
#         # 加载SST2（train/validation，validation作为测试集）
#         self.dataset = load_dataset('sst2')[split]
#         # 过滤长度适中的句子（避免过短/过长）
#         self.sentences = [item['sentence'] for item in self.dataset
#                           if 4 <= len(item['sentence'].split()) <= 30]

#     def __getitem__(self, index):
#         return self.sentences[index]  # 返回句子文本

#     def __len__(self):
#         return len(self.sentences)

# def collate_data(batch):
#     # 后续用DistilBERT的tokenizer处理，这里返回文本列表
#     return batch

class SST2Dataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = load_dataset('stanfordnlp/sst2')[split]
        self.sentences = [item['sentence'] for item in self.dataset 
                          if 4 <= len(item['sentence'].split()) <= 30]

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)

def collate_data(batch):
    return batch








def plot_bleu_vs_snr(snr_list, bleu_scores, channel, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, bleu_scores, marker='o', color='b', linewidth=2, markersize=8)
    plt.title(f'{channel}信道：不同SNR下的BLEU分数', fontsize=14)
    plt.xlabel('信噪比(dB)', fontsize=12)
    plt.ylabel('BLEU分数', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(snr_list)
    plt.ylim(0, max(bleu_scores) + 0.1)
    for x, y in zip(snr_list, bleu_scores):
        plt.text(x, y + 0.01, f'{y:.3f}', ha='center')
    plt.savefig(f'{output_dir}/bleu_vs_snr.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_snr(snr_list, acc_scores, channel, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, acc_scores, marker='s', color='g', linewidth=2, markersize=8)
    plt.title(f'{channel}信道：不同SNR下的情感分类准确率', fontsize=14)
    plt.xlabel('信噪比(dB)', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(snr_list)
    plt.ylim(0, 1.0)
    for x, y in zip(snr_list, acc_scores):
        plt.text(x, y + 0.02, f'{y:.3f}', ha='center')
    plt.savefig(f'{output_dir}/accuracy_vs_snr.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_reconstruction_examples(original, reconstructed, snr, idx, channel, output_dir):
    # 文本保存
    with open(f'{output_dir}/reconstruction_examples.txt', 'a', encoding='utf-8') as f:
        f.write(f'\n===== SNR={snr}dB 示例 {idx+1} =====\n')
        f.write(f'原始句子: {original}\n')
        f.write(f'重建句子: {reconstructed}\n')
    
    # 可视化对比
    plt.figure(figsize=(12, 4))
    plt.text(0.5, 0.7, f'原始句子: {original}', 
             ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(0.5, 0.3, f'重建句子 (SNR={snr}dB): {reconstructed}', 
             ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.axis('off')
    plt.title(f'{channel}信道：句子重建对比示例 {idx+1}', fontsize=12)
    plt.savefig(f'{output_dir}/reconstruction_example_{snr}dB_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


def performance(args, SNR, net, tokenizer, cls_model, output_dir):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    test_dataset = SST2Dataset(split='validation')
    test_iterator = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data
    )

    def decode_tokens(token_ids):
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    all_bleu = []
    all_accuracy = []
    reconstruction_examples = {snr: [] for snr in SNR}

    net.eval()
    with torch.no_grad():
        for snr in tqdm(SNR, desc=f"评估{args.channel}信道"):
            pred_sentences = []
            target_sentences = []
            noise_std = SNR_to_noise(snr)

            for batch in test_iterator:
                inputs = tokenizer(
                    batch,
                    padding='max_length',
                    truncation=True,
                    max_length=args.MAX_LENGTH,
                    return_tensors='pt'
                ).to(device)
                src = inputs['input_ids']
                target = src

                out = greedy_decode(
                    net,
                    src,
                    noise_std,
                    args.MAX_LENGTH,
                    tokenizer.pad_token_id,
                    tokenizer.cls_token_id,
                    args.channel
                )

                pred_ids = out.cpu().numpy().tolist()
                preds = [decode_tokens(ids) for ids in pred_ids]
                targets = [decode_tokens(ids.cpu().numpy()) for ids in src]

                pred_sentences.extend(preds)
                target_sentences.extend(targets)

                if len(reconstruction_examples[snr]) < args.num_examples:
                    for i in range(min(args.num_examples - len(reconstruction_examples[snr]), len(preds))):
                        reconstruction_examples[snr].append((targets[i], preds[i]))

            # 计算BLEU
            bleu = bleu_score_1gram.compute_blue_score(target_sentences, pred_sentences)
            all_bleu.append(bleu)

            # 计算分类准确率（需确保SST2Dataset能返回标签）
            correct = 0
            total = 0
            labels = test_dataset.get_labels()  # 假设数据集实现了获取标签的方法
            for i in range(0, len(pred_sentences), 32):
                batch_text = pred_sentences[i:i+32]
                batch_labels = labels[i:i+32]
                inputs = tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=args.MAX_LENGTH,
                    return_tensors='pt'
                ).to(device)
                outputs = cls_model(** inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                correct += (preds == batch_labels).sum()
                total += len(batch_labels)
            
            accuracy = correct / total if total > 0 else 0
            all_accuracy.append(accuracy)

        # 保存重建示例
        for snr in SNR:
            for idx, (original, recon) in enumerate(reconstruction_examples[snr]):
                save_reconstruction_examples(original, recon, snr, idx, args.channel, output_dir)

    # 保存结果为JSON（用于后续对比）
    results = {
        "channel": args.channel,
        "snr": SNR,
        "bleu_scores": all_bleu,
        "accuracy_scores": all_accuracy
    }
    with open(f'{output_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return np.array(all_bleu), np.array(all_accuracy)


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]
    #output_dir = create_output_dirs(args.channel)  # 按信道创建输出目录
    output_dir, drive_root = create_output_dirs(args.channel)
    # 加载tokenizer和分类模型
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        # 方案1：使用eos_token作为pad_token（推荐，DistilBERT常用做法）
        tokenizer.pad_token = tokenizer.eos_token
    cls_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased-finetuned-sst-2-english'
    ).to(device)
    cls_model.eval()

    # 初始化通信模型
    num_vocab = tokenizer.vocab_size
    deepsc = DeepSC(
        args.num_layers,
        num_vocab,
        num_vocab,
        args.MAX_LENGTH,
        args.MAX_LENGTH,
        args.d_model,
        args.num_heads,
        args.dff,
        0.1
    ).to(device)

    # 加载对应信道的最佳模型（假设best_model.pth在checkpoint路径下）
    model_path = os.path.join(args.checkpoint_path, args.channel, 'best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)  # 加载完整checkpoint
        # 提取模型参数（关键修改）
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        # 加载到模型中
        deepsc.load_state_dict(model_state_dict)
        print(f'Loaded {args.channel} best model from {model_path}')
    else:
        raise FileNotFoundError(f"No best_model.pth found for {args.channel} at {model_path}")
    


    # 评估并保存结果
    bleu_scores, acc_scores = performance(args, SNR, deepsc, tokenizer, cls_model, output_dir)
    
    # 生成单信道可视化
    plot_bleu_vs_snr(SNR, bleu_scores, args.channel, output_dir)
    plot_accuracy_vs_snr(SNR, acc_scores, args.channel, output_dir)
    
    print(f"{args.channel}信道评估完成，结果保存至 {output_dir}")
