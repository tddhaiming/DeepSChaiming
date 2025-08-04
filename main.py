import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SST2Dataset  # 导入SST2数据集类
from models.transceiver import DeepSC
from utils import *
from transformers import DistilBertTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 设置随机种子，保证实验可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def val_step(net, src, trg, p, pad_idx, criterion, channel):
    trg_input = trg[:, :-1]
    trg_output = trg[:, 1:]
    src_mask, trg_mask = create_masks(src, trg_input, pad_idx)
    preds = net(src, trg_input, src_mask, trg_mask, p, channel)
    
    loss = criterion(preds.reshape(-1, preds.size(-1)), trg_output.reshape(-1))
    return loss

def validate(epoch, args, net, criterion, pad_idx, tokenizer):
    net.eval()
    total_loss = 0.0
    test_dataset = SST2Dataset(split='validation')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x
    )
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}"):
            # 使用DistilBERT的tokenizer处理文本
            inputs = tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt'
            ).to(args.device)
            src = inputs['input_ids']
            trg = src.clone()  # 自重构任务，目标等于输入
            
            loss = val_step(net, src, trg, 0.0, pad_idx, criterion, args.channel)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    net.train()
    return avg_loss

def train(epoch, args, net, optimizer, criterion, pad_idx, tokenizer, mi_net=None, mi_opt=None):
    net.train()
    total_loss = 0.0
    train_dataset = SST2Dataset(split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        if mi_opt is not None:
            mi_opt.zero_grad()
            
        # 使用DistilBERT的tokenizer处理文本
        inputs = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        ).to(args.device)
        src = inputs['input_ids']
        trg = src.clone()  # 自重构任务，目标等于输入
        
        loss = val_step(net, src, trg, args.dropout, pad_idx, criterion, args.channel)
        
        # 如果有互信息网络，添加互信息损失
        if mi_net is not None:
            mi_loss = mutual_info_loss(mi_net, src, trg, args.device)
            total_loss_val = loss + args.lamb * mi_loss
            total_loss_val.backward()
        else:
            loss.backward()
        
        optimizer.step()
        if mi_opt is not None:
            mi_opt.step()
            
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('--d-model', type=int, default=128, help='模型维度')
    parser.add_argument('--num-layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--dff', type=int, default=256, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout概率')
    parser.add_argument('--max-length', type=int, default=30, help='句子最大长度')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮次')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--lamb', type=float, default=0.5, help='互信息损失权重')
    
    # 路径参数
    parser.add_argument('--checkpoint-path', default='./checkpoints_sst2/', type=str, help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备选择')
    
    # 信道参数
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh', 'Rician'], help='信道类型')
    
    args = parser.parse_args()
    print(f"使用设备: {args.device}")
    
    # 设置随机种子
    setup_seed(10)
    
    # 初始化DistilBERT分词器
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 词表参数（使用DistilBERT内置词表）
    src_vocab_size = 30522
    trg_vocab_size = 30522  # DistilBERT-base-uncased的词表大小
    pad_idx = 0  # DistilBERT的pad token id
    start_idx = 101  # [CLS]作为起始标记
    end_idx = 102   # [SEP]作为结束标记
    
    # 初始化模型
    deepsc = DeepSC(
        num_layers=args.num_layers,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_max_len=args.max_length,
        trg_max_len=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        dropout=args.dropout
    ).to(args.device)
    
    # 互信息网络（如果需要）
    mi_net = None
    mi_opt = None
    if args.lamb > 0:
        mi_net = MutualInfoEstimator(args.d_model).to(args.device)
        mi_opt = optim.Adam(mi_net.parameters(), lr=args.lr)
    
    # 优化器和损失函数
    optimizer = optim.Adam(deepsc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # 创建模型保存目录
    checkpoint_dir = os.path.join(args.checkpoint_path, f'deepsc-{args.channel}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练
        train(epoch, args, deepsc, optimizer, criterion, pad_idx, tokenizer, mi_net, mi_opt)
        
        # 验证
        val_loss = validate(epoch, args, deepsc, criterion, pad_idx, tokenizer)
        
        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': deepsc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"保存最优模型到 {checkpoint_dir}/best_model.pth")
        
        # 每轮保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': deepsc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))
    
    print("训练完成!")

if __name__ == '__main__':
    main()
