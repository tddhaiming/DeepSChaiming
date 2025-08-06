import os
import sys
import argparse
import time
import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import (
    SNR_to_noise, initNetParams, train_step, val_step, train_mi,
    create_masks
)
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from dataset import SST2Dataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import sample_batch, mutual_information
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer
import pickle
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default='/content/drive/MyDrive/deepsc_sst2_checkpoints', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, choices=['AWGN', 'Rayleigh', 'Rician'])
parser.add_argument('--max-length', default=30, type=int)
parser.add_argument('--min-length', default=4, type=int)
parser.add_argument('--d-model', default=768, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--mi-lr', default=1e-3, type=float)
parser.add_argument('--resume', default='', type=str, help='Path to resume checkpoint (e.g., ./checkpoints/epoch_3.pth)')  # 新增参数

# 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin.weight = torch.nn.Parameter(torch.normal(0.0, 0.02, size=lin.weight.shape))
    if bias:
        lin.bias.data.zero_()
    return lin

class Mine(nn.Module):
    def __init__(self, in_dim=2, hidden_size=10):
        super(Mine, self).__init__()
        self.dense1 = linear(in_dim,hidden_size)
        self.dense2 = linear(hidden_size,hidden_size)
        self.dense3 = linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        output = self.dense3(x)
        return output

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def sample_batch(rec, noise):
    rec = torch.reshape(rec, shape=(-1, 1))
    noise = torch.reshape(noise, shape=(-1, 1))
    rec_sample1, rec_sample2 = torch.split(rec, int(rec.shape[0]/2), dim=0)
    noise_sample1, noise_sample2 = torch.split(noise, int(noise.shape[0]/2), dim=0)
    joint = torch.cat((rec_sample1, noise_sample1), 1)
    marg = torch.cat((rec_sample1, noise_sample2), 1)
    assert joint.shape[1] == 2, f"joint特征维度错误，实际为{joint.shape[1]}，预期为2"
    return joint, marg

def initNetParams(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

def create_masks(src, trg, padding_idx):
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    return src_mask.to(device), combined_mask.to(device)

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def loss_function(x, trg, padding_idx, criterion):
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    loss *= mask
    return loss.mean()

class Channels():
    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    channels = Channels()
    opt.zero_grad()
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine

    loss.backward()
    opt.step()
    return loss.item()

def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb
    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()
    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    return loss.item()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, tokenizer, pad_idx, criterion):
    net.eval()
    total_loss = 0.0
    test_dataset = SST2Dataset(split='validation')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_data
    )
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}"):
            inputs = tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt'
            ).to(device)
            src = inputs['input_ids']
            trg = src.clone()
            loss = val_step(
                model=net,
                src=src,
                trg=trg,
                n_var=SNR_to_noise(10),
                pad=pad_idx,
                criterion=criterion,
                channel=args.channel
            )
            total_loss += loss
    
    avg_loss = total_loss / len(test_loader)
    print(f"Validation Avg Loss: {avg_loss:.4f}")
    net.train()
    return avg_loss

def train(epoch, args, net, mi_net, optimizer, mi_optimizer, tokenizer, pad_idx, criterion):
    net.train()
    total_loss = 0.0
    train_dataset = SST2Dataset(split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_data
    )
    
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=1)[0]

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        if mi_optimizer is not None:
            mi_optimizer.zero_grad()
        
        inputs = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        ).to(device)
        src = inputs['input_ids']
        trg = src.clone()
        
        if mi_net is not None:
            mi_loss = train_mi(
                model=net,
                mi_net=mi_net,
                src=src,
                n_var=noise_std,
                padding_idx=pad_idx,
                opt=mi_optimizer,
                channel=args.channel
            )
        
        main_loss = train_step(
            model=net,
            src=src,
            trg=trg,
            n_var=noise_std,
            pad=pad_idx,
            opt=optimizer,
            criterion=criterion,
            channel=args.channel
        )
        
        if mi_net is not None:
            tqdm.write(f"Main Loss: {main_loss:.4f} | MI Loss: {mi_loss:.4f}")
        else:
            tqdm.write(f"Main Loss: {main_loss:.4f}")
        
        total_loss += main_loss
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training Avg Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    args = parser.parse_args()
    setup_seed(42)
    print(f"使用设备: {device}")
    
    # 初始化分词器
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 词表参数
    num_vocab = 30522
    pad_idx = tokenizer.pad_token_id
    start_idx = tokenizer.cls_token_id
    end_idx = tokenizer.sep_token_id
    
    # 初始化模型
    deepsc = DeepSC(
        num_layers=args.num_layers,
        src_vocab_size=num_vocab,
        trg_vocab_size=num_vocab,
        src_max_len=args.max_length,
        trg_max_len=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        dropout=0.1
    ).to(device)
    initNetParams(deepsc)
    
    # 初始化互信息网络
    mi_net = Mine(in_dim=2).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        deepsc.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    mi_optimizer = torch.optim.Adam(
        mi_net.parameters(),
        lr=args.mi_lr
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,reduction='none')
    
    # 创建模型保存目录
    checkpoint_dir = os.path.join(args.checkpoint_path, f'deepsc-{args.channel}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 恢复检查点（如果提供）
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"从检查点恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            best_val_loss = checkpoint['best_val_loss']
            deepsc.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mi_net.load_state_dict(checkpoint['mi_net_state_dict'])
            mi_optimizer.load_state_dict(checkpoint['mi_optimizer_state_dict'])
            print(f"已恢复至 epoch {start_epoch}")
        else:
            print(f"警告: 未找到检查点文件 {args.resume}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):  # 从start_epoch开始
        start_time = time.time()
        
        # 训练
        train(
            epoch=epoch,
            args=args,
            net=deepsc,
            mi_net=mi_net,
            optimizer=optimizer,
            mi_optimizer=mi_optimizer,
            tokenizer=tokenizer,
            pad_idx=pad_idx,
            criterion=criterion
        )
        
        # 验证
        val_loss = validate(
            epoch=epoch,
            args=args,
            net=deepsc,
            tokenizer=tokenizer,
            pad_idx=pad_idx,
            criterion=criterion
        )
        
        # 保存每个epoch的检查点
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': deepsc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mi_net_state_dict': mi_net.state_dict(),
            'mi_optimizer_state_dict': mi_optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, epoch_checkpoint_path)
        print(f"已保存第 {epoch+1} 个epoch的检查点至 {epoch_checkpoint_path}")
        
        # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': deepsc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mi_net_state_dict': mi_net.state_dict(),
                'mi_optimizer_state_dict': mi_optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_checkpoint_path)
            print(f"已更新最佳模型至 {best_checkpoint_path}")
        
        print(f"Epoch {epoch+1} 耗时: {time.time() - start_time:.2f}秒\n")
    
    print("训练完成!")

if __name__ == '__main__':
    main()
