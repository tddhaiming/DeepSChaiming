# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@File: compare_channels.py
@Description: Compare performance across different channels and generate comparative visualizations
@Usage: python compare_channels.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for English display
# plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
# plt.rcParams["axes.unicode_minus"] = False

# Channel styling configuration
CHANNEL_STYLES = {
    "Rayleigh": {"color": "blue", "marker": "o", "linestyle": "-"},
    "AWGN": {"color": "red", "marker": "s", "linestyle": "--"},
    "Rician": {"color": "green", "marker": "^", "linestyle": "-."}
}


# 1. 修改根目录为你的实际路径：/content/result
def load_channel_results(drive_root="/content/result"):  # 核心修改点1
    """Load results from all channels"""
    results = {}
    if not os.path.exists(drive_root):
        raise FileNotFoundError(f"Results directory not found: {drive_root}\nRun performance.py first to generate results.")
    
    # 遍历根目录下的所有信道文件夹（如AWGN、Rayleigh等）
    for channel in os.listdir(drive_root):
        # 构建完整路径：/content/result/信道名称（如/content/result/AWGN）
        channel_dir = os.path.join(drive_root, channel)
        if os.path.isdir(channel_dir):
            # 构建JSON文件路径：/content/result/信道名称/results.json
            result_file = os.path.join(channel_dir, "results.json")
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    results[channel] = json.load(f)  # 读取JSON文件
    return results


def plot_bleu_comparison(results, drive_root):
    """Generate BLEU score comparison plot across channels"""
    save_path = f"{drive_root}/bleu_comparison.png"  # 图表保存到根目录
    plt.figure(figsize=(12, 7))
    
    for channel, data in results.items():
        style = CHANNEL_STYLES.get(channel, {"color": "black", "marker": "x"})
        plt.plot(
            data["snr"],
            data["bleu_scores"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=8,
            label=channel
        )
    
    plt.title("BLEU Score Comparison Across Channels", fontsize=16)
    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=14)
    plt.ylabel("BLEU Score", fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(next(iter(results.values()))["snr"])
    plt.ylim(0, max([max(v["bleu_scores"]) for v in results.values()]) + 0.1)
    plt.legend(fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BLEU comparison plot saved to {save_path}")


def plot_accuracy_comparison(results, drive_root):
    """Generate accuracy comparison plot across channels"""
    save_path = f"{drive_root}/accuracy_comparison.png"  # 图表保存到根目录
    plt.figure(figsize=(12, 7))
    
    for channel, data in results.items():
        style = CHANNEL_STYLES.get(channel, {"color": "black", "marker": "x"})
        plt.plot(
            data["snr"],
            data["accuracy_scores"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=8,
            label=channel
        )
    
    plt.title("Classification Accuracy Comparison Across Channels", fontsize=16)
    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(next(iter(results.values()))["snr"])
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison plot saved to {save_path}")


def plot_bar_comparison(results, metric, drive_root):
    """Generate bar plot comparison for highest SNR"""
    save_path = f"{drive_root}/{metric}_bar_comparison.png"  # 图表保存到根目录
    plt.figure(figsize=(10, 6))
    
    # Use highest SNR value for comparison
    snr_values = next(iter(results.values()))["snr"]
    target_snr = snr_values[-1]
    target_idx = snr_values.index(target_snr)
    
    channels = list(results.keys())
    scores = [results[ch][f"{metric}_scores"][target_idx] for ch in channels]
    colors = [CHANNEL_STYLES.get(ch, {"color": "black"})["color"] for ch in channels]
    
    plt.bar(channels, scores, color=colors, alpha=0.7)
    plt.title(f"{target_snr}dB {metric.capitalize()} Score Comparison", fontsize=16)
    plt.xlabel("Channel Type", fontsize=14)
    plt.ylabel(f"{metric.capitalize()} Score", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{metric.capitalize()} bar plot saved to {save_path}")


if __name__ == '__main__':
    # 2. 主函数中同样使用根目录 /content/result（与load函数保持一致）
    drive_root = "/content/result"  # 核心修改点2
    channel_results = load_channel_results(drive_root)  # 传入根目录
    
    if not channel_results:
        print(f"No channel results found in {drive_root}. Run performance.py first.")
        exit(1)
    
    # 生成的对比图表会保存到 /content/result 目录下
    plot_bleu_comparison(channel_results, drive_root)
    plot_accuracy_comparison(channel_results, drive_root)
    plot_bar_comparison(channel_results, "bleu", drive_root)
    plot_bar_comparison(channel_results, "accuracy", drive_root)
    
    print(f"All channel comparisons completed! Results saved in {drive_root}")
