#!/usr/bin/env python3
"""简单的Gini系数和Pareto@20计算脚本"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from datasets.coco import CocoDetection
from configs.dino_pp.dino_pp_resnet50_800_1333 import model
from util.utils import load_state_dict, load_checkpoint

# ========== 命令行参数解析 ==========
parser = argparse.ArgumentParser(description='Compute Gini and Pareto@20 for DETR models')
parser.add_argument('--model_path', type=str, default="/root/autodl-tmp/paq-dino/best_ap.pth",
                    help='Path to model checkpoint')
parser.add_argument('--threshold', type=float, default=0.3,
                    help='Activation threshold')
parser.add_argument('--num_samples', type=int, default=5000,
                    help='Number of validation samples to process')
parser.add_argument('--coco_path', type=str, default="/root/autodl-tmp/COCO2017",
                    help='Path to COCO dataset')
parser.add_argument('--output_dir', type=str, default="gini_results",
                    help='Output directory for results and plots')
parser.add_argument('--model_name', type=str, default=None,
                    help='Model name for labeling (e.g., paq-dino, orig-dino)')
args = parser.parse_args()

# ========== 参数配置 ==========
ACTIVATION_THRESHOLD = args.threshold
NUM_SAMPLES = args.num_samples
MODEL_PATH = args.model_path
COCO_PATH = args.coco_path
OUTPUT_DIR = args.output_dir

# 自动推断模型名称
if args.model_name:
    MODEL_NAME = args.model_name
elif 'paq-dino' in MODEL_PATH:
    MODEL_NAME = 'paq-dino'
elif 'orig-dino' in MODEL_PATH:
    MODEL_NAME = 'orig-dino'
else:
    MODEL_NAME = 'unknown'
# =============================

# 保存log到文件
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = f"{OUTPUT_DIR}/{MODEL_NAME}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_f = open(log_file, 'w', buffering=1)  # 行缓冲

def log_print(msg):
    """同时输出到终端和文件"""
    print(msg)
    log_f.write(msg + '\n')
    log_f.flush()

log_print(f"Log file: {log_file}")
log_print(f"Model: {MODEL_NAME}")
log_print(f"Model path: {MODEL_PATH}")

def compute_gini(counts):
    counts = np.array(counts)
    n = len(counts)
    sorted_counts = np.sort(counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    return gini

def compute_pareto(counts, percent=20):
    counts = np.array(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    sorted_counts = np.sort(counts)[::-1]
    top_k = int(len(counts) * percent / 100)
    top_k_sum = sorted_counts[:top_k].sum()
    return top_k_sum / total

log_print(f"Threshold: {ACTIVATION_THRESHOLD}, Samples: {NUM_SAMPLES}")

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_print(f"Device: {device}")

weight = load_checkpoint(MODEL_PATH)
load_state_dict(model, weight)
model = model.to(device).eval()
log_print("Model loaded")

# 加载数据
coco_val = CocoDetection(
    img_folder=f"{COCO_PATH}/val2017",
    ann_file=f"{COCO_PATH}/annotations/instances_val2017.json",
    transforms=None,
    train=False
)

def simple_collate(batch):
    images, targets = [], []
    for img, target in batch:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        images.append(img)
        targets.append(target)
    return list(zip(images, targets))

val_loader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=2, collate_fn=simple_collate)
log_print(f"Dataset: {len(coco_val)} images\n")

# 推理
query_activation_counts = torch.zeros(900)

with torch.no_grad():
    for idx, batch in enumerate(tqdm(val_loader, total=min(NUM_SAMPLES, len(coco_val)))):
        if idx >= NUM_SAMPLES:
            break
        
        img_data, target = batch[0]
        images = [img_data.to(device)]
        outputs = model(images)[0]
        
        pred_logits = outputs["pred_logits"][0]
        probs = pred_logits.sigmoid()
        max_probs, _ = probs.max(dim=-1)
        activated = (max_probs > ACTIVATION_THRESHOLD).float()
        query_activation_counts += activated.cpu()

# 计算指标
activation_counts_np = query_activation_counts.numpy()
gini = compute_gini(activation_counts_np)
pareto_20 = compute_pareto(activation_counts_np, 20)

log_print(f"\n{'='*60}")
log_print(f"Gini: {gini:.3f}")
log_print(f"Pareto@20: {pareto_20:.3f}")
log_print(f"{'='*60}\n")

# ========== 自检：确认原始数据没有被排序 ==========
log_print("🔍 Self-check: Verify original data is NOT sorted")
log_print(f"   First 10 queries (0-9): {activation_counts_np[:10]}")
log_print(f"   Random 10 queries (100-109): {activation_counts_np[100:110]}")
log_print(f"   Random 10 queries (500-509): {activation_counts_np[500:510]}")
log_print(f"   Last 10 queries (890-899): {activation_counts_np[-10:]}")

# 检查是否意外排序了（如果前10个平均值明显大于后面，可能有问题）
avg_first_10 = activation_counts_np[:10].mean()
avg_mid_10 = activation_counts_np[100:110].mean()
avg_last_10 = activation_counts_np[-10:].mean()
log_print(f"\n   Avg (first 10): {avg_first_10:.2f}")
log_print(f"   Avg (mid 10): {avg_mid_10:.2f}")
log_print(f"   Avg (last 10): {avg_last_10:.2f}")

if avg_first_10 > avg_mid_10 * 2 and avg_first_10 > avg_last_10 * 2:
    log_print("   ⚠️  WARNING: First 10 queries have much higher activation!")
    log_print("   ⚠️  This might indicate accidental sorting or position bias.")
else:
    log_print("   ✓ Data appears to be in original order (no obvious sorting)")
log_print("")
# =====================================================

# 保存结果
results_file = f"{OUTPUT_DIR}/{MODEL_NAME}_results_gini_{gini:.3f}_pareto_{pareto_20:.3f}.txt"
with open(results_file, 'w') as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Model path: {MODEL_PATH}\n")
    f.write(f"Gini: {gini:.3f}\n")
    f.write(f"Pareto@20: {pareto_20:.3f}\n")
    f.write(f"Threshold: {ACTIVATION_THRESHOLD}\n")
    f.write(f"Samples: {min(NUM_SAMPLES, len(coco_val))}\n")
log_print(f"Results saved: {results_file}")

# 绘图 - 图1: 排序后的分布（Ranked）
sorted_counts = np.sort(activation_counts_np)[::-1]
query_indices_ranked = np.arange(len(sorted_counts))

log_print("📊 Plotting Fig.1 (Ranked):")
log_print(f"   Using sorted_counts (descending)")
log_print(f"   Top 5 values: {sorted_counts[:5]}")
log_print(f"   Last 5 values: {sorted_counts[-5:]}")

fig, ax = plt.subplots(figsize=(1.6, 1.3))
# 使用scatter plot代替line plot，避免虚假的趋势感
ax.scatter(query_indices_ranked, sorted_counts, s=1, color='#2ca02c', alpha=0.6)  # 绿色点（最小）
ax.set_yscale('log')
ax.axvline(x=int(len(sorted_counts) * 0.2), color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='Top 20%')
ax.set_xlabel('Query Index (Ranked)', fontsize=5)
ax.set_ylabel('Activation Count', fontsize=5)
# ax.set_title(f'{gini:.3f}, {pareto_20:.3f}', fontsize=6)
ax.grid(True, alpha=0.3, linewidth=0.3)
ax.legend(fontsize=4)
ax.tick_params(axis='both', which='major', labelsize=4)

# 设置Y轴范围：从10^0到10^4.5
ax.set_ylim(1e0, 3e4)
ax.set_xlim(0, len(sorted_counts))

plot_file_ranked = f"{OUTPUT_DIR}/{MODEL_NAME}_plot_ranked_gini_{gini:.3f}_pareto_{pareto_20:.3f}.png"
plt.savefig(plot_file_ranked, dpi=300, bbox_inches='tight')
plt.close()
log_print(f"Ranked plot saved: {plot_file_ranked}")

# 绘图 - 图2: 未排序的分布（Original Query Order）
query_indices_original = np.arange(len(activation_counts_np))

log_print("\n📊 Plotting Fig.2 (Original Query Order):")
log_print(f"   Using activation_counts_np (ORIGINAL order, NOT sorted)")
log_print(f"   First 5 values (index 0-4): {activation_counts_np[:5]}")
log_print(f"   Middle 5 values (index 450-454): {activation_counts_np[450:455]}")
log_print(f"   Last 5 values (index 895-899): {activation_counts_np[-5:]}")

# 双重检查：确保这不是排序后的数组
if np.array_equal(activation_counts_np, sorted_counts):
    log_print("   ❌ ERROR: activation_counts_np is IDENTICAL to sorted_counts!")
    log_print("   ❌ This means Fig.2 will show sorted data, not original order!")
else:
    log_print("   ✓ Confirmed: activation_counts_np ≠ sorted_counts (good!)")

fig, ax = plt.subplots(figsize=(1.6, 1.3))
# 使用scatter plot代替line plot，避免虚假的趋势感
ax.scatter(query_indices_original, activation_counts_np, s=1, color='#2ca02c', alpha=0.6)  # 绿色点（最小）
ax.set_yscale('log')
ax.set_xlabel('Query Index', fontsize=5)
ax.set_ylabel('Activation Count (log)', fontsize=5)
# No title for this plot
ax.grid(True, alpha=0.3, linewidth=0.3)
ax.tick_params(axis='both', which='major', labelsize=4)

# 设置Y轴范围：从10^0到10^4.5
ax.set_ylim(1e0, 3e4)
ax.set_xlim(0, len(activation_counts_np))

plot_file_original = f"{OUTPUT_DIR}/{MODEL_NAME}_plot_original_gini_{gini:.3f}_pareto_{pareto_20:.3f}.png"
plt.savefig(plot_file_original, dpi=300, bbox_inches='tight')
plt.close()
log_print(f"Original plot saved: {plot_file_original}")

log_print("\nDone!")
log_f.close()
