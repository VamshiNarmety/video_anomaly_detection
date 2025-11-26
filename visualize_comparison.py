#!/usr/bin/env python
"""
Create comparison visualizations between baseline and attention models.
Generates:
1. Side-by-side ROC curves
2. Confusion matrix comparison
3. Temporal anomaly score plots
4. Performance metrics table
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
import argparse
import torch
import os
import glob
from collections import OrderedDict
import cv2
import torch.nn as nn
import torchvision.transforms as transforms

from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from utils import *


parser = argparse.ArgumentParser(description="Compare Baseline vs Attention Model")
parser.add_argument('--dataset_type', type=str, default='ped2', help='ped2 or avenue')
parser.add_argument('--dataset_path', type=str, default='./data', help='directory of data')
parser.add_argument('--baseline_model', type=str, required=True, help='baseline model path')
parser.add_argument('--baseline_keys', type=str, required=True, help='baseline keys path')
parser.add_argument('--attn_model', type=str, required=True, help='attention model path')
parser.add_argument('--attn_keys', type=str, required=True, help='attention keys path')
parser.add_argument('--gpus', type=str, default='0', help='gpu id')
parser.add_argument('--h', type=int, default=256, help='height')
parser.add_argument('--w', type=int, default=256, help='width')
parser.add_argument('--t_length', type=int, default=5, help='time length')
parser.add_argument('--method', type=str, default='pred', help='pred or recon')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

torch.backends.cudnn.enabled = True

# Setup output directory
vis_dir = Path(f'./visualizations/comparison_{args.dataset_type}')
vis_dir.mkdir(parents=True, exist_ok=True)

print(f'Saving comparison visualizations to: {vis_dir}')

# Load ground truth labels
labels = np.load(f'./data/frame_labels_{args.dataset_type}.npy')

# Load test data
test_folder = f'{args.dataset_path}/{args.dataset_type}/testing/frames'
test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor()]), 
                          resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Loading labels...')
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

# Function to evaluate model and get anomaly scores
def evaluate_model(model_path, keys_path, attn_decoder='none'):
    print(f'\nEvaluating model: {model_path}')
    
    # Load model
    try:
        from torch.serialization import safe_globals
        with safe_globals([convAE]):
            model = torch.load(model_path, weights_only=False)
    except:
        model = torch.load(model_path, weights_only=False)
    
    model.cuda()
    model.eval()
    m_items = torch.load(keys_path)
    
    loss_func_mse = nn.MSELoss(reduction='none')
    
    # Compute anomaly scores
    anomaly_score_total_list = []
    
    for video_idx, video in enumerate(sorted(videos_list)):
        video_name = video.split('/')[-1]
        anomaly_score_list = []
        
        for frame_idx in range(len(test_dataset.samples)):
            sample = test_dataset.samples[frame_idx]
            if video_name not in sample:
                continue
            
            imgs = test_dataset.__getitem__(frame_idx)
            imgs = torch.unsqueeze(imgs, 0).cuda()
            
            if args.method == 'pred':
                input_seq = imgs[:, 0:12]
                target = imgs[:, 12:]
            else:
                input_seq = imgs
                target = imgs
            
            with torch.no_grad():
                output, _, _, _, _ = model.forward(input_seq, m_items, False)
            
            mse = torch.mean(loss_func_mse((output[0]+1)/2, (target[0]+1)/2)).item()
            anomaly_score_list.append(mse)
        
        anomaly_score_total_list += anomaly_score_list
        print(f'  Video {video_name}: {len(anomaly_score_list)} frames')
    
    return np.array(anomaly_score_total_list)

# Evaluate both models
print('=' * 60)
print('Evaluating Baseline Model')
print('=' * 60)
baseline_scores = evaluate_model(args.baseline_model, args.baseline_keys, 'none')

print('\n' + '=' * 60)
print('Evaluating Attention Model')
print('=' * 60)
attn_scores = evaluate_model(args.attn_model, args.attn_keys, 'gate')

# Normalize scores
baseline_scores = (baseline_scores - baseline_scores.min()) / (baseline_scores.max() - baseline_scores.min())
attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min())

# Match lengths
min_len = min(len(baseline_scores), len(attn_scores), len(labels_list))
baseline_scores = baseline_scores[:min_len]
attn_scores = attn_scores[:min_len]
labels_list = labels_list[:min_len]

print(f'\nTotal frames evaluated: {min_len}')

# ===== 1. ROC Curve Comparison =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Baseline ROC
fpr_base, tpr_base, _ = roc_curve(labels_list, baseline_scores)
auc_base = auc(fpr_base, tpr_base)
axes[0].plot(fpr_base, tpr_base, linewidth=2, label=f'Baseline (AUC={auc_base:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('Baseline Model ROC Curve', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Attention ROC
fpr_attn, tpr_attn, _ = roc_curve(labels_list, attn_scores)
auc_attn = auc(fpr_attn, tpr_attn)
axes[1].plot(fpr_attn, tpr_attn, linewidth=2, color='orange', label=f'Attention (AUC={auc_attn:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('Attention Model ROC Curve', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(vis_dir / 'roc_comparison_separate.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: roc_comparison_separate.png')

# Combined ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, linewidth=2, label=f'Baseline (AUC={auc_base:.4f})')
plt.plot(fpr_attn, tpr_attn, linewidth=2, color='orange', label=f'Attention (AUC={auc_attn:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve Comparison - {args.dataset_type.upper()}', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(vis_dir / 'roc_comparison_combined.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: roc_comparison_combined.png')

# ===== 2. Confusion Matrix Comparison =====
# Find optimal thresholds
thresholds = np.linspace(0, 1, 100)
f1_base_list = [f1_score(labels_list, baseline_scores > t) for t in thresholds]
f1_attn_list = [f1_score(labels_list, attn_scores > t) for t in thresholds]

best_thresh_base = thresholds[np.argmax(f1_base_list)]
best_thresh_attn = thresholds[np.argmax(f1_attn_list)]

pred_base = (baseline_scores > best_thresh_base).astype(int)
pred_attn = (attn_scores > best_thresh_attn).astype(int)

cm_base = confusion_matrix(labels_list, pred_base)
cm_attn = confusion_matrix(labels_list, pred_attn)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline confusion matrix
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
axes[0].set_title(f'Baseline (F1={max(f1_base_list):.4f})', fontsize=14)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Attention confusion matrix
sns.heatmap(cm_attn, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
axes[1].set_title(f'Attention (F1={max(f1_attn_list):.4f})', fontsize=14)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(vis_dir / 'confusion_matrix_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: confusion_matrix_comparison.png')

# ===== 3. Temporal Anomaly Score Plot =====
fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

# Baseline
axes[0].plot(baseline_scores, linewidth=0.8, color='blue', label='Anomaly Score')
axes[0].fill_between(range(len(labels_list)), 0, labels_list, alpha=0.3, color='red', label='Ground Truth')
axes[0].set_ylabel('Anomaly Score', fontsize=12)
axes[0].set_title(f'Baseline Model - Temporal Anomaly Scores (AUC={auc_base:.4f})', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0, 1.1])

# Attention
axes[1].plot(attn_scores, linewidth=0.8, color='orange', label='Anomaly Score')
axes[1].fill_between(range(len(labels_list)), 0, labels_list, alpha=0.3, color='red', label='Ground Truth')
axes[1].set_xlabel('Frame Index', fontsize=12)
axes[1].set_ylabel('Anomaly Score', fontsize=12)
axes[1].set_title(f'Attention Model - Temporal Anomaly Scores (AUC={auc_attn:.4f})', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(vis_dir / 'temporal_scores_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: temporal_scores_comparison.png')

# ===== 4. Performance Metrics Table =====
prec_base = precision_score(labels_list, pred_base)
rec_base = recall_score(labels_list, pred_base)
f1_base = f1_score(labels_list, pred_base)

prec_attn = precision_score(labels_list, pred_attn)
rec_attn = recall_score(labels_list, pred_attn)
f1_attn = f1_score(labels_list, pred_attn)

metrics_data = {
    'Model': ['Baseline', 'Attention', 'Improvement'],
    'AUC': [f'{auc_base:.4f}', f'{auc_attn:.4f}', f'+{(auc_attn-auc_base):.4f}'],
    'Precision': [f'{prec_base:.4f}', f'{prec_attn:.4f}', f'+{(prec_attn-prec_base):.4f}'],
    'Recall': [f'{rec_base:.4f}', f'{rec_attn:.4f}', f'+{(rec_attn-rec_base):.4f}'],
    'F1-Score': [f'{f1_base:.4f}', f'{f1_attn:.4f}', f'+{(f1_attn-f1_base):.4f}']
}

df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.15, 0.12, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Color code
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, 3):
    for j in range(len(df.columns)):
        if i == 1:
            table[(i, j)].set_facecolor('#D9E1F2')
        else:
            table[(i, j)].set_facecolor('#FFF2CC')

plt.title(f'Performance Metrics Comparison - {args.dataset_type.upper()}', fontsize=14, pad=20)
plt.savefig(vis_dir / 'metrics_table.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: metrics_table.png')

# Save metrics as CSV
df.to_csv(vis_dir / 'metrics_comparison.csv', index=False)
print(f'Saved: metrics_comparison.csv')

# ===== 5. Score Distribution Comparison =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline distribution
normal_base = baseline_scores[labels_list == 0]
anomaly_base = baseline_scores[labels_list == 1]
axes[0].hist(normal_base, bins=50, alpha=0.6, label='Normal', color='blue')
axes[0].hist(anomaly_base, bins=50, alpha=0.6, label='Anomaly', color='red')
axes[0].axvline(best_thresh_base, color='black', linestyle='--', linewidth=2, label=f'Threshold={best_thresh_base:.3f}')
axes[0].set_xlabel('Anomaly Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Baseline - Score Distribution', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Attention distribution
normal_attn = attn_scores[labels_list == 0]
anomaly_attn = attn_scores[labels_list == 1]
axes[1].hist(normal_attn, bins=50, alpha=0.6, label='Normal', color='blue')
axes[1].hist(anomaly_attn, bins=50, alpha=0.6, label='Anomaly', color='red')
axes[1].axvline(best_thresh_attn, color='black', linestyle='--', linewidth=2, label=f'Threshold={best_thresh_attn:.3f}')
axes[1].set_xlabel('Anomaly Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Attention - Score Distribution', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(vis_dir / 'score_distribution_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: score_distribution_comparison.png')

print(f'\n{"="*60}')
print('Summary:')
print(f'{"="*60}')
print(f'Dataset: {args.dataset_type.upper()}')
print(f'Baseline AUC: {auc_base:.4f}')
print(f'Attention AUC: {auc_attn:.4f}')
print(f'AUC Improvement: +{(auc_attn-auc_base):.4f} ({(auc_attn-auc_base)/auc_base*100:.2f}%)')
print(f'\nAll visualizations saved to: {vis_dir}')
print(f'{"="*60}')
