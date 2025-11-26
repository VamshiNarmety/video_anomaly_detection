#!/usr/bin/env python
"""
Visualization script for MNAD with attention gates.
Creates comprehensive visualizations showing:
1. Input frames vs reconstructed frames
2. Attention gate activations at each decoder level
3. Memory attention weights
4. Anomaly score heatmaps
5. Per-frame anomaly scores over time
"""

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from collections import OrderedDict
import glob
import argparse
from pathlib import Path

from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from utils import *


parser = argparse.ArgumentParser(description="MNAD Visualization")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='pred or recon')
parser.add_argument('--t_length', type=int, default=5, help='length of frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of memory')
parser.add_argument('--msize', type=int, default=10, help='number of memory items')
parser.add_argument('--dataset_type', type=str, default='ped2', help='ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./data', help='directory of data')
parser.add_argument('--model_dir', type=str, required=True, help='directory of model')
parser.add_argument('--m_items_dir', type=str, required=True, help='directory of memory items')
parser.add_argument('--attn_decoder', type=str, default='none', help='decoder attention: none or gate')
parser.add_argument('--num_videos', type=int, default=3, help='number of test videos to visualize')
parser.add_argument('--frames_per_video', type=int, default=5, help='frames to show per video')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True

# Setup output directory
exp_name = os.path.basename(os.path.dirname(args.model_dir))
# Use dataset_baseline for baseline models
if exp_name == 'baseline':
    vis_dir = Path(f'./visualizations/{args.dataset_type}_baseline')
else:
    vis_dir = Path(f'./visualizations/{exp_name}')
vis_dir.mkdir(parents=True, exist_ok=True)

print(f'Saving visualizations to: {vis_dir}')

# Load model
try:
    from torch.serialization import safe_globals
except:
    safe_globals = None

if safe_globals is not None:
    with safe_globals([convAE]):
        model = torch.load(args.model_dir, weights_only=False)
else:
    model = torch.load(args.model_dir, weights_only=False)

model.cuda()
model.eval()
m_items = torch.load(args.m_items_dir)

# Load dataset
test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"
test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor()]), 
                          resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

# Get video list
videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

print(f'Found {len(videos_list)} test videos')

# Modified forward pass to capture attention maps
def forward_with_attention(model, x, keys):
    """Forward pass that captures intermediate attention activations."""
    fea, skip1, skip2, skip3 = model.encoder(x)
    updated_fea, keys, _, _, _, _, _, compactness_loss = model.memory(fea, keys, train=False)
    
    attention_maps = {}
    
    if model.attn_decoder == 'gate':
        d = model.feature_dim
        query_feat = updated_fea[:, :d, :, :]
        mem_read = updated_fea[:, d:, :, :]
        
        # Capture attention at each level
        gated_skip3, a3 = model.att_gate3(skip3, mem_read)
        gated_skip2, a2 = model.att_gate2(skip2, mem_read)
        gated_skip1, a1 = model.att_gate1(skip1, mem_read)
        
        attention_maps['level3'] = a3.detach()
        attention_maps['level2'] = a2.detach()
        attention_maps['level1'] = a1.detach()
        attention_maps['mem_attn'] = ((a1 + a2 + a3) / 3.0).detach()
        
        alpha_mem = (a1 + a2 + a3) / 3.0
        mem_read_mod = mem_read * alpha_mem
        updated_fea_mod = torch.cat((query_feat, mem_read_mod), dim=1)
        output = model.decoder(updated_fea_mod, gated_skip1, gated_skip2, gated_skip3)
    else:
        output = model.decoder(updated_fea, skip1, skip2, skip3)
    
    return output, fea, updated_fea, attention_maps, compactness_loss


loss_func_mse = nn.MSELoss(reduction='none')

# Visualize selected videos
for vid_idx in range(min(args.num_videos, len(videos_list))):
    video_path = videos_list[vid_idx]
    video_name = video_path.split('/')[-1]
    
    print(f'\nProcessing video {video_name}...')
    
    # Select frames to visualize (evenly spaced)
    total_frames = videos[video_name]['length']
    frame_indices = np.linspace(0, total_frames - args.t_length - 1, 
                                args.frames_per_video, dtype=int)
    
    for frame_idx in frame_indices:
        # Load frame sequence
        imgs_np = []
        for i in range(args.t_length):
            frame_path = videos[video_name]['frame'][frame_idx + i]
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (args.w, args.h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 127.5 - 1.0
            imgs_np.append(img)
        
        # Convert to tensor
        imgs_tensor = torch.from_numpy(np.array(imgs_np)).permute(0, 3, 1, 2).cuda()  # [T, C, H, W]
        
        if args.method == 'pred':
            # Reshape from [T, C, H, W] to [1, T*C, H, W] for encoder
            input_frames = imgs_tensor[0:4]  # First 4 frames
            target_frame = imgs_tensor[4:5]  # 5th frame
            input_seq = input_frames.unsqueeze(0)  # [1, 4, C, H, W]
            input_seq = input_seq.reshape(1, -1, args.h, args.w)  # [1, 4*C, H, W]
            target_frame = target_frame.unsqueeze(0)  # [1, 1, C, H, W]
            target_frame = target_frame.reshape(1, -1, args.h, args.w)  # [1, C, H, W]
        else:
            input_seq = imgs_tensor.unsqueeze(0).reshape(1, -1, args.h, args.w)
            target_frame = imgs_tensor.unsqueeze(0).reshape(1, -1, args.h, args.w)
        
        # Forward pass with attention capture
        with torch.no_grad():
            output, fea, updated_fea, attention_maps, _ = forward_with_attention(model, input_seq, m_items)
        
        # Compute reconstruction error
        recon_error = torch.mean(loss_func_mse((output+1)/2, (target_frame+1)/2), dim=1)[0].cpu().numpy()
        
        # Create visualization based on model type
        if args.attn_decoder == 'gate' and attention_maps:
            # Full visualization with attention (3 rows)
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.3)
            has_attention = True
        else:
            # Simplified visualization for baseline (2 rows, no attention)
            fig = plt.figure(figsize=(18, 8))
            gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.3)
            has_attention = False
        
        # Row 1: Input frames
        for i in range(min(4, args.t_length-1)):
            ax = fig.add_subplot(gs[0, i])
            frame = ((imgs_np[i] + 1) / 2 * 255).astype(np.uint8)
            ax.imshow(frame)
            ax.set_title(f'Input Frame {i+1}')
            ax.axis('off')
        
        # Target frame
        ax = fig.add_subplot(gs[0, 4])
        target = ((imgs_np[-1] + 1) / 2 * 255).astype(np.uint8)
        ax.imshow(target)
        ax.set_title('Target Frame')
        ax.axis('off')
        
        # Reconstructed frame
        ax = fig.add_subplot(gs[0, 5])
        recon = ((output[0, :3].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        ax.imshow(recon)
        ax.set_title('Reconstructed')
        ax.axis('off')
        
        # Row 2: Reconstruction error heatmap
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(recon_error, cmap='jet')
        ax.set_title('Reconstruction Error')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Overlay error on target
        ax = fig.add_subplot(gs[1, 1])
        error_norm = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-8)
        error_colored = plt.cm.jet(error_norm)[:, :, :3]
        overlay = (target.astype(float) * 0.5 + error_colored * 255 * 0.5).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title('Error Overlay')
        ax.axis('off')
        
        # Attention visualizations (only if attention model)
        if has_attention and attention_maps:
            # Level 3 attention
            ax = fig.add_subplot(gs[1, 2])
            attn3 = attention_maps['level3'][0, 0].cpu().numpy()
            im = ax.imshow(attn3, cmap='viridis')
            ax.set_title('Attention Level 3')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Level 2 attention
            ax = fig.add_subplot(gs[1, 3])
            attn2 = attention_maps['level2'][0, 0].cpu().numpy()
            im = ax.imshow(attn2, cmap='viridis')
            ax.set_title('Attention Level 2')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Level 1 attention
            ax = fig.add_subplot(gs[1, 4])
            attn1 = attention_maps['level1'][0, 0].cpu().numpy()
            im = ax.imshow(attn1, cmap='viridis')
            ax.set_title('Attention Level 1')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Memory attention
            ax = fig.add_subplot(gs[1, 5])
            mem_attn = attention_maps['mem_attn'][0, 0].cpu().numpy()
            im = ax.imshow(mem_attn, cmap='viridis')
            ax.set_title('Memory Attention')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Row 3: Attention overlays on input
            for i, (level_name, attn_map) in enumerate([
                ('Level 3', attention_maps['level3']),
                ('Level 2', attention_maps['level2']),
                ('Level 1', attention_maps['level1'])
            ]):
                ax = fig.add_subplot(gs[2, i*2])
                attn_resized = F.interpolate(attn_map, size=(args.h, args.w), 
                                            mode='bilinear', align_corners=False)
                attn_np = attn_resized[0, 0].cpu().numpy()
                attn_colored = plt.cm.viridis(attn_np)[:, :, :3]
                attn_overlay = (target.astype(float) * 0.4 + attn_colored * 255 * 0.6).astype(np.uint8)
                ax.imshow(attn_overlay)
                ax.set_title(f'{level_name} Overlay')
                ax.axis('off')
        
        plt.suptitle(f'Video {video_name} - Frame {frame_idx}', fontsize=16)
        
        # Save figure
        out_path = vis_dir / f'{video_name}_frame_{frame_idx:04d}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  Saved: {out_path.name}')

print(f'\nVisualization complete! Saved to: {vis_dir}')
print(f'Generated {args.num_videos * args.frames_per_video} visualization images')
