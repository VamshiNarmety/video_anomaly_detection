import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from utils import *
import random
import glob

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.001, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--attn_decoder', type=str, default='none', help='decoder attention: none or gate')

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

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
try:
    # Try the default load (works for state_dict or older PyTorch setups)
    model = torch.load(args.model_dir)
except TypeError:
    # Newer PyTorch versions may default to weights_only=True; allow legacy unpickling
    model = torch.load(args.model_dir, weights_only=False)
except Exception:
    # Fallback: explicitly allowlist the local convAE class for safe unpickling
    try:
        from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE
        # Try to use the safe_globals context manager if available
        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None

        try:
            from torch.serialization import add_safe_globals
        except Exception:
            add_safe_globals = None

        if safe_globals is not None:
            with safe_globals([convAE]):
                model = torch.load(args.model_dir, weights_only=False)
        elif add_safe_globals is not None:
            # add_safe_globals registers allowed globals (not a context manager)
            add_safe_globals([convAE])
            model = torch.load(args.model_dir, weights_only=False)
        else:
            # Last resort: try loading directly (may raise security-related errors)
            model = torch.load(args.model_dir, weights_only=False)
    except Exception as e:
        raise

model.cuda()
m_items = torch.load(args.m_items_dir)
labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

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

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    else:
        labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

for k,(imgs) in enumerate(test_batch):
    
    if args.method == 'pred':
        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()
    
    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])
    
    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

    if  point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                     anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')
 
# Prepare confusion matrix and save figure
try:
    # labels_list: original repo stores 1 for normal, 0 for anomaly -> invert to get anomaly=1
    y_true = (1 - labels_list).astype(int)
    y_scores = anomaly_score_total_list

    # find best threshold by F1 on a grid
    thr_vals = np.linspace(np.min(y_scores), np.max(y_scores), 1000)
    best_thr = thr_vals[0]
    best_f1 = -1.0
    for thr in thr_vals:
        y_pred = (y_scores > thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # final predictions
    y_pred_best = (y_scores > best_thr).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred_best, labels=[0,1])
    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]

    prec = precision_score(y_true, y_pred_best, zero_division=0)
    rec = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = f1_score(y_true, y_pred_best, zero_division=0)

    print('Confusion matrix (rows=true [Normal,Anomaly], cols=pred [Normal,Anomaly]):')
    print(cm)
    print(f'Best decision threshold (by F1): {best_thr:.6f}')
    print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')

    # save figure
    import matplotlib.pyplot as plt

    # derive experiment name from model_dir
    if args.model_dir:
        exp_name = os.path.basename(os.path.dirname(args.model_dir))
    else:
        exp_name = f"{args.dataset_type}_{args.method}"

    results_dir = os.path.join('./results', exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['Normal', 'Anomaly']
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label', title=f'Confusion Matrix: {exp_name}')

    # annotate counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'confusion_matrix.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Confusion matrix image saved to: {out_path}')

    # Save ROC curve (AUC)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'Receiver Operating Characteristic: {exp_name}')
        ax2.legend(loc="lower right")
        roc_path = os.path.join(results_dir, 'roc_curve.png')
        fig2.savefig(roc_path)
        plt.close(fig2)
        print(f'ROC curve image saved to: {roc_path}')
    except Exception as e:
        print('Could not compute/save ROC curve:', e)
except Exception as e:
    print('Could not compute/save confusion matrix:', e)
