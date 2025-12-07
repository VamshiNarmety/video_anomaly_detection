# Video Anomaly Detection with Attention-Gated Memory Networks

Memory-guided Normality for Anomaly Detection (MNAD) with attention gate mechanisms for improved spatial localization of anomalies in video sequences.

## Overview

This project implements video anomaly detection using memory-augmented autoencoders with optional attention gate mechanisms. The model learns normal patterns from training videos and detects anomalies by measuring reconstruction errors on test videos.

[link to the Comprehensive Report](https://github.com/VamshiNarmety/video_anomaly_detection/blob/main/video_anomaly_detection.pdf) <br>
**Key Features:**
- Memory-guided autoencoder architecture
- Attention U-Net style gating at decoder skip connections (optional)
- Multi-scale spatial attention (3 levels: 32x32, 64x64, 128x128)
- Support for UCSD Ped2 and CUHK Avenue datasets
- Comprehensive visualization tools for attention maps and reconstruction errors

**Based on:**
- Park et al., "Learning Memory-guided Normality for Anomaly Detection", CVPR 2020
- Enhanced with attention gates inspired by Oktay et al., "Attention U-Net", MIDL 2018

---

## Directory Structure

```
video_anomaly_detection/
├── data/                           # Dataset directory
│   ├── ped2/
│   │   ├── training/frames/
│   │   └── testing/frames/
│   ├── avenue/
│   │   ├── training/frames/
│   │   └── testing/frames/
│   ├── frame_labels_ped2.npy      # Ground truth labels
│   └── frame_labels_avenue.npy
├── model/
│   ├── final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1.py
│   ├── memory_final_spatial_sumonly_weight_ranking_top1.py
│   ├── attention.py               # Attention gate implementation
│   └── utils.py                   # Data loader
├── exp/                            # Saved models
│   ├── ped2/pred/
│   │   ├── baseline/
│   │   └── ped2_attn_decoder/
│   └── avenue/pred/
│       ├── baseline/
│       └── avenue_attn_decoder/
├── results/                        # Evaluation results
├── visualizations/                 # Generated visualizations
├── Train.py                        # Training script
├── Evaluate.py                     # Evaluation script
├── visualize_results.py           # Individual frame visualization
├── visualize_comparison.py        # Baseline vs attention comparison
└── README.md
```

---

## Setup Environment

```bash
# Clone or navigate to project directory
cd /path/to/video_anomaly_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python scikit-learn matplotlib scipy seaborn pandas tqdm
```

### Dataset Preparation

**UCSD Ped2:**
1. Download from [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
2. Extract frames and organize as:
```
data/ped2/
├── training/frames/
│   ├── 01/
│   │   ├── 001.jpg
│   │   └── ...
│   └── ...
└── testing/frames/
    ├── 01/
    └── ...
```

**CUHK Avenue:**
1. Download from [CUHK Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
2. Extract frames from videos:
```bash
python extract_avenue_from_videos.py
```
3. Organize similar to Ped2 structure

---

## Usage

### Training

**Baseline Model (No Attention):**

```bash
# Ped2
python Train.py \
  --gpus 0 \
  --dataset_type ped2 \
  --dataset_path ./data \
  --exp_dir baseline \
  --epochs 50 \
  --batch_size 32

# Avenue
python Train.py \
  --gpus 0 \
  --dataset_type avenue \
  --dataset_path ./data \
  --exp_dir baseline \
  --epochs 50 \
  --batch_size 32
```

**Attention Model (With Attention Gates):**

```bash
# Ped2
python Train.py \
  --gpus 0 \
  --dataset_type ped2 \
  --dataset_path ./data \
  --exp_dir ped2_attn_decoder \
  --attn_decoder gate \
  --epochs 50 \
  --batch_size 32

# Avenue
python Train.py \
  --gpus 0 \
  --dataset_type avenue \
  --dataset_path ./data \
  --exp_dir avenue_attn_decoder \
  --attn_decoder gate \
  --epochs 50 \
  --batch_size 32
```

**Training Parameters:**
- `--gpus`: GPU device ID (default: 0)
- `--epochs`: Number of training epochs (default: 60)
- `--batch_size`: Batch size (default: 4, reduce to 2 if OOM)
- `--lr`: Learning rate (default: 2e-4)
- `--attn_decoder`: Attention type - 'none' or 'gate'
- `--exp_dir`: Experiment directory name

**Output:**
- Models saved to: `./exp/{dataset}/pred/{exp_dir}/model.pth`
- Memory keys saved to: `./exp/{dataset}/pred/{exp_dir}/keys.pt`

---

### Evaluation

**Baseline Model:**

```bash
# Ped2
python Evaluate.py \
  --gpus 0 \
  --dataset_path ./data \
  --dataset_type ped2 \
  --t_length 5 \
  --model_dir ./exp/ped2/pred/baseline/model.pth \
  --m_items_dir ./exp/ped2/pred/baseline/keys.pt

# Avenue
python Evaluate.py \
  --gpus 0 \
  --dataset_path ./data \
  --dataset_type avenue \
  --t_length 5 \
  --model_dir ./exp/avenue/pred/baseline/model.pth \
  --m_items_dir ./exp/avenue/pred/baseline/keys.pt
```

**Attention Model:**

```bash
# Ped2
python Evaluate.py \
  --gpus 0 \
  --dataset_path ./data \
  --dataset_type ped2 \
  --t_length 5 \
  --attn_decoder gate \
  --model_dir ./exp/ped2/pred/ped2_attn_decoder/model.pth \
  --m_items_dir ./exp/ped2/pred/ped2_attn_decoder/keys.pt

# Avenue
python Evaluate.py \
  --gpus 0 \
  --dataset_path ./data \
  --dataset_type avenue \
  --t_length 5 \
  --attn_decoder gate \
  --model_dir ./exp/avenue/pred/avenue_attn_decoder/model.pth \
  --m_items_dir ./exp/avenue/pred/avenue_attn_decoder/keys.pt
```

**Output:**
- AUC score printed to console
- Confusion matrix: `./results/{exp_dir}/confusion_matrix.png`
- ROC curve: `./results/{exp_dir}/roc_curve.png`
- Precision, Recall, F1-Score metrics

---

### Visualization

**Individual Frame Analysis:**

```bash
# Ped2 Baseline
python visualize_results.py \
  --gpus 0 \
  --dataset_type ped2 \
  --dataset_path ./data \
  --model_dir ./exp/ped2/pred/baseline/model.pth \
  --m_items_dir ./exp/ped2/pred/baseline/keys.pt \
  --attn_decoder none \
  --num_videos 3 \
  --frames_per_video 5

# Ped2 Attention
python visualize_results.py \
  --gpus 0 \
  --dataset_type ped2 \
  --dataset_path ./data \
  --model_dir ./exp/ped2/pred/ped2_attn_decoder/model.pth \
  --m_items_dir ./exp/ped2/pred/ped2_attn_decoder/keys.pt \
  --attn_decoder gate \
  --num_videos 3 \
  --frames_per_video 5

# Avenue Baseline
python visualize_results.py \
  --gpus 0 \
  --dataset_type avenue \
  --dataset_path ./data \
  --model_dir ./exp/avenue/pred/baseline/model.pth \
  --m_items_dir ./exp/avenue/pred/baseline/keys.pt \
  --attn_decoder none \
  --num_videos 3 \
  --frames_per_video 5

# Avenue Attention
python visualize_results.py \
  --gpus 0 \
  --dataset_type avenue \
  --dataset_path ./data \
  --model_dir ./exp/avenue/pred/avenue_attn_decoder/model.pth \
  --m_items_dir ./exp/avenue/pred/avenue_attn_decoder/keys.pt \
  --attn_decoder gate \
  --num_videos 3 \
  --frames_per_video 5
```

**Output:**
- Baseline: `./visualizations/{dataset}_baseline/*.png` (2-row layout)
- Attention: `./visualizations/{dataset}_attn_decoder/*.png` (3-row layout with attention maps)

**Visualization Contents:**
- Baseline: Input frames, target, reconstruction, error heatmap, error overlay
- Attention: All baseline content plus attention maps at 3 levels and attention overlays

---

## Model Architecture

### Baseline Architecture

**Encoder:**
- 4 convolutional blocks: 64 -> 128 -> 256 -> 512 channels
- Skip connections to decoder at 3 levels (256, 128, 64)

**Memory Module:**
- 10 memory items, 512-dimensional features
- Compact loss: encourages feature diversity
- Separation loss: enforces inter-cluster separation

**Decoder:**
- Symmetric upsampling with skip connections
- Reconstructs future frame from encoded features + memory

### Attention-Gated Architecture

**Additional Components:**
- Attention gates at 3 decoder levels (Level 1: 128x128, Level 2: 64x64, Level 3: 32x32)
- Gating signal from memory-augmented features (512-dim)
- Skip features modulated by learned attention weights
- Memory read modulated by aggregated attention

**Attention Gate Operation:**
```
g = gating_signal (from memory)
x = skip_features (from encoder)
alpha = sigmoid(W_g * g + W_x * x)
output = x * alpha
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{park2020learning,
  title={Learning Memory-guided Normality for Anomaly Detection},
  author={Park, Hyunjong and Noh, Jongyoun and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14372--14381},
  year={2020}
}

@inproceedings{oktay2018attention,
  title={Attention u-net: Learning where to look for the pancreas},
  author={Oktay, Ozan and Schlemper, Jo and Folgoc, Loic Le and Lee, Matthew and Heinrich, Mattias and Misawa, Kazunari and Mori, Kensaku and McDonagh, Steven and Hammerla, Nils Y and Kainz, Bernhard and others},
  booktitle={Medical Imaging with Deep Learning},
  year={2018}
}
```
