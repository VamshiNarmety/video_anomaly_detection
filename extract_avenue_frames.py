#!/usr/bin/env python
"""Extract Avenue testing frames from avenue.mat file."""
import scipy.io
import numpy as np
import cv2
import os
from pathlib import Path

# Load mat file
mat_path = './data/avenue/avenue.mat'
print(f'Loading {mat_path}...')
data = scipy.io.loadmat(mat_path)

# Inspect keys
print('Keys in mat file:', list(data.keys()))

# Extract testing frames (adjust key name based on actual structure)
# Common keys: 'testing_videos', 'test_data', 'X_test', etc.
# Try to find the right key
test_key = None
for key in data.keys():
    if not key.startswith('__'):
        print(f'Key: {key}, Type: {type(data[key])}, Shape: {data[key].shape if hasattr(data[key], "shape") else "N/A"}')
        if 'test' in key.lower():
            test_key = key

if test_key:
    print(f'\nUsing key: {test_key}')
    test_data = data[test_key]
    
    # test_data is typically (num_videos, 1) array of arrays
    # or (num_frames, height, width, channels) for all videos concatenated
    
    # Create output directory
    out_dir = Path('./data/avenue/testing/frames')
    
    # If structure is per-video arrays
    if test_data.shape[0] == 21 or test_data.shape[1] == 21:  # 21 test videos for Avenue
        for vid_idx in range(min(21, test_data.shape[0] if test_data.shape[0] <= 21 else test_data.shape[1])):
            video_dir = out_dir / f'{vid_idx+1:02d}'
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract video frames
            if test_data.shape[0] <= 21:
                vid_frames = test_data[vid_idx, 0] if test_data.ndim > 1 else test_data[vid_idx]
            else:
                vid_frames = test_data[0, vid_idx] if test_data.ndim > 1 else test_data[vid_idx]
            
            # vid_frames could be (num_frames, h, w, c) or (h, w, c, num_frames)
            if vid_frames.ndim == 4:
                num_frames = vid_frames.shape[0] if vid_frames.shape[0] < 10000 else vid_frames.shape[-1]
                for frame_idx in range(num_frames):
                    if vid_frames.shape[0] < 10000:
                        frame = vid_frames[frame_idx]
                    else:
                        frame = vid_frames[:, :, :, frame_idx]
                    
                    # Normalize if needed
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for cv2
                    if frame.shape[-1] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    out_path = video_dir / f'{frame_idx+1:03d}.jpg'
                    cv2.imwrite(str(out_path), frame)
            
            print(f'Extracted video {vid_idx+1:02d}: {len(list(video_dir.glob("*.jpg")))} frames')
    else:
        print(f'Unexpected data structure. Shape: {test_data.shape}')
        print('Please inspect the mat file manually with scipy.io.loadmat')
else:
    print('\nNo testing key found. Available keys:', [k for k in data.keys() if not k.startswith('__')])
    print('\nTry inspecting the mat file manually to find the correct key.')

print('\nDone!')
