#!/usr/bin/env python
"""Extract Avenue testing frames from .avi videos."""
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Source and destination
src_dir = Path('/mnt/home2/home/vamshi_n/projects/course_project/Avenue Dataset/testing_videos')
dst_dir = Path('./data/avenue/testing/frames')

# Get all .avi files
video_files = sorted(src_dir.glob('*.avi'))
print(f'Found {len(video_files)} testing videos')

dst_dir.mkdir(parents=True, exist_ok=True)

for video_file in tqdm(video_files, desc='Extracting videos'):
    # Get video number (e.g., 01.avi -> 01)
    video_num = video_file.stem
    
    # Create output directory
    out_dir = dst_dir / video_num
    out_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_file))
    frame_idx = 1  # Start from 1, not 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        out_path = out_dir / f'{frame_idx:03d}.jpg'
        cv2.imwrite(str(out_path), frame)
        frame_idx += 1
    
    cap.release()
    print(f'Video {video_num}: extracted {frame_idx} frames')

print('\nDone! Frames extracted to:', dst_dir)
