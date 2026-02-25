#!/usr/bin/env python3
"""
Check total duration of processed Candor dataset

Usage:
    python check_dataset_duration.py --data-dir /path/to/candor_output/candor_video
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def get_video_duration(video_path):
    """Get duration of a video file in seconds"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps > 0 and frame_count > 0:
            return frame_count / fps
        return 0
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Check total duration of Candor dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing processed video files')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return 1
    
    print(f"📁 Scanning directory: {data_dir}")
    
    # Find all video files
    video_files = list(data_dir.rglob('*.mp4'))
    
    if not video_files:
        print("❌ No video files found")
        return 1
    
    print(f"🎬 Found {len(video_files)} video files")
    print("⏱️  Calculating total duration...\n")
    
    total_duration = 0
    valid_files = 0
    error_files = 0
    min_duration = float('inf')
    max_duration = 0
    durations = []
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        duration = get_video_duration(video_path)
        if duration > 0:
            total_duration += duration
            valid_files += 1
            durations.append(duration)
            min_duration = min(min_duration, duration)
            max_duration = max(max_duration, duration)
        else:
            error_files += 1
    
    # Convert to hours, minutes, seconds
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"\n{'='*50}")
    print(f"📊 Dataset Statistics")
    print(f"{'='*50}")
    print(f"✅ Valid files: {valid_files:,}")
    print(f"❌ Error files: {error_files:,}")
    print(f"⏱️  Total duration: {hours}h {minutes}m {seconds}s")
    print(f"⏱️  Total duration: {total_duration:.2f} seconds")
    print(f"⏱️  Total duration: {total_duration/3600:.2f} hours")
    print(f"📈 Average clip length: {total_duration/valid_files:.2f} seconds")
    print(f"📉 Min clip length: {min_duration:.2f} seconds")
    print(f"📈 Max clip length: {max_duration:.2f} seconds")
    
    # Calculate median
    if durations:
        durations.sort()
        median_duration = durations[len(durations)//2]
        print(f"📊 Median clip length: {median_duration:.2f} seconds")
    
    print(f"{'='*50}")
    
    return 0


if __name__ == '__main__':
    exit(main())
