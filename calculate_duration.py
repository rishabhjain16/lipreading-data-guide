#!/usr/bin/env python3
"""
Calculate total duration of RoomReader processed videos
"""

import cv2
from pathlib import Path
from tqdm import tqdm
import sys

def calculate_duration(data_dir):
    """Calculate total duration of all videos in directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: Directory not found: {data_path}")
        return
    
    # Find all mp4 files
    video_files = list(data_path.rglob("*.mp4"))
    
    if not video_files:
        print(f"❌ No video files found in {data_path}")
        return
    
    print(f"Found {len(video_files)} video files")
    print("Calculating total duration...\n")
    
    total_frames = 0
    total_duration = 0
    valid_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if frame_count > 0 and fps > 0:
                duration = frame_count / fps
                total_frames += frame_count
                total_duration += duration
                valid_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            continue
    
    # Calculate statistics
    hours = total_duration / 3600
    minutes = (total_duration % 3600) / 60
    seconds = total_duration % 60
    
    print(f"\n{'='*60}")
    print(f"RoomReader Duration Statistics")
    print(f"{'='*60}")
    print(f"Total videos processed: {valid_count:,}")
    print(f"Failed videos: {error_count:,}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total duration: {hours:.2f} hours ({int(hours)}h {int(minutes)}m {int(seconds)}s)")
    print(f"Average duration per video: {total_duration/valid_count:.2f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_duration.py <path_to_roomreader_video_dir>")
        print("\nExample:")
        print("  python calculate_duration.py /media/rishabhjain/SSD/Data/RoomReader_lips/roomreader_video/conversational")
        print("  python calculate_duration.py /media/rishabhjain/SSD/Data/RoomReader_lips/roomreader_video/individual")
        sys.exit(1)
    
    calculate_duration(sys.argv[1])
