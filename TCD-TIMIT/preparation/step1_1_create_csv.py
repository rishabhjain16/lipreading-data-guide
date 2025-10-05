#!/usr/bin/env python3
"""
TCD-TIMIT Step 1.1: Create CSV from Processed Files
====================================================

Creates CSV metadata file from already-processed TCD-TIMIT data.
Use this if step1 completed processing but failed at CSV creation.

Usage:
    python step1_1_create_csv.py \
        --root-dir /path/to/output \
        --subset lipspeakers \
        --crop-type lips \
        --detector retinaface
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser(description="Create CSV from processed TCD-TIMIT files")
parser.add_argument("--root-dir", type=str, required=True, help="Root directory with processed data")
parser.add_argument("--subset", type=str, required=True, choices=["volunteers", "lipspeakers"], help="Subset")
parser.add_argument("--crop-type", type=str, default="lips", choices=["lips", "face"], help="Crop type")
parser.add_argument("--detector", type=str, default="retinaface", help="Detector used")
args = parser.parse_args()

# Setup paths
output_size = 96 if args.crop_type == "lips" else 224
crop_suffix = f"_{args.crop_type}" if args.crop_type != "lips" else ""
size_suffix = f"_{output_size}x{output_size}" if output_size != 96 else ""

video_dir = Path(args.root_dir) / "tcd_timit" / f"tcd_timit_video{crop_suffix}{size_suffix}"
text_dir = Path(args.root_dir) / "tcd_timit" / f"tcd_timit_text{crop_suffix}{size_suffix}"
labels_dir = Path(args.root_dir) / "tcd_timit" / "labels"

labels_dir.mkdir(parents=True, exist_ok=True)

print(f"Scanning for processed files in: {video_dir}")

# Find all processed video files
video_files = sorted(video_dir.rglob("*.mp4"))
print(f"Found {len(video_files)} video files")

if len(video_files) == 0:
    print("ERROR: No video files found!")
    print(f"Check that this directory exists and has files: {video_dir}")
    exit(1)

# Create CSV data
csv_data = []

for video_path in tqdm(video_files, desc="Creating CSV"):
    # Get relative path from video_dir
    rel_path = video_path.relative_to(video_dir)
    
    # Extract info from path
    # Path format: subset/speaker/.../video_id.mp4 (variable depth)
    parts = rel_path.parts
    if len(parts) < 2:
        continue
    
    subset = parts[0]
    speaker = parts[1]
    video_id = video_path.stem  # filename without extension
    
    # Check if corresponding text file exists (same relative structure)
    text_path = text_dir / rel_path.with_suffix('.txt')
    if not text_path.exists():
        print(f"Warning: Missing text file for {video_id}")
        continue
    
    # Read transcript
    try:
        with open(text_path, 'r') as f:
            transcript = f.read().strip()
    except:
        transcript = ""
    
    # Create CSV entry
    video_rel_path = f"tcd_timit_video{crop_suffix}{size_suffix}/{rel_path}"
    
    csv_data.append([
        subset,
        speaker,
        video_id,
        video_rel_path,
        transcript,
        len(transcript.split()) if transcript else 0,
        args.detector,
        args.crop_type,
        f"{output_size}x{output_size}"
    ])

# Save CSV
if csv_data:
    csv_filename = f"tcd_timit_{args.subset}{crop_suffix}{size_suffix}_{args.detector}.csv"
    csv_path = labels_dir / csv_filename
    
    df = pd.DataFrame(csv_data, columns=[
        'subset', 'speaker', 'video_id', 'video_path', 'transcript', 'word_count',
        'detector', 'crop_type', 'resolution'
    ])
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ CSV created successfully!")
    print(f"   File: {csv_path}")
    print(f"   Samples: {len(df)}")
    print(f"   Crop: {args.crop_type} ({output_size}x{output_size})")
    print(f"   Detector: {args.detector}")
else:
    print("\n❌ No data found to create CSV!")
    exit(1)
