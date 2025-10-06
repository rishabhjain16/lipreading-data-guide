#!/usr/bin/env python3
"""
GRID Dataset - Step 2: Generate File Lists
==========================================

Generate file.list and label.list from processed GRID videos.
Can generate for all speakers combined or individual speakers.

Usage:
    # All speakers combined
    python step2_generate_file_lists.py \
        --grid-data-dir /path/to/output/grid_video

    # Individual speaker
    python step2_generate_file_lists.py \
        --grid-data-dir /path/to/output/grid_video \
        --speaker s1
"""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate file lists for GRID dataset")
parser.add_argument(
    "--grid-data-dir",
    type=str,
    required=True,
    help="Path to processed GRID video directory",
)
parser.add_argument(
    "--speaker",
    type=str,
    default=None,
    help="Process specific speaker (s1, s2, ..., s34) or omit for all speakers",
)
args = parser.parse_args()

data_dir = Path(args.grid_data_dir)

# Determine which speakers to process
if args.speaker:
    speakers = [args.speaker]
    output_suffix = f"_{args.speaker}"
else:
    # Find all speaker directories
    speakers = sorted([d.name for d in data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('s') and d.name[1:].isdigit()])
    output_suffix = ""

# Collect all video files
video_files = []
for speaker in speakers:
    speaker_dir = data_dir / speaker
    if speaker_dir.exists():
        videos = sorted(speaker_dir.glob("*.mp4"))
        video_files.extend(videos)
        print(f"Found {len(videos)} videos for speaker {speaker}")

if not video_files:
    print(f"❌ Error: No video files found in {data_dir}")
    exit(1)

print(f"\nTotal videos: {len(video_files)}")

# Determine text directory (handle both crop types)
if "face_224x224" in str(data_dir):
    text_dir = data_dir.parent / "grid_text_face_224x224"
else:
    text_dir = data_dir.parent / "grid_text"

# Generate file.list and label.list
file_list = []
label_list = []

for video_path in video_files:
    # Get relative path from data_dir
    rel_path = video_path.relative_to(data_dir)
    
    # Construct text file path
    text_path = text_dir / rel_path.with_suffix('.txt')
    
    if not text_path.exists():
        print(f"Warning: Text file not found for {video_path.name}")
        continue
    
    # Read transcript
    with open(text_path, 'r') as f:
        transcript = f.read().strip()
    
    # Add to lists
    file_list.append(str(rel_path))
    label_list.append(transcript)

# Save file.list
file_list_path = data_dir / f"file{output_suffix}.list"
with open(file_list_path, 'w') as f:
    f.write('\n'.join(file_list))

# Save label.list
label_list_path = data_dir / f"label{output_suffix}.list"
with open(label_list_path, 'w') as f:
    f.write('\n'.join(label_list))

print(f"\n✅ Generated file lists:")
print(f"   {file_list_path} ({len(file_list)} entries)")
print(f"   {label_list_path} ({len(label_list)} entries)")
