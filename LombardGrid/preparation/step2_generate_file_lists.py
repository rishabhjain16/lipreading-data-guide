#!/usr/bin/env python3
"""
Lombard GRID Dataset - Step 2: Generate File Lists
==================================================

Generate file.list and label.list from processed Lombard GRID videos.

Usage:
    # All speakers
    python step2_generate_file_lists.py \
        --lombardgrid-data-dir /path/to/output/lombardgrid_video

    # Individual speaker
    python step2_generate_file_lists.py \
        --lombardgrid-data-dir /path/to/output/lombardgrid_video \
        --speaker s2
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate file lists for Lombard GRID dataset")
parser.add_argument(
    "--lombardgrid-data-dir",
    type=str,
    required=True,
    help="Path to processed Lombard GRID video directory",
)
parser.add_argument(
    "--speaker",
    type=str,
    default=None,
    help="Process specific speaker or omit for all speakers",
)
args = parser.parse_args()

data_dir = Path(args.lombardgrid_data_dir)

# Check if we have front/side structure or direct speaker structure
has_view_dirs = (data_dir / "front").exists() or (data_dir / "side").exists()

if has_view_dirs:
    # Process front and side views
    view_dirs = [d for d in ["front", "side"] if (data_dir / d).exists()]
    print(f"Found view directories: {view_dirs}")
    
    # Determine which speakers to process
    if args.speaker:
        speakers = [args.speaker]
        output_suffix = f"_{args.speaker}"
    else:
        # Get speakers from first available view
        first_view = data_dir / view_dirs[0]
        speakers = sorted([d.name for d in first_view.iterdir() 
                          if d.is_dir() and d.name.startswith('s') and d.name[1:].isdigit()])
        output_suffix = ""
    
    # Collect all video files
    video_files = []
    for view in view_dirs:
        for speaker in speakers:
            speaker_dir = data_dir / view / speaker
            if speaker_dir.exists():
                videos = sorted(speaker_dir.glob("*.mp4"))
                video_files.extend(videos)
                print(f"Found {len(videos)} videos for {view}/{speaker}")
else:
    # Original structure without view directories
    if args.speaker:
        speakers = [args.speaker]
        output_suffix = f"_{args.speaker}"
    else:
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

# Determine text directory
if "face_224x224" in str(data_dir):
    text_dir = data_dir.parent / "lombardgrid_text_face_224x224"
else:
    text_dir = data_dir.parent / "lombardgrid_text"

# Generate file.list and label.list
file_list = []
label_list = []

for video_path in video_files:
    rel_path = video_path.relative_to(data_dir)
    text_path = text_dir / rel_path.with_suffix('.txt')
    
    if not text_path.exists():
        print(f"Warning: Text file not found for {video_path.name}, skipping...")
        continue
    
    with open(text_path, 'r') as f:
        transcript = f.read().strip()
    
    file_list.append(str(rel_path))
    label_list.append(transcript)

# Save lists
file_list_path = data_dir / f"file{output_suffix}.list"
label_list_path = data_dir / f"label{output_suffix}.list"

with open(file_list_path, 'w') as f:
    f.write('\n'.join(file_list))

with open(label_list_path, 'w') as f:
    f.write('\n'.join(label_list))

print(f"\n✅ Generated file lists:")
print(f"   {file_list_path} ({len(file_list)} entries)")
print(f"   {label_list_path} ({len(label_list)} entries)")
