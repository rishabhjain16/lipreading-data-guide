#!/usr/bin/env python3
"""
Lombard GRID Step 3: Metadata Preparation

This script counts frames and creates manifest files for Lombard GRID dataset.
Creates a single test manifest with all data.

Usage:
    python step3_metadata_prep.py \
        --lombardgrid-data-dir /path/to/output/lombardgrid_video \
    --metadata-dir /path/to/output/metadata
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile

# Shared SPM tokenizer
from transforms import TextTransform

parser = argparse.ArgumentParser(description="Generate metadata for Lombard GRID dataset")
parser.add_argument(
    "--lombardgrid-data-dir",
    type=str,
    required=True,
    help="Path to processed Lombard GRID video directory",
)
parser.add_argument(
    "--metadata-dir",
    type=str,
    required=True,
    help="Output directory for metadata files",
)
args = parser.parse_args()

data_dir = Path(args.lombardgrid_data_dir)
metadata_dir = Path(args.metadata_dir)
metadata_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories for front, side, and combined
front_metadata_dir = metadata_dir / "front"
side_metadata_dir = metadata_dir / "side"
combined_metadata_dir = metadata_dir / "combined"

front_metadata_dir.mkdir(parents=True, exist_ok=True)
side_metadata_dir.mkdir(parents=True, exist_ok=True)
combined_metadata_dir.mkdir(parents=True, exist_ok=True)

# Load file and label lists
file_list_path = data_dir / "file.list"
label_list_path = data_dir / "label.list"

if not file_list_path.exists() or not label_list_path.exists():
    print(f"❌ Error: File lists not found. Run step2 first.")
    print(f"   Expected: {file_list_path}")
    print(f"   Expected: {label_list_path}")
    exit(1)

# Read file and label lists
with open(file_list_path, 'r') as f:
    fids = [line.strip().replace('.mp4', '') for line in f.readlines()]

with open(label_list_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Loaded {len(fids)} files")

# Separate files by view (front/side)
front_fids = []
front_labels = []
side_fids = []
side_labels = []

for fid, label in zip(fids, labels):
    if fid.startswith('front/'):
        front_fids.append(fid)
        front_labels.append(label)
    elif fid.startswith('side/'):
        side_fids.append(fid)
        side_labels.append(label)

print(f"Front view: {len(front_fids)} files")
print(f"Side view: {len(side_fids)} files")

# Initialize SPM tokenizer once (shared root model)
print("\nInitializing SentencePiece tokenizer...")
text_transform = TextTransform()
print(f"✅ SPM model loaded with {len(text_transform.token_list)} tokens")

# Count frames for each view
def count_frames_for_view(fids, labels, view_name):
    print(f"\nCounting frames for {view_name} view...")
    audio_num_frames = []
    video_num_frames = []
    valid_fids = []
    valid_labels = []
    
    for fid, label in tqdm(zip(fids, labels), total=len(fids), desc=f"Counting {view_name} frames"):
        wav_fn = data_dir / f"{fid}.wav"
        video_fn = data_dir / f"{fid}.mp4"
        
        if not wav_fn.exists() or not video_fn.exists():
            print(f"Warning: Missing files for {fid}")
            continue
        
        try:
            num_frames_audio = len(wavfile.read(str(wav_fn))[1])
            cap = cv2.VideoCapture(str(video_fn))
            num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if num_frames_audio > 0 and num_frames_video > 0:
                audio_num_frames.append(num_frames_audio)
                video_num_frames.append(num_frames_video)
                valid_fids.append(fid)
                valid_labels.append(label)
        except Exception as e:
            print(f"Warning: Error processing {fid}: {str(e)}")
            continue
    
    print(f"Successfully counted frames for {len(valid_fids)} {view_name} files")
    return audio_num_frames, video_num_frames, valid_fids, valid_labels

# Process front view
front_audio_frames, front_video_frames, front_valid_fids, front_valid_labels = count_frames_for_view(
    front_fids, front_labels, "front"
)

# Process side view
side_audio_frames, side_video_frames, side_valid_fids, side_valid_labels = count_frames_for_view(
    side_fids, side_labels, "side"
)

# Combined data
combined_audio_frames = front_audio_frames + side_audio_frames
combined_video_frames = front_video_frames + side_video_frames
combined_valid_fids = front_valid_fids + side_valid_fids
combined_valid_labels = front_valid_labels + side_valid_labels

print(f"\nCombined: {len(combined_valid_fids)} files")

# Function to create metadata for a view
def create_metadata_for_view(view_name, view_dir, audio_frames, video_frames, valid_fids, valid_labels):
    print(f"\n{'='*60}")
    print(f"Creating metadata for {view_name} view...")
    print(f"{'='*60}")
    
    # Create nframes files
    nframes_audio_path = data_dir / f"nframes_{view_name}.audio"
    nframes_video_path = data_dir / f"nframes_{view_name}.video"
    
    with open(nframes_audio_path, 'w') as f:
        f.write('\n'.join([str(x) for x in audio_frames]))
    
    with open(nframes_video_path, 'w') as f:
        f.write('\n'.join([str(x) for x in video_frames]))
    
    print(f"✅ Created: {nframes_audio_path}")
    print(f"✅ Created: {nframes_video_path}")
    
    # Create test manifest
    manifest_path = view_dir / "test.tsv"
    wrd_path = view_dir / "test.wrd"
    
    with open(manifest_path, 'w') as f:
        # Header
        f.write("/\n")
        
        # Write all entries
        for idx in range(len(valid_fids)):
            fid = valid_fids[idx]
            video_path = data_dir / f"{fid}.mp4"
            audio_path = data_dir / f"{fid}.wav"
            num_audio = audio_frames[idx]
            num_video = video_frames[idx]
            
            # Format: id, video_path, audio_path, num_video_frames, num_audio_frames
            f.write(f"{fid}\t{video_path}\t{audio_path}\t{num_video}\t{num_audio}\n")
    
    with open(wrd_path, 'w') as f:
        for label in valid_labels:
            f.write(label + '\n')
    
    print(f"✅ Created test manifest: {manifest_path} ({len(valid_fids)} entries)")

    # Create dictionary file from shared SPM model
    dict_path = view_dir / "dict.wrd.txt"
    with open(dict_path, 'w') as f:
        for idx, token in enumerate(text_transform.token_list):
            if token not in ['<blank>', '<eos>', '<unk>']:
                f.write(f"{token} {idx}\n")
    print(f"✅ Created dictionary: {dict_path}")

    # Create SPM-tokenized labels file
    tokens_path = view_dir / "tokens.txt"
    with open(tokens_path, 'w') as f:
        for label in valid_labels:
            token_ids = text_transform.tokenize(label)
            token_str = " ".join(str(t.item()) for t in token_ids)
            f.write(f"{token_str}\n")
    print(f"✅ Created tokenized labels: {tokens_path}")

    # Create simple label.csv for inference pipelines
    # Format (no header): dataset,video_path,token_ids
    label_csv_path = view_dir / "label.csv"
    dataset_name = "lombardgrid"
    with open(label_csv_path, 'w') as f:
        for fid, label in zip(valid_fids, valid_labels):
            video_abs = str((data_dir / f"{fid}.mp4").resolve())
            token_ids = text_transform.tokenize(label)
            token_str = " ".join(str(t.item()) for t in token_ids)
            f.write(f"{dataset_name},{video_abs},{token_str}\n")
    print(f"✅ Created label CSV: {label_csv_path}")

    return {
        'nframes_audio': nframes_audio_path,
        'nframes_video': nframes_video_path,
        'manifest': manifest_path,
        'wrd': wrd_path,
        'dict': dict_path,
        'tokens': tokens_path,
        'label_csv': label_csv_path,
    }

# Create metadata for front view
front_files = create_metadata_for_view(
    "front", front_metadata_dir, 
    front_audio_frames, front_video_frames, 
    front_valid_fids, front_valid_labels
)

# Create metadata for side view
side_files = create_metadata_for_view(
    "side", side_metadata_dir,
    side_audio_frames, side_video_frames,
    side_valid_fids, side_valid_labels
)

# Create metadata for combined view
combined_files = create_metadata_for_view(
    "combined", combined_metadata_dir,
    combined_audio_frames, combined_video_frames,
    combined_valid_fids, combined_valid_labels
)

print("\n" + "="*60)
print("🎉 Metadata preparation complete!")
print("="*60)
print(f"\nFront view metadata ({len(front_valid_fids)} files):")
print(f"  - {front_metadata_dir}/")
print(f"\nSide view metadata ({len(side_valid_fids)} files):")
print(f"  - {side_metadata_dir}/")
print(f"\nCombined metadata ({len(combined_valid_fids)} files):")
print(f"  - {combined_metadata_dir}/")
