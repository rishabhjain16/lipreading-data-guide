#!/usr/bin/env python3
"""
Lombard GRID Step 3: Metadata Preparation

This script counts frames and creates manifest files for Lombard GRID dataset.
Creates a single test manifest with all data.

Usage:
    python step3_metadata_prep.py \
        --lombardgrid-data-dir /path/to/output/lombardgrid_video \
        --metadata-dir /path/to/output/metadata \
        --vocab-size 100
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile

# Import vocabulary generation from LRS3
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "LRS3" / "preparation"))
from gen_subword import gen_vocab

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
parser.add_argument(
    "--vocab-size",
    type=int,
    default=100,
    help="Vocabulary size for sentencepiece (default: 100)",
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
    
    # Generate vocabulary
    print(f"Generating sentencepiece vocabulary for {view_name}...")
    vocab_size = args.vocab_size
    
    vocab_dir = view_dir / f"spm{vocab_size}"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    spm_filename_prefix = f"spm_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w", delete=False) as f:
        for label in valid_labels:
            f.write(label.lower() + "\n")
        temp_file = f.name
    
    gen_vocab(Path(temp_file), vocab_dir / spm_filename_prefix, 'unigram', vocab_size)
    os.unlink(temp_file)
    
    vocab_path = vocab_dir / f"{spm_filename_prefix}.txt"
    print(f"✅ Created vocabulary: {vocab_path}")
    
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
            
            # Format: id, video_path, audio_path, num_audio_frames, num_video_frames
            f.write(f"{fid}\t{video_path}\t{audio_path}\t{num_audio}\t{num_video}\n")
    
    with open(wrd_path, 'w') as f:
        for label in valid_labels:
            f.write(label + '\n')
    
    print(f"✅ Created test manifest: {manifest_path} ({len(valid_fids)} entries)")
    
    # Create dictionary file
    dict_path = view_dir / "dict.wrd.txt"
    
    vocab_file = vocab_dir / f"{spm_filename_prefix}.vocab"
    if vocab_file.exists():
        with open(vocab_file, 'r') as f:
            vocab_lines = f.readlines()
        
        with open(dict_path, 'w') as f:
            for line in vocab_lines:
                token = line.split('\t')[0]
                if token not in ['<unk>', '<s>', '</s>']:
                    f.write(f"{token} 1\n")
        
        print(f"✅ Created dictionary: {dict_path}")
    
    return {
        'nframes_audio': nframes_audio_path,
        'nframes_video': nframes_video_path,
        'manifest': manifest_path,
        'wrd': wrd_path,
        'dict': dict_path,
        'vocab_dir': vocab_dir
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
