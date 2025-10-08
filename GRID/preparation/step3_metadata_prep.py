#!/usr/bin/env python3
"""
GRID Step 3: Metadata Preparation

This script counts frames and creates manifest files for GRID dataset.
Supports speaker-based train/val/test splits.

Usage:
    # All speakers with speaker-based splits
    python step3_metadata_prep.py \
        --grid-data-dir /path/to/output/grid_video \
        --metadata-dir /path/to/output/metadata \
        --split-ratios 0.7,0.15,0.15 \
        --vocab-size 100

    # Individual speaker
    python step3_metadata_prep.py \
        --grid-data-dir /path/to/output/grid_video \
        --metadata-dir /path/to/output/metadata_s1 \
        --speaker s1 \
        --vocab-size 100
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile
import random

# Import vocabulary generation from LRS3
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "LRS3" / "preparation"))
from gen_subword import gen_vocab

parser = argparse.ArgumentParser(description="Generate metadata for GRID dataset")
parser.add_argument(
    "--grid-data-dir",
    type=str,
    required=True,
    help="Path to processed GRID video directory",
)
parser.add_argument(
    "--metadata-dir",
    type=str,
    required=True,
    help="Output directory for metadata files",
)
parser.add_argument(
    "--speaker",
    type=str,
    default=None,
    help="Process specific speaker or omit for all speakers",
)
parser.add_argument(
    "--split-ratios",
    type=str,
    default="0.7,0.15,0.15",
    help="Train/val/test split ratios (default: 0.7,0.15,0.15)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=100,
    help="Vocabulary size for sentencepiece (default: 100)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible splits (default: 42)",
)
args = parser.parse_args()

data_dir = Path(args.grid_data_dir)
metadata_dir = Path(args.metadata_dir)
metadata_dir.mkdir(parents=True, exist_ok=True)

# Parse split ratios
split_ratios = [float(x) for x in args.split_ratios.split(',')]
assert len(split_ratios) == 3, "Must provide 3 split ratios (train,val,test)"
assert abs(sum(split_ratios) - 1.0) < 0.01, "Split ratios must sum to 1.0"

# Determine file list suffix
output_suffix = f"_{args.speaker}" if args.speaker else ""

# Load file and label lists
file_list_path = data_dir / f"file{output_suffix}.list"
label_list_path = data_dir / f"label{output_suffix}.list"

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

# Count frames
print("\nCounting frames in audio and video files...")
audio_num_frames = []
video_num_frames = []
valid_fids = []
valid_labels = []

for fid, label in tqdm(zip(fids, labels), total=len(fids), desc="Counting frames"):
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

print(f"Successfully counted frames for {len(valid_fids)} files")

# Create nframes files
nframes_audio_path = data_dir / f"nframes{output_suffix}.audio"
nframes_video_path = data_dir / f"nframes{output_suffix}.video"

with open(nframes_audio_path, 'w') as f:
    f.write('\n'.join([str(x) for x in audio_num_frames]))

with open(nframes_video_path, 'w') as f:
    f.write('\n'.join([str(x) for x in video_num_frames]))

print(f"\n✅ Created: {nframes_audio_path}")
print(f"✅ Created: {nframes_video_path}")

# Generate vocabulary
print("\nGenerating sentencepiece vocabulary...")
vocab_size = args.vocab_size

vocab_dir = metadata_dir / f"spm{vocab_size}"
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

# Create train/val/test splits
if args.speaker:
    # Single speaker: random split
    print(f"\nCreating random splits for speaker {args.speaker}...")
    random.seed(args.seed)
    indices = list(range(len(valid_fids)))
    random.shuffle(indices)
    
    n_train = int(len(indices) * split_ratios[0])
    n_val = int(len(indices) * split_ratios[1])
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    splits = {
        'train': train_indices,
        'valid': val_indices,
        'test': test_indices,
    }
else:
    # Multiple speakers: speaker-based split
    print("\nCreating speaker-based splits...")
    
    # Group files by speaker
    speaker_files = {}
    for idx, fid in enumerate(valid_fids):
        speaker = fid.split('/')[0]
        if speaker not in speaker_files:
            speaker_files[speaker] = []
        speaker_files[speaker].append(idx)
    
    speakers = sorted(speaker_files.keys())
    random.seed(args.seed)
    random.shuffle(speakers)
    
    n_train = int(len(speakers) * split_ratios[0])
    n_val = int(len(speakers) * split_ratios[1])
    
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]
    
    train_indices = [idx for s in train_speakers for idx in speaker_files[s]]
    val_indices = [idx for s in val_speakers for idx in speaker_files[s]]
    test_indices = [idx for s in test_speakers for idx in speaker_files[s]]
    
    splits = {
        'train': train_indices,
        'valid': val_indices,
        'test': test_indices,
    }
    
    print(f"  Train speakers ({len(train_speakers)}): {', '.join(train_speakers)}")
    print(f"  Val speakers ({len(val_speakers)}): {', '.join(val_speakers)}")
    print(f"  Test speakers ({len(test_speakers)}): {', '.join(test_speakers)}")

# Create manifest files
print("\nCreating manifest files...")
for split_name, indices in splits.items():
    if not indices:
        continue
    
    manifest_path = metadata_dir / f"{split_name}.tsv"
    wrd_path = metadata_dir / f"{split_name}.wrd"
    
    with open(manifest_path, 'w') as f:
        # Header
        f.write("/\n")
        
        # Write entries
        for idx in indices:
            fid = valid_fids[idx]
            video_path = data_dir / f"{fid}.mp4"
            audio_path = data_dir / f"{fid}.wav"
            num_audio = audio_num_frames[idx]
            num_video = video_num_frames[idx]
            
            # Format: id, video_path, audio_path, num_video_frames, num_audio_frames
            f.write(f"{fid}\t{video_path}\t{audio_path}\t{num_video}\t{num_audio}\n")
    
    with open(wrd_path, 'w') as f:
        for idx in indices:
            f.write(valid_labels[idx] + '\n')
    
    print(f"✅ Created {split_name}: {manifest_path} ({len(indices)} entries)")

# Create dictionary file
print("\nCreating dictionary file...")
dict_path = metadata_dir / "dict.wrd.txt"

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

print("\n🎉 Metadata preparation complete!")
print(f"\nOutput files:")
print(f"  - {nframes_audio_path}")
print(f"  - {nframes_video_path}")
for split_name in ['train', 'valid', 'test']:
    manifest_path = metadata_dir / f"{split_name}.tsv"
    if manifest_path.exists():
        print(f"  - {manifest_path}")
print(f"  - {dict_path}")
print(f"  - {vocab_dir}/")
