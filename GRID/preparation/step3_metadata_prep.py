#!/usr/bin/env python3
"""
GRID Step 3: Metadata Preparation with SPM Tokenization

This script counts frames, creates manifest files, and tokenizes labels using 
the shared SentencePiece model (unigram5000) for GRID dataset.
Supports speaker-based train/val/test splits.

Usage:
    # Inference-only (no train/valid/test split; writes all.tsv/all.wrd)
    python step3_metadata_prep.py \
        --grid-data-dir /path/to/output/grid_video \
        --metadata-dir /path/to/output/metadata \
        --no-split

    # All speakers with speaker-based splits
    python step3_metadata_prep.py \
        --grid-data-dir /path/to/output/grid_video \
        --metadata-dir /path/to/output/metadata \
        --split-ratios 0.7,0.15,0.15

    # Individual speaker
    python step3_metadata_prep.py \
        --grid-data-dir /path/to/output/grid_video \
        --metadata-dir /path/to/output/metadata_s1 \
        --speaker s1
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile
import random

# Import SPM tokenizer
from transforms import TextTransform

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
    "--no-split",
    action="store_true",
    help="Don't create train/valid/test splits; write a single all.tsv/all.wrd for inference",
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

# Parse split ratios (only used when splitting)
split_ratios = None
if not args.no_split:
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

# Initialize SPM tokenizer
print("\nInitializing SentencePiece tokenizer...")
text_transform = TextTransform()
print(f"✅ SPM model loaded with {len(text_transform.token_list)} tokens")

# Create nframes files
nframes_audio_path = data_dir / f"nframes{output_suffix}.audio"
nframes_video_path = data_dir / f"nframes{output_suffix}.video"

with open(nframes_audio_path, 'w') as f:
    f.write('\n'.join([str(x) for x in audio_num_frames]))

with open(nframes_video_path, 'w') as f:
    f.write('\n'.join([str(x) for x in video_num_frames]))

print(f"\n✅ Created: {nframes_audio_path}")
print(f"✅ Created: {nframes_video_path}")

splits = None
if args.no_split:
    print("\nNo-split mode enabled: will write a single all.tsv/all.wrd for inference.")
    splits = {"all": list(range(len(valid_fids)))}
else:
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

# Create dictionary file from SPM model
print("\nCreating dictionary file from SPM model...")
dict_path = metadata_dir / "dict.wrd.txt"

with open(dict_path, 'w') as f:
    for idx, token in enumerate(text_transform.token_list):
        if token not in ['<blank>', '<eos>', '<unk>']:
            f.write(f"{token} {idx}\n")

print(f"✅ Created dictionary: {dict_path}")

# Create SPM-tokenized labels file
print("\nCreating SPM-tokenized labels...")
tokens_path = metadata_dir / "tokens.txt"

with open(tokens_path, 'w') as f:
    for label in valid_labels:
        token_ids = text_transform.tokenize(label)
        token_str = " ".join(str(t.item()) for t in token_ids)
        f.write(f"{token_str}\n")

print(f"✅ Created tokenized labels: {tokens_path}")

# Create simple label.csv for inference pipelines
# Format (no header):
#   dataset,video_path,token_ids(space-separated)
# Example:
#   grid,/abs/path/to/grid_video/s1/bbaf2n.mp4,3253 1629 46 330 138 76
print("\nCreating label.csv (simple format)...")
label_csv_path = metadata_dir / "label.csv"

dataset_name = "grid"
with open(label_csv_path, "w") as f:
    for fid, label in zip(valid_fids, valid_labels):
        # fid is like: s1/bbaf2n (no extension)
        video_abs = str((data_dir / f"{fid}.mp4").resolve())
        token_ids = text_transform.tokenize(label)
        token_str = " ".join(str(t.item()) for t in token_ids)
        f.write(f"{dataset_name},{video_abs},{token_str}\n")

print(f"✅ Created label CSV: {label_csv_path}")

# Create Auto-AVSR style 4-column CSV (matches LRS2):
#   dataset,rel_video_path,input_length(nframes_video),token_ids
print("\nCreating Auto-AVSR 4-column CSV...")
avsr_csv_path = metadata_dir / "grid_test_transcript_lengths_seg16s.csv"
with open(avsr_csv_path, "w") as f:
    for idx, (fid, label) in enumerate(zip(valid_fids, valid_labels)):
        # Relative path is relative to dataset root (data_dir)
        rel_vid = os.path.relpath(str((data_dir / f"{fid}.mp4").resolve()), start=str(data_dir.resolve()))
        # Use video frame counts as input_length
        nf = video_num_frames[idx]
        token_ids = text_transform.tokenize(label)
        token_str = " ".join(str(t.item()) for t in token_ids)
        f.write(f"{dataset_name},{rel_vid},{nf},{token_str}\n")

print(f"✅ Created Auto-AVSR CSV: {avsr_csv_path}")

print("\n🎉 Metadata preparation complete!")
print(f"\nOutput files:")
print(f"  - {nframes_audio_path}")
print(f"  - {nframes_video_path}")
for split_name in ['train', 'valid', 'test']:
    manifest_path = metadata_dir / f"{split_name}.tsv"
    if manifest_path.exists():
        print(f"  - {manifest_path}")
print(f"  - {dict_path}")
print(f"  - {tokens_path}")


