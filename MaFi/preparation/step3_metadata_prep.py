#!/usr/bin/env python3
"""
MaFi Step 3: Metadata Preparation

This script counts frames and creates manifest files for MaFi dataset.
Note: MaFi has no audio, so only video frames are counted.

Usage:
    # All speakers combined
    python step3_metadata_prep.py \
        --mafi-data-dir /path/to/output/mafi_video \
        --metadata-dir /path/to/output/metadata \
        --vocab-size 1000

    # Individual speaker
    python step3_metadata_prep.py \
        --mafi-data-dir /path/to/output/mafi_video \
        --metadata-dir /path/to/output/metadata_A1 \
        --speaker A1 \
        --vocab-size 1000
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from tempfile import NamedTemporaryFile

# Import vocabulary generation from LRS3
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "LRS3" / "preparation"))
from gen_subword import gen_vocab

parser = argparse.ArgumentParser(description="Generate metadata for MaFi dataset")
parser.add_argument(
    "--mafi-data-dir",
    type=str,
    required=True,
    help="Path to processed MaFi video directory",
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
    choices=['A1', 'A2', 'B1', 'B2', 'B3'],
    help="Process specific speaker (omit for all speakers combined)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=1000,
    help="Vocabulary size for sentencepiece (default: 1000)",
)
args = parser.parse_args()

data_dir = Path(args.mafi_data_dir)
metadata_dir = Path(args.metadata_dir)
metadata_dir.mkdir(parents=True, exist_ok=True)

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

# Count video frames (no audio for MaFi)
print("\nCounting video frames...")
video_num_frames = []
valid_fids = []
valid_labels = []

for fid, label in tqdm(zip(fids, labels), total=len(fids), desc="Counting frames"):
    video_fn = data_dir / f"{fid}.mp4"
    
    if not video_fn.exists():
        print(f"Warning: Missing video file: {video_fn}")
        continue
    
    try:
        cap = cv2.VideoCapture(str(video_fn))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if num_frames > 0:
            video_num_frames.append(num_frames)
            valid_fids.append(fid)
            valid_labels.append(label)
    except Exception as e:
        print(f"Warning: Error processing {fid}: {str(e)}")
        continue

print(f"Successfully counted frames for {len(valid_fids)} files")

# Create nframes.video file (no audio for MaFi)
nframes_video_path = data_dir / f"nframes{output_suffix}.video"
with open(nframes_video_path, 'w') as f:
    f.write('\n'.join([str(x) for x in video_num_frames]))

print(f"\n✅ Created: {nframes_video_path}")

# Generate vocabulary
print("\nGenerating sentencepiece vocabulary...")
vocab_size = args.vocab_size

# Check if we have enough data
total_words = len(set(" ".join(valid_labels).lower().split()))
if total_words < vocab_size:
    print(f"Warning: Only {total_words} unique words, adjusting vocab size")
    vocab_size = min(total_words, max(50, vocab_size // 2))

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

# Create manifest TSV file
print("\nCreating manifest file...")
manifest_path = metadata_dir / f"test{output_suffix}.tsv"

with open(manifest_path, 'w') as f:
    # Header
    f.write("/\n")
    
    # Write entries
    for fid, num_frames in zip(valid_fids, video_num_frames):
        video_path = data_dir / f"{fid}.mp4"
        # Format: id, video_path, num_frames
        f.write(f"{fid}\t{video_path}\t{num_frames}\n")

print(f"✅ Created manifest: {manifest_path} ({len(valid_fids)} entries)")

# Create word file
wrd_path = metadata_dir / f"test{output_suffix}.wrd"
with open(wrd_path, 'w') as f:
    f.write('\n'.join(valid_labels))

print(f"✅ Created word file: {wrd_path}")

# Create dictionary file
print("\nCreating dictionary file...")
dict_path = metadata_dir / "dict.wrd.txt"

# Read vocabulary from sentencepiece
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
print(f"  - {nframes_video_path}")
print(f"  - {manifest_path}")
print(f"  - {wrd_path}")
print(f"  - {dict_path}")
print(f"  - {vocab_dir}/")
