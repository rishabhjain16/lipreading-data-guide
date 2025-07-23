#!/usr/bin/env python3
"""
TCD-TIMIT Step 3: Metadata Preparation

This script counts frames and creates manifest files for TCD-TIMIT dataset.
It follows the same pattern as LRS2/LRS3 but handles TCD-TIMIT's speaker-based splits.

Usage:
    python step3_metadata_prep.py \
      --tcd-data-dir /path/to/processed/tcd_timit/tcd_timit_video_seg16s \
      --metadata-dir /path/to/processed/tcd_timit/metadata \
      --vocab-size 1000
"""

import os
import cv2
import shutil
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile

# Import from LRS3 preparation (reuse the vocabulary generation)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "LRS3" / "preparation"))
from gen_subword import gen_vocab

def count_frames(fids, base_dir):
    """Count frames in audio and video files"""
    print("Counting frames in audio and video files...")
    total_num_frames = []
    
    for fid in tqdm(fids, desc="Counting frames"):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        
        if not os.path.exists(wav_fn):
            print(f"Warning: Missing audio file: {wav_fn}")
            continue
        if not os.path.exists(video_fn):
            print(f"Warning: Missing video file: {video_fn}")
            continue
            
        try:
            num_frames_audio = len(wavfile.read(wav_fn)[1])
            cap = cv2.VideoCapture(video_fn)
            num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_num_frames.append([num_frames_audio, num_frames_video])
        except Exception as e:
            print(f"Warning: Error processing {fid}: {str(e)}")
            continue
    
    print(f"  Successfully counted frames for {len(total_num_frames)} files")
    return total_num_frames

def check_missing_files(fids, base_dir):
    """Check for missing audio/video files"""
    print("Checking for missing files...")
    missing = []
    
    for fid in tqdm(fids, desc="Checking files"):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        is_file = os.path.isfile(wav_fn) and os.path.isfile(video_fn)
        
        if not is_file:
            if not os.path.isfile(wav_fn):
                print(f"  Missing audio: {wav_fn}")
            if not os.path.isfile(video_fn):
                print(f"  Missing video: {video_fn}")
            missing.append(fid)
    
    if len(missing) > 0:
        print(f"  Found {len(missing)} files with missing audio/video")
    else:
        print(f"  All files present")
    
    return missing

def create_frame_files(tcd_data_dir, fids, num_frames):
    """Create nframes.audio and nframes.video files"""
    print("Creating frame count files...")
    
    audio_num_frames = [x[0] for x in num_frames]
    video_num_frames = [x[1] for x in num_frames]

    nframes_audio_path = os.path.join(tcd_data_dir, 'nframes.audio')
    nframes_video_path = os.path.join(tcd_data_dir, 'nframes.video')
    
    with open(nframes_audio_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
    with open(nframes_video_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in video_num_frames]))

    print(f"  Created: {nframes_audio_path}")
    print(f"  Created: {nframes_video_path}")
    
    return nframes_audio_path, nframes_video_path

def create_manifest_files(tcd_data_dir, metadata_dir, vocab_size):
    """Create manifest files for training"""
    print("Creating manifest files...")
    
    # Required files
    file_list = os.path.join(tcd_data_dir, "file.list")
    label_list = os.path.join(tcd_data_dir, "label.list")
    nframes_audio_file = os.path.join(tcd_data_dir, "nframes.audio")
    nframes_video_file = os.path.join(tcd_data_dir, "nframes.video")
    
    # Check if all required files exist
    required_files = [file_list, label_list, nframes_audio_file, nframes_video_file]
    for req_file in required_files:
        if not os.path.exists(req_file):
            print(f"Error: Required file not found: {req_file}")
            return None
    
    # Generate vocabulary
    print("Generating sentencepiece vocabulary...")
    vocab_dir = (Path(metadata_dir) / f"spm{vocab_size}").absolute()
    vocab_dir.mkdir(parents=True, exist_ok=True)
    smp_filename_prefix = f"spm_unigram{vocab_size}"
    
    # Read all label text
    label_text = [ln.strip() for ln in open(label_list).readlines()]
    
    # Check if we have enough data for the requested vocab size
    total_words = len(set(" ".join(label_text).lower().split()))
    
    if len(label_text) < 10 or total_words < vocab_size:
        print(f"Warning: Small dataset detected: {len(label_text)} samples, {total_words} unique words")
        print(f"Adjusting vocabulary size from {vocab_size} to {min(total_words, max(5, vocab_size//2))}")
        vocab_size = min(total_words, max(5, vocab_size//2))
        vocab_dir = (Path(metadata_dir) / f"smp{vocab_size}").absolute()
        vocab_dir.mkdir(parents=True, exist_ok=True)
        smp_filename_prefix = f"smp_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w") as f:
        for t in label_text:
            f.write(t.lower() + "\n")
        f.flush()  # Ensure data is written before training
        gen_vocab(Path(f.name), vocab_dir/smp_filename_prefix, 'unigram', vocab_size)
    
    vocab_path = (vocab_dir/smp_filename_prefix).as_posix() + '.txt'
    print(f"  Created vocabulary: {vocab_path}")

    def setup_target(target_dir, train, valid, test):
        """Setup target directory with train/valid/test splits"""
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        for name, data in zip(["train", "valid", "test"], [train, valid, test]):
            if not data:
                continue
                
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                for fid, label, nf_audio, nf_video in data:
                    # Convert file ID to full paths
                    video_path = os.path.abspath(f"{tcd_data_dir}/{fid}.mp4")
                    audio_path = os.path.abspath(f"{tcd_data_dir}/{fid}.wav")
                    
                    fo.write('\t'.join([
                        fid,
                        video_path,
                        audio_path,
                        str(nf_video),
                        str(nf_audio)
                    ])+'\n')
            
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")
        print(f"  Copied vocabulary to: {target_dir}/dict.wrd.txt")

    # Read all data
    fids = [x.strip() for x in open(file_list).readlines()]
    labels = [x.strip().lower() for x in open(label_list).readlines()]
    nfs_audio = [x.strip() for x in open(nframes_audio_file).readlines()]
    nfs_video = [x.strip() for x in open(nframes_video_file).readlines()]

    # Read dataset splits
    train, valid, test = [], [], []
    
    # Load split files created by step 2
    train_split_file = f"{tcd_data_dir}/train.txt"
    val_split_file = f"{tcd_data_dir}/val.txt"
    test_split_file = f"{tcd_data_dir}/test.txt"
    
    train_ids = set()
    valid_ids = set()
    test_ids = set()
    
    if os.path.exists(train_split_file):
        train_ids = set(line.strip() for line in open(train_split_file))
    if os.path.exists(val_split_file):
        valid_ids = set(line.strip() for line in open(val_split_file))
    if os.path.exists(test_split_file):
        test_ids = set(line.strip() for line in open(test_split_file))
    
    # Assign files to splits
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        # Extract transcript ID from file path for comparison
        transcript_id = fid.split('/')[-1]  # Last part after final /
        
        data_item = [fid, label, nf_audio, nf_video]
        
        if transcript_id in train_ids:
            train.append(data_item)
        elif transcript_id in valid_ids:
            valid.append(data_item)
        elif transcript_id in test_ids:
            test.append(data_item)
        else:
            # Default to train if not in any split
            train.append(data_item)

    output_dir = metadata_dir
    print(f"Setting up metadata directory: {output_dir}")
    setup_target(output_dir, train, valid, test)
    
    print(f"  Dataset splits:")
    print(f"    Train: {len(train)} samples")
    print(f"    Valid: {len(valid)} samples") 
    print(f"    Test: {len(test)} samples")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description='TCD-TIMIT Processing - Count frames and create manifest files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tcd-data-dir', type=str, required=True,
                        help='TCD-TIMIT data directory (contains file.list, label.list, and video files)')
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size for sentencepiece')
    
    args = parser.parse_args()
    
    # Validate input directory
    tcd_data_dir = Path(args.tcd_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    
    if not tcd_data_dir.exists():
        print(f"Error: TCD-TIMIT data directory not found: {tcd_data_dir}")
        return 1
    
    file_list_path = tcd_data_dir / 'file.list'
    if not file_list_path.exists():
        print(f"Error: file.list not found in: {tcd_data_dir}")
        print("Try running step2_generate_file_lists.py first to generate file.list and label.list")
        return 1
    
    print(f"Starting TCD-TIMIT processing...")
    print(f"Data directory: {tcd_data_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"Vocabulary size: {args.vocab_size}")
    print("-" * 60)
    
    try:
        # Read file list
        with open(file_list_path, 'r') as f:
            fids = [ln.strip() for ln in f.readlines()]
        print(f"Found {len(fids)} files to process")
        
        # Step 1: Check for missing files
        missing_fids = check_missing_files(fids, str(tcd_data_dir))
        
        if len(missing_fids) > 0:
            missing_list_path = tcd_data_dir / 'missing.list'
            with open(missing_list_path, 'w') as fo:
                fo.write('\n'.join(missing_fids) + '\n')
            print(f"Some audio/video files are missing. See: {missing_list_path}")
            print("Please resolve missing files before proceeding.")
            return 1
        
        # Step 2: Count frames
        num_frames = count_frames(fids, str(tcd_data_dir))
        
        if len(num_frames) == 0:
            print("No valid files found for frame counting")
            return 1
        
        # Step 3: Create frame count files
        nframes_audio_path, nframes_video_path = create_frame_files(
            str(tcd_data_dir), fids, num_frames
        )
        
        # Step 4: Create manifest files
        output_dir = create_manifest_files(str(tcd_data_dir), str(metadata_dir), args.vocab_size)
        
        print("-" * 60)
        print("Processing completed successfully!")
        print(f"Output directory: {output_dir}")
        print("Generated files:")
        print(f"   Frame counts: {nframes_audio_path}, {nframes_video_path}")
        print(f"   Manifests: train.tsv, valid.tsv, test.tsv")
        print(f"   Word files: train.wrd, valid.wrd, test.wrd")
        print(f"   Dictionary: dict.wrd.txt")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
