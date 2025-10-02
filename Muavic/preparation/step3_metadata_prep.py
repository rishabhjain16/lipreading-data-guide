#!/usr/bin/env python3
"""
MuAViC Step 3: Metadata Preparation

This script counts frames and creates manifest files for MuAViC dataset.
It follows the same pattern as TCD-TIMIT but handles MuAViC's multilingual structure.

Usage:
    python step3_metadata_prep.py \
        --muavic-data-dir /path/to/processed/muavic/muavic_video \
        --metadata-dir /path/to/processed/muavic/metadata \
        --language en \
        --vocab-size 1000
"""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

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
            
            total_num_frames.append([fid, num_frames_audio, num_frames_video])
        except Exception as e:
            print(f"Error processing {fid}: {e}")
            continue
    
    return total_num_frames

def create_tsv_manifest(fids, base_dir, output_file, text_dir):
    """Create TSV manifest file"""
    print(f"Creating TSV manifest: {output_file}")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("id\taudio\tn_frames\ttgt_text\tspeaker\n")
        
        for fid in tqdm(fids, desc="Creating manifest"):
            audio_path = os.path.join(base_dir, fid + ".wav")
            video_path = os.path.join(base_dir, fid + ".mp4")
            text_path = os.path.join(text_dir, fid + ".txt")
            
            if not os.path.exists(audio_path) or not os.path.exists(video_path):
                continue
            
            # Get number of frames
            try:
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except:
                continue
            
            # Get transcript
            transcript = ""
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as tf:
                    transcript = tf.read().strip()
            
            # Extract speaker/video ID from fid
            video_id = os.path.basename(fid)
            
            # Write TSV line
            f.write(f"{fid}\t{audio_path}\t{n_frames}\t{transcript}\t{video_id}\n")

def create_wrd_file(fids, text_dir, output_file):
    """Create word file with transcriptions"""
    print(f"Creating word file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for fid in tqdm(fids, desc="Creating word file"):
            text_path = os.path.join(text_dir, fid + ".txt")
            
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as tf:
                    transcript = tf.read().strip()
                    f.write(f"{transcript}\n")
            else:
                f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description='MuAViC Metadata Preparation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--muavic-data-dir', type=str, required=True,
                       help='MuAViC processed data directory')
    parser.add_argument('--metadata-dir', type=str, required=True,
                       help='Output directory for metadata files')
    parser.add_argument('--language', type=str, required=True,
                       help='Language code (en, es, fr, pt, it, el, ar, de, ru)')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size for subword tokenization')
    
    args = parser.parse_args()
    
    data_dir = Path(args.muavic_data_dir)
    metadata_dir = Path(args.metadata_dir)
    language = args.language
    
    # Create metadata directory
    os.makedirs(metadata_dir, exist_ok=True)
    
    print(f"Processing metadata for {language}...")
    
    # Determine text directory (same structure as video dir)
    text_dir = Path(str(data_dir).replace('muavic_video', 'muavic_text'))
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print(f"{'='*50}")
        
        # Read file list
        file_list = data_dir / f"{language}_{split}.txt"
        
        if not file_list.exists():
            print(f"Warning: File list not found: {file_list}")
            continue
        
        with open(file_list, 'r') as f:
            fids = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(fids)} files in {split} split")
        
        # Count frames
        frame_counts = count_frames(fids, data_dir)
        
        # Save frame counts
        nframes_audio_file = metadata_dir / f"nframes.audio.{split}"
        nframes_video_file = metadata_dir / f"nframes.video.{split}"
        
        with open(nframes_audio_file, 'w') as f:
            for fid, n_audio, n_video in frame_counts:
                f.write(f"{fid} {n_audio}\n")
        
        with open(nframes_video_file, 'w') as f:
            for fid, n_audio, n_video in frame_counts:
                f.write(f"{fid} {n_video}\n")
        
        print(f"✅ Saved frame counts: {nframes_audio_file}, {nframes_video_file}")
        
        # Create TSV manifest
        tsv_file = metadata_dir / f"{split}.tsv"
        create_tsv_manifest(fids, data_dir, tsv_file, text_dir)
        print(f"✅ Saved TSV manifest: {tsv_file}")
        
        # Create word file
        wrd_file = metadata_dir / f"{split}.wrd"
        create_wrd_file(fids, text_dir, wrd_file)
        print(f"✅ Saved word file: {wrd_file}")
    
    print(f"\n{'='*50}")
    print(f"✅ Metadata preparation complete!")
    print(f"   Output directory: {metadata_dir}")
    print(f"{'='*50}")
    
    return 0

if __name__ == "__main__":
    exit(main())
