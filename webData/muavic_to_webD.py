#!/usr/bin/env python3
"""
MuAViC to WebDataset Converter
===============================

Converts preprocessed MuAViC dataset to WebDataset format for efficient training.

Usage:
    # Standard preprocessing output
    python webData/muavic_to_webD.py \
        --video_root /path/to/output/muavic/muavic_video \
        --text_root /path/to/output/muavic/muavic_text \
        --csv_file /path/to/output/muavic/labels/muavic_ar_train.csv \
        --output_dir /path/to/webdataset/muavic \
        --dataset_name muavic_ar \
        --samples_per_shard 500

    # Smart preprocessing with seg-duration
    python webData/muavic_to_webD.py \
        --video_root /path/to/output/muavic/muavic_video_seg16s \
        --text_root /path/to/output/muavic/muavic_text_seg16s \
        --csv_file /path/to/output/muavic/labels/muavic_ar_train_seg16s.csv \
        --output_dir /path/to/webdataset/muavic \
        --dataset_name muavic_ar_seg16s \
        --samples_per_shard 500
"""

import os
import json
import argparse
from pathlib import Path
import csv
import re
import shutil
import tempfile
import hashlib
import uuid
import ffmpeg
import cv2
import tqdm
import webdataset as wds


class MuavicWebDatasetConverter:
    def __init__(self, video_root, text_root, csv_file, output_dir, samples_per_shard, dataset_name):
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.samples_per_shard = samples_per_shard
        self.dataset_name = dataset_name

        if not self.video_root.is_dir():
            raise FileNotFoundError(f"Video root {video_root} does not exist.")
        if not self.text_root.is_dir():
            raise FileNotFoundError(f"Text root {text_root} does not exist.")
        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file {csv_file} does not exist.")

        # Extract split from CSV filename (e.g., muavic_ar_train.csv -> train)
        parts = self.csv_file.stem.split('_')
        # Handle different naming patterns
        if 'train' in self.csv_file.stem:
            self.split = 'train'
        elif 'valid' in self.csv_file.stem or 'val' in self.csv_file.stem:
            self.split = 'valid'
        elif 'test' in self.csv_file.stem:
            self.split = 'test'
        else:
            self.split = 'custom'

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_pattern = str(self.output_dir / f"{self.dataset_name}_{self.split}-%06d.tar")

    def parse_csv(self):
        """Parse MuAViC CSV file"""
        samples = []
        with self.csv_file.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # MuAViC CSV format: language,split,seg_id,video_id,video_path,transcript,word_count,start_sec,end_sec,detector,crop_type,resolution
                video_path_str = row['video_path']
                
                # Extract relative path (remove first component which is the dataset folder name)
                video_rel_path = Path(*Path(video_path_str).parts[1:])
                
                video_path = self.video_root / video_rel_path
                text_path = self.text_root / video_rel_path.parent / (video_rel_path.stem + '.txt')
                wav_path = video_path.with_suffix('.wav')

                # Verify files exist
                if not video_path.is_file():
                    print(f"Warning: Video not found: {video_path}")
                    continue
                if not text_path.is_file():
                    print(f"Warning: Text not found: {text_path}")
                    continue

                sample_id = video_rel_path.with_suffix('').as_posix()

                samples.append({
                    'video_path': video_path,
                    'text_path': text_path,
                    'wav_path': wav_path if wav_path.is_file() else None,
                    'sample_id': sample_id,
                    'language': row.get('language', 'unknown'),
                    'video_id': row.get('video_id', ''),
                    'seg_id': row.get('seg_id', ''),
                    'start_sec': float(row.get('start_sec', 0)),
                    'end_sec': float(row.get('end_sec', 0)),
                })
        
        print(f"Loaded {len(samples)} samples from {self.csv_file.name}")
        return samples

    def read_text(self, path):
        """Read and normalize transcript text"""
        text = path.read_text(encoding='utf-8').strip()
        # For Arabic and other non-Latin scripts, don't filter characters
        # Just normalize whitespace
        text = ' '.join(text.split())
        return text

    def get_video_metadata(self, video_path):
        """Extract video metadata using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        duration = frames / fps if fps > 0 else 0
        
        return {
            "fps": fps,
            "num_frames": frames,
            "duration": duration,
            "width": width,
            "height": height
        }

    def generate_temp_path(self, video_path, wav_path, temp_dir):
        """Generate unique temporary file path"""
        key_str = str(video_path) + str(wav_path)
        hashed = hashlib.md5(key_str.encode()).hexdigest()
        uid = uuid.uuid4().hex[:6]
        return temp_dir / f"{hashed}_{uid}.mp4"

    def merge_audio_video(self, video_path, wav_path, out_path):
        """Merge video and audio using ffmpeg"""
        v = ffmpeg.input(str(video_path))
        a = ffmpeg.input(str(wav_path))
        ffmpeg.output(
            v.video, a.audio, str(out_path),
            vcodec='copy', acodec='aac', strict='experimental',
            loglevel='panic'
        ).overwrite_output().run()

    def convert(self):
        """Convert MuAViC dataset to WebDataset format"""
        samples = self.parse_csv()
        if len(samples) == 0:
            raise ValueError("No valid samples found")

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"muavic_merge_{self.split}_"))
        print(f"Temporary dir: {tmp_dir}")

        written_count = 0
        skipped_count = 0

        with wds.ShardWriter(self.output_pattern, maxcount=self.samples_per_shard) as sink:
            for idx, sample in enumerate(tqdm.tqdm(samples, desc=f"Converting {self.split}")):
                try:
                    # Merge audio and video if separate
                    if sample['wav_path'] and sample['wav_path'].is_file():
                        merged_path = self.generate_temp_path(
                            sample['video_path'], sample['wav_path'], tmp_dir
                        )
                        if not merged_path.is_file():
                            self.merge_audio_video(
                                sample['video_path'], sample['wav_path'], merged_path
                            )
                        video_file = merged_path
                    else:
                        video_file = sample['video_path']

                    # Read video bytes
                    video_bytes = video_file.read_bytes()
                    
                    # Read transcript
                    transcript = self.read_text(sample['text_path'])
                    
                    # Get video metadata
                    meta = self.get_video_metadata(video_file)
                    length_str = str(meta['num_frames'])
                    
                    # Add MuAViC-specific metadata
                    meta['sample_id'] = sample['sample_id']
                    meta['length'] = length_str
                    meta['language'] = sample['language']
                    meta['video_id'] = sample['video_id']
                    meta['seg_id'] = sample['seg_id']
                    meta['start_sec'] = sample['start_sec']
                    meta['end_sec'] = sample['end_sec']
                    meta['transcript'] = transcript

                    # Create WebDataset item
                    item = {
                        '__key__': f"{idx:08d}",
                        'video': video_bytes,
                        'label': transcript.encode('utf-8'),
                        'length': length_str.encode('utf-8'),
                        'sample_id': sample['sample_id'].encode('utf-8'),
                        'language': sample['language'].encode('utf-8'),
                        'json': json.dumps(meta, ensure_ascii=False).encode('utf-8'),
                    }

                    sink.write(item)
                    written_count += 1
                    
                    # Progress update
                    if (idx + 1) % 500 == 0:
                        print(f"\nWritten {written_count} samples, skipped {skipped_count}...")

                except Exception as e:
                    print(f"\nError processing sample {sample['sample_id']}: {e}")
                    skipped_count += 1
                    continue

        # Cleanup
        shutil.rmtree(tmp_dir)
        
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"Total samples: {len(samples)}")
        print(f"Written: {written_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert MuAViC dataset to WebDataset format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--video_root', required=True, help='Root directory of video files')
    parser.add_argument('--text_root', required=True, help='Root directory of text files')
    parser.add_argument('--csv_file', required=True, help='Path to CSV metadata file')
    parser.add_argument('--output_dir', required=True, help='Output directory for WebDataset shards')
    parser.add_argument('--dataset_name', required=True, help='Dataset name (e.g., muavic_ar)')
    parser.add_argument('--samples_per_shard', type=int, default=500, help='Number of samples per shard')

    args = parser.parse_args()

    converter = MuavicWebDatasetConverter(
        video_root=Path(args.video_root),
        text_root=Path(args.text_root),
        csv_file=Path(args.csv_file),
        output_dir=Path(args.output_dir),
        samples_per_shard=args.samples_per_shard,
        dataset_name=args.dataset_name,
    )
    converter.convert()
