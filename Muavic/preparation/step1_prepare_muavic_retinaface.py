#!/usr/bin/env python3
"""
MuAViC Dataset Preprocessing Pipeline (RetinaFace)
==================================================

Simple preprocessing script for MuAViC dataset using RetinaFace detector.
Extracts video segments using ffmpeg (no full video loading) and processes with RetinaFace.

Usage:
    python step1_prepare_muavic_retinaface.py \
        --data-dir /path/to/muavic/data \
        --root-dir /path/to/output \
        --language ar \
        --split test \
        --crop-type lips
"""

import argparse
import math
import os
import tempfile
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, OrderedDict

import torch
import ffmpeg

warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="MuAViC Preprocessing with RetinaFace")
parser.add_argument("--data-dir", type=str, required=True, help="Directory where MuAViC dataset is stored")
parser.add_argument("--root-dir", type=str, required=True, help="Root directory of preprocessed dataset")
parser.add_argument("--language", type=str, required=True, help="Language code (en, es, fr, pt, it, el, ar, de, ru)")
parser.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"], help="Dataset split")
parser.add_argument("--detector", type=str, default="retinaface", help="Type of face detector")
parser.add_argument("--crop-type", type=str, default="lips", choices=["lips", "face"], help="Crop type")
parser.add_argument("--face-threshold", type=float, default=0.0, help="Minimum face presence ratio (0.0-1.0). 0=disabled, 0.7=strict filtering")
parser.add_argument("--groups", type=int, default=1, help="Number of parallel jobs")
parser.add_argument("--job-index", type=int, default=0, help="Job index for parallel processing")
args = parser.parse_args()

# Set output size based on crop type
output_size = 96 if args.crop_type == "lips" else 224
crop_suffix = f"_{args.crop_type}" if args.crop_type != "lips" else ""
size_suffix = f"_{output_size}x{output_size}" if output_size != 96 else ""

dst_vid_dir = os.path.join(args.root_dir, "muavic", f"muavic_video{crop_suffix}{size_suffix}")
dst_txt_dir = os.path.join(args.root_dir, "muavic", f"muavic_text{crop_suffix}{size_suffix}")

# Setup RetinaFace
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'TCD-TIMIT', 'preparation'))

from detectors.retinaface.detector import LandmarksDetector
from detectors.retinaface.video_process import VideoProcess
from utils import save_vid_aud_txt

landmarks_detector = LandmarksDetector(device="cuda:0")

if args.crop_type == "lips":
    start_idx, stop_idx = 48, 68
elif args.crop_type == "face":
    start_idx, stop_idx = 17, 68

video_process = VideoProcess(
    crop_width=output_size,
    crop_height=output_size,
    start_idx=start_idx,
    stop_idx=stop_idx,
    convert_gray=False
)

# Find segments file
segments_file = Path(args.data_dir) / "mtedx" / f"{args.language}-{args.language}" / "data" / args.split / "txt" / "segments"
if not segments_file.exists():
    raise FileNotFoundError(f"Segments file not found: {segments_file}")

print(f"Reading segments from: {segments_file}")

# Parse segments and group by video
video_to_segments = defaultdict(list)
all_segments = []

with open(segments_file, 'r') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split()
        if len(parts) == 4:
            seg_id, video_id, start_sec, end_sec = parts
            segment = {
                'seg_id': seg_id,
                'video_id': video_id,
                'start_sec': float(start_sec),
                'end_sec': float(end_sec),
                'transcript_idx': idx
            }
            video_to_segments[video_id].append(segment)
            all_segments.append(segment)

print(f"Found {len(all_segments)} segments from {len(video_to_segments)} videos")

# Load transcripts
txt_file = Path(args.data_dir) / "mtedx" / f"{args.language}-{args.language}" / "data" / args.split / "txt" / f"{args.split}.{args.language}"
with open(txt_file, 'r', encoding='utf-8') as f:
    transcripts = [line.strip() for line in f]

# Sort videos by number of segments
video_to_segments = OrderedDict(sorted(video_to_segments.items(), key=lambda x: len(x[1])))

# Split for parallel processing
video_ids = list(video_to_segments.keys())
unit = math.ceil(len(video_ids) / args.groups)
videos_to_process = video_ids[args.job_index * unit : (args.job_index + 1) * unit]

total_segs = sum(len(video_to_segments[vid]) for vid in videos_to_process)
print(f"Processing {len(videos_to_process)} videos ({total_segs} segments)")

# Process videos
csv_data = []
processed_count = 0
skipped_count = 0

for video_id in tqdm(videos_to_process, desc="Videos"):
    segments = video_to_segments[video_id]
    
    # Find video file
    video_path = Path(args.data_dir) / "mtedx" / "video" / args.language / args.split / f"{video_id}.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        skipped_count += len(segments)
        continue
    
    # Process each segment
    for segment in segments:
        seg_id = segment['seg_id']
        start_sec = segment['start_sec']
        end_sec = segment['end_sec']
        duration = end_sec - start_sec
        transcript = transcripts[segment['transcript_idx']]
        
        # Output paths
        out_dir = Path(dst_vid_dir) / args.language / args.split / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        dst_vid = out_dir / f"{seg_id}.mp4"
        dst_aud = out_dir / f"{seg_id}.wav"
        dst_txt = Path(dst_txt_dir) / args.language / args.split / video_id / f"{seg_id}.txt"
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if dst_vid.exists():
            processed_count += 1
            continue
        
        tmp_path = None
        try:
            # Extract segment to temp file using ffmpeg (no full video loading!)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Extract video segment
            (
                ffmpeg
                .input(str(video_path), ss=start_sec, t=duration)
                .output(tmp_path, vcodec='libx264', acodec='aac', loglevel='error')
                .overwrite_output()
                .run()
            )
            
            # Load segment (small, only this segment!)
            import torchvision
            import torchaudio
            
            segment_video = torchvision.io.read_video(tmp_path, pts_unit="sec")[0].numpy()
            
            # Process with RetinaFace
            landmarks = landmarks_detector(segment_video)
            
            # Optional face presence filtering (filters slides/audience shots)
            if args.face_threshold > 0:
                if landmarks is not None and len(landmarks) > 0:
                    # Count frames with valid, good-quality landmarks
                    valid_frames = 0
                    for lm in landmarks:
                        if lm is not None and len(lm) > 0:
                            # Check if landmarks are reasonable (not too small/blurry)
                            # Calculate face bounding box size from landmarks
                            lm_array = lm[0] if isinstance(lm, list) else lm
                            if len(lm_array) >= 68:  # Full face landmarks
                                # Get face width/height from landmarks
                                x_coords = lm_array[:, 0]
                                y_coords = lm_array[:, 1]
                                face_width = x_coords.max() - x_coords.min()
                                face_height = y_coords.max() - y_coords.min()
                                
                                # Filter out very small faces (likely blurry/far away)
                                # Minimum 40 pixels for lips crop, 80 for face crop
                                min_size = 40 if args.crop_type == "lips" else 80
                                if face_width >= min_size and face_height >= min_size:
                                    valid_frames += 1
                            else:
                                valid_frames += 1  # Accept if we can't check size
                    
                    total_frames = len(segment_video)
                    face_ratio = valid_frames / total_frames if total_frames > 0 else 0
                    
                    if face_ratio < args.face_threshold:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        skipped_count += 1
                        continue
            
            video_data = video_process(segment_video, landmarks)
            
            if video_data is None:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                skipped_count += 1
                continue
            
            # Load audio segment
            waveform, sr = torchaudio.load(tmp_path, normalize=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            audio_data = torch.mean(waveform, dim=0, keepdim=True)
            
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Quality check
            if len(video_data) < 5 or audio_data.size(1) < 1000:
                skipped_count += 1
                continue
            
            # Save
            save_vid_aud_txt(
                str(dst_vid), str(dst_aud), str(dst_txt),
                video_data, audio_data, transcript,
                video_fps=25, audio_sample_rate=16000
            )
            
            # CSV entry
            rel_path = f"muavic_video{crop_suffix}{size_suffix}/{args.language}/{args.split}/{video_id}/{seg_id}.mp4"
            csv_data.append([
                args.language, args.split, seg_id, video_id, rel_path,
                transcript, len(transcript.split()), start_sec, end_sec,
                args.detector, args.crop_type, f"{output_size}x{output_size}"
            ])
            
            processed_count += 1
            
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            skipped_count += 1
            continue

# Save CSV
if csv_data:
    import pandas as pd
    labels_dir = Path(args.root_dir) / "muavic" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    csv_file = labels_dir / f"muavic_{args.language}_{args.split}{crop_suffix}{size_suffix}.csv"
    
    df = pd.DataFrame(csv_data, columns=[
        'language', 'split', 'seg_id', 'video_id', 'video_path', 'transcript', 'word_count',
        'start_sec', 'end_sec', 'detector', 'crop_type', 'resolution'
    ])
    df.to_csv(csv_file, index=False)
    print(f"\n📊 Saved: {csv_file} ({len(df)} samples)")

print(f"✅ Done! Processed: {processed_count}, Skipped: {skipped_count}")
