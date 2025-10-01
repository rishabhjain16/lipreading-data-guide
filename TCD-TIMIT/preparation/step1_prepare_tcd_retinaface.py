#!/usr/bin/env python3
"""
TCD-TIMIT Dataset Preprocessing Pipeline (RetinaFace)
====================================================

Preprocessing script for TCD-TIMIT dataset using RetinaFace detector,
following the same approach as VoxCeleb2 preprocessing.

Usage:
    python step1_prepare_tcd_retinaface.py \
        --data-dir /path/to/TCD-TIMIT \
        --root-dir /path/to/output \
        --subset volunteers \
        --detector retinaface

Note: Requires RetinaFace dependencies. Install with: pip install ibug-face_detection ibug-face_alignment
"""

import argparse
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
import torch
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud_txt
import json
from pathlib import Path

warnings.filterwarnings("ignore")

# TCD-TIMIT sentence mapping (TIMIT corpus standard sentences)
with open("./timit_sentences.json", "r", encoding="utf-8") as f:
    TIMIT_SENTENCES = json.load(f)

# Argument parsing
parser = argparse.ArgumentParser(description="TCD-TIMIT Preprocessing with RetinaFace")
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="Directory where TCD-TIMIT dataset is stored",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--subset",
    type=str,
    required=True,
    choices=["volunteers", "lipspeakers"],
    help="Subset to process",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Type of face detector",
)
parser.add_argument(
    "--crop-type",
    type=str,
    default="lips",
    choices=["lips", "face"],
    help="Crop type: lips (96x96, mouth region) or face (224x224, full face)",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Max duration (second) for each segment, (Default: 16)",
)
parser.add_argument(
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
args = parser.parse_args()

def parse_mlf_transcripts(mlf_file_path):
    """Parse .mlf file to extract transcript mappings"""
    
    transcripts = {}
    current_file = None
    current_phonemes = []
    
    with open(mlf_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line == "#!MLF!#":
                continue
                
            if line.startswith('"') and line.endswith('.rec"'):
                # New file entry - save previous if exists
                if current_file and current_phonemes:
                    transcript = ' '.join(current_phonemes)
                    transcripts[current_file] = transcript
                
                # Extract transcript ID from filename
                current_file = line[1:-5]  # Remove quotes and .rec extension
                current_file = os.path.basename(current_file)  # Get just the filename
                current_phonemes = []
                
            elif line == ".":
                # End of current file's transcript
                if current_file and current_phonemes:
                    transcript = ' '.join(current_phonemes)
                    transcripts[current_file] = transcript
                current_file = None
                current_phonemes = []
                
            else:
                # Phoneme entry: start_time end_time phoneme
                parts = line.split()
                if len(parts) == 3:
                    start_time, end_time, phoneme = parts
                    # Skip silence markers
                    if phoneme.lower() not in ['sil', 'sp']:
                        current_phonemes.append(phoneme)
    
    # Handle last entry if file doesn't end with '.'
    if current_file and current_phonemes:
        transcript = ' '.join(current_phonemes)
        transcripts[current_file] = transcript
    
    return transcripts

# Constants
seg_vid_len = args.seg_duration * 25
seg_aud_len = args.seg_duration * 16000

# Set output size based on crop type
if args.crop_type == "lips":
    output_size = 96
elif args.crop_type == "face":
    output_size = 224

# Create directory names with crop type and size suffixes
crop_suffix = f"_{args.crop_type}" if args.crop_type != "lips" else ""
size_suffix = f"_{output_size}x{output_size}" if output_size != 96 else ""
detector_suffix = f"_{args.detector}" if args.detector != "retinaface" else ""

dst_vid_dir = os.path.join(args.root_dir, "tcd_timit", f"tcd_timit_video{crop_suffix}{size_suffix}{detector_suffix}")
dst_txt_dir = os.path.join(args.root_dir, "tcd_timit", f"tcd_timit_text{crop_suffix}{size_suffix}{detector_suffix}")

# Load data with custom crop settings
if args.detector == "retinaface":
    from detectors.retinaface.detector import LandmarksDetector
    from detectors.retinaface.video_process import VideoProcess
    
    landmarks_detector = LandmarksDetector(device="cuda:0")
    
    # Configure crop settings based on crop_type
    if args.crop_type == "lips":
        start_idx, stop_idx = 48, 68  # Mouth landmarks
        output_size = 96
    elif args.crop_type == "face":
        start_idx, stop_idx = 17, 68  # Face landmarks (eyebrows to chin)
        output_size = 224
    
    video_process = VideoProcess(
        crop_width=output_size,
        crop_height=output_size,
        start_idx=start_idx,
        stop_idx=stop_idx,
        convert_gray=False
    )
    
    class CustomAVSRDataLoader:
        def __init__(self, modality):
            self.modality = modality
            if modality == "video":
                self.landmarks_detector = landmarks_detector
                self.video_process = video_process
        
        def load_data(self, data_filename, landmarks=None):
            if self.modality == "audio":
                import torchaudio
                waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                return waveform
            elif self.modality == "video":
                import torchvision
                video = torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
                if not landmarks:
                    landmarks = self.landmarks_detector(video)
                video = self.video_process(video, landmarks)
                if video is None:
                    raise TypeError("video cannot be None")
                return torch.tensor(video)
    
    vid_dataloader = CustomAVSRDataLoader(modality="video")
    aud_dataloader = CustomAVSRDataLoader(modality="audio")
else:
    # Fallback to original AVSRDataLoader for other detectors
    vid_dataloader = AVSRDataLoader(
        modality="video", detector=args.detector, convert_gray=False
    )
    aud_dataloader = AVSRDataLoader(modality="audio")

def find_video_files(data_dir, subset):
    """Find all video files in the subset directory"""
    subset_path = Path(data_dir) / subset
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(subset_path.rglob(f"*{ext}"))
    
    return sorted(video_files)

def extract_file_info(video_path, subset_dir):
    """Extract speaker ID and transcript ID from video path"""
    rel_path = video_path.relative_to(subset_dir)
    parts = rel_path.parts
    # Expecting: speaker/Clips/camera_view/filename
    if len(parts) >= 4:
        speaker_id = parts[0]
        session = parts[1]  # e.g., 'Clips'
        camera_view = parts[2]  # e.g., '30degcam', 'frontalcam'
        video_name = parts[-1]
        transcript_id = video_name.split('.')[0]
        return speaker_id, session, camera_view, transcript_id
    elif len(parts) >= 2:
        # Fallback: just speaker and filename
        speaker_id = parts[0]
        session = None
        camera_view = None
        video_name = parts[-1]
        transcript_id = video_name.split('.')[0]
        return speaker_id, session, camera_view, transcript_id
    return None, None, None, None

# Find transcript files
data_path = Path(args.data_dir)

# The MLF files are in the root data directory
subset_mapping = {
    "volunteers": "volunteer",
    "lipspeakers": "lipspeaker"
}

mlf_prefix = subset_mapping.get(args.subset, args.subset)
mlf_filename = f"{mlf_prefix}_labelfiles.mlf"
mlf_file = data_path / mlf_filename

if not mlf_file.exists():
    print(f"❌ Error: MLF file not found at {mlf_file}")
    exit(1)

# Parse the transcript file
all_transcripts = parse_mlf_transcripts(mlf_file)

# Find video files
video_files = find_video_files(args.data_dir, args.subset)
if not video_files:
    print("❌ Error: No video files found")
    exit(1)

# Group by (speaker, session, camera_view) and create file list similar to VoxCeleb2
filenames = []
subset_dir = data_path / args.subset

for video_path in video_files:
    speaker_id, session, camera_view, transcript_id = extract_file_info(video_path, subset_dir)
    if speaker_id and transcript_id and transcript_id in all_transcripts:
        filenames.append(str(video_path))

unit = math.ceil(len(filenames) / args.groups)
files_to_process = filenames[args.job_index * unit : (args.job_index + 1) * unit]

print(f"Processing {len(files_to_process)} files out of {len(filenames)} total files")

# Initialize CSV data collection
csv_data = []
processed_count = 0
skipped_count = 0

for vid_filename in tqdm(files_to_process):
    vid_path = Path(vid_filename)
    
    # Extract file info
    speaker_id, session, camera_view, transcript_id = extract_file_info(vid_path, subset_dir)
    
    if transcript_id not in all_transcripts:
        continue
    
    transcript = all_transcripts[transcript_id]
    sentence = TIMIT_SENTENCES.get(transcript_id, transcript)
    
    try:
        video_data = vid_dataloader.load_data(vid_filename)
        audio_data = aud_dataloader.load_data(vid_filename)
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        skipped_count += 1
        continue
    if video_data is None:
        skipped_count += 1
        continue

    # Process whole video (TCD-TIMIT videos are short single utterances)
    # Create unique file ID
    if session and camera_view:
        unique_id = f"{speaker_id}_{session}_{camera_view}_{transcript_id}"
        dst_vid_filename = os.path.join(dst_vid_dir, args.subset, speaker_id, session, camera_view, f"{unique_id}.mp4")
        dst_aud_filename = os.path.join(dst_vid_dir, args.subset, speaker_id, session, camera_view, f"{unique_id}.wav")
        dst_txt_filename = os.path.join(dst_txt_dir, args.subset, speaker_id, session, camera_view, f"{unique_id}.txt")
    else:
        unique_id = f"{speaker_id}_{transcript_id}"
        dst_vid_filename = os.path.join(dst_vid_dir, args.subset, speaker_id, f"{unique_id}.mp4")
        dst_aud_filename = os.path.join(dst_vid_dir, args.subset, speaker_id, f"{unique_id}.wav")
        dst_txt_filename = os.path.join(dst_txt_dir, args.subset, speaker_id, f"{unique_id}.txt")
    
    # Use the whole video (no segmentation needed for TCD-TIMIT)
    video_length = len(video_data)
    audio_length = audio_data.size(1)
    
    # Basic quality checks (more lenient for TCD-TIMIT)
    if video_length < 5 or audio_length < 1000:  # Very basic minimum checks
        skipped_count += 1
        continue

    # Save video, audio, and text
    save_vid_aud_txt(
        dst_vid_filename,
        dst_aud_filename,
        dst_txt_filename,
        video_data,
        audio_data,
        sentence,
        video_fps=25,
        audio_sample_rate=16000,
    )

    # Merge video and audio if requested
    if args.combine_av:
        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(
            in1["v"],
            in2["a"],
            dst_vid_filename[:-4] + ".m.mp4",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        out.run()
        os.remove(dst_aud_filename)
        os.remove(dst_vid_filename)
        shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)

    # Add to CSV data
    if session and camera_view:
        rel_video_path = f"tcd_timit_video{crop_suffix}{size_suffix}{detector_suffix}/{args.subset}/{speaker_id}/{session}/{camera_view}/{unique_id}.mp4"
    else:
        rel_video_path = f"tcd_timit_video{crop_suffix}{size_suffix}{detector_suffix}/{args.subset}/{speaker_id}/{unique_id}.mp4"
    
    csv_data.append([
        speaker_id,
        rel_video_path,
        sentence,
        len(sentence.split()),  # word count
        unique_id,
        transcript_id,
        args.detector,
        args.crop_type,
        f"{output_size}x{output_size}"
    ])
    
    processed_count += 1

# Save CSV at the end
if csv_data:
    import pandas as pd
    labels_dir = os.path.join(args.root_dir, "tcd_timit", "labels")
    os.makedirs(labels_dir, exist_ok=True)
    csv_filename = f"tcd_timit_{args.subset}{crop_suffix}{size_suffix}_{args.detector}.csv"
    csv_path = os.path.join(labels_dir, csv_filename)
    
    df = pd.DataFrame(csv_data, columns=[
        'speaker_id', 'video_path', 'transcript', 'word_count', 
        'unique_id', 'transcript_id', 'detector', 'crop_type', 'resolution'
    ])
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Saved CSV: {csv_path}")
    print(f"📊 Total: {len(df)} samples ({output_size}x{output_size}, {args.crop_type}, {args.detector})")

print(f"\n✅ Complete! Processed: {processed_count}, Skipped: {skipped_count}")