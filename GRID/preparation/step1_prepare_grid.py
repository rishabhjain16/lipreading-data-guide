#!/usr/bin/env python3
"""
GRID Corpus Dataset Preprocessing Pipeline (RetinaFace)
=======================================================

Preprocessing script for GRID Corpus - a controlled vocabulary audiovisual dataset.
Uses RetinaFace detector for face detection and lip/face cropping.

Usage:
    python step1_prepare_grid.py \
        --data-dir /media/rishabhjain/SSD/GRID \
        --root-dir /path/to/output \
        --crop-type lips

Note: GRID has 33 speakers (s1-s34, s21 missing), each with 1000 utterances.
"""

import argparse
import math
import os
import warnings
from pathlib import Path

import torch
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud_txt
import pandas as pd

warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="GRID Corpus Preprocessing with RetinaFace")
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="Directory where GRID dataset is stored",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Type of face detector (default: retinaface)",
)
parser.add_argument(
    "--crop-type",
    type=str,
    default="lips",
    choices=["lips", "face"],
    help="Crop type: lips (96x96, mouth region) or face (224x224, full face)",
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

# Setup output directories based on crop type
if args.crop_type == "face":
    dst_vid_dir = os.path.join(args.root_dir, "grid_video_face_224x224")
    dst_txt_dir = os.path.join(args.root_dir, "grid_text_face_224x224")
    csv_suffix = "_face_224x224"
else:
    dst_vid_dir = os.path.join(args.root_dir, "grid_video")
    dst_txt_dir = os.path.join(args.root_dir, "grid_text")
    csv_suffix = ""

labels_dir = os.path.join(args.root_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)

print(f"Output directories:")
print(f"  Videos: {dst_vid_dir}")
print(f"  Text: {dst_txt_dir}")
print(f"  Labels: {labels_dir}")

# Initialize video dataloader with RetinaFace
if args.detector == "retinaface":
    from detectors.retinaface.detector import LandmarksDetector
    from detectors.retinaface.video_process import VideoProcess
    
    landmarks_detector = LandmarksDetector(device="cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get the absolute path to the mean face file
    script_dir = Path(__file__).parent
    mean_face_path = script_dir / "detectors" / "retinaface" / "20words_mean_face.npy"
    
    if not mean_face_path.exists():
        print(f"❌ Error: Mean face file not found at {mean_face_path}")
        exit(1)
    
    if args.crop_type == "face":
        video_process = VideoProcess(
            mean_face_path=str(mean_face_path),
            crop_width=224,
            crop_height=224,
            start_idx=48,
            stop_idx=68,
            window_margin=12,
            convert_gray=False,  # Keep RGB for video output
        )
    else:  # lips
        video_process = VideoProcess(
            mean_face_path=str(mean_face_path),
            crop_width=96,
            crop_height=96,
            start_idx=48,
            stop_idx=68,
            window_margin=12,
            convert_gray=False,  # Keep RGB for video output
        )
    
    class CustomAVSRDataLoader:
        def __init__(self, modality):
            self.modality = modality
            self.landmarks_detector = landmarks_detector
            self.video_process = video_process
        
        def load_data(self, data_filename):
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
                landmarks = self.landmarks_detector(video)
                video = self.video_process(video, landmarks)
                if video is None:
                    raise TypeError("video cannot be None")
                return torch.tensor(video)
    
    vid_dataloader = CustomAVSRDataLoader(modality="video")
    aud_dataloader = CustomAVSRDataLoader(modality="audio")
else:
    vid_dataloader = AVSRDataLoader(
        modality="video", detector=args.detector, convert_gray=False
    )
    aud_dataloader = AVSRDataLoader(modality="audio")

def parse_grid_utterance(utterance_code):
    """
    Parse GRID utterance code to text.
    Format: {command}{color}{preposition}{letter}{digit}{adverb}
    Example: bbat9p -> bin blue at t nine please
    """
    # GRID vocabulary
    commands = {'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'}
    colors = {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'}
    prepositions = {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'}
    letters = {
        'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g',
        'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
        'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u',
        'v': 'v', 'x': 'x', 'y': 'y', 'z': 'z'
    }
    digits = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        'z': 'zero'
    }
    adverbs = {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}
    
    if len(utterance_code) < 6:
        return None
    
    try:
        command = commands.get(utterance_code[0], '')
        color = colors.get(utterance_code[1], '')
        preposition = prepositions.get(utterance_code[2], '')
        letter = letters.get(utterance_code[3], '')
        digit = digits.get(utterance_code[4], '')
        adverb = adverbs.get(utterance_code[5], '')
        
        transcript = f"{command} {color} {preposition} {letter} {digit} {adverb}"
        return transcript.strip()
    except:
        return None

def find_video_files(data_dir):
    """Find all video files organized by speaker"""
    data_path = Path(data_dir)
    
    # Find all speaker directories (s1, s2, ..., s34, excluding s21)
    speaker_dirs = sorted([d for d in data_path.iterdir() 
                          if d.is_dir() and d.name.startswith('s') and d.name[1:].isdigit()])
    
    video_files = {}
    for speaker_dir in speaker_dirs:
        speaker = speaker_dir.name
        # Videos are in speaker_dir/speaker/*.mpg
        video_subdir = speaker_dir / speaker
        if not video_subdir.exists():
            print(f"Warning: Video subdirectory not found for {speaker}")
            continue
        
        # Find all mpg files
        videos = list(video_subdir.glob("*.mpg"))
        video_files[speaker] = sorted(videos)
        print(f"Found {len(videos)} videos for speaker {speaker}")
    
    return video_files

# Find all video files
print(f"\nScanning GRID dataset at: {args.data_dir}")
video_files_by_speaker = find_video_files(args.data_dir)

# Flatten to single list for processing
all_filenames = []
for speaker, videos in video_files_by_speaker.items():
    for video_path in videos:
        all_filenames.append((speaker, str(video_path)))

if not all_filenames:
    print("❌ Error: No video files found")
    exit(1)

print(f"\nTotal videos to process: {len(all_filenames)}")

# Split work for parallel processing
unit = math.ceil(len(all_filenames) / args.groups)
files_to_process = all_filenames[args.job_index * unit : (args.job_index + 1) * unit]

print(f"Processing {len(files_to_process)} files (job {args.job_index + 1}/{args.groups})")

# Initialize CSV data collection for all speakers + combined
csv_data_all = []
csv_data_by_speaker = {}

processed_count = 0
skipped_count = 0

# Path to alignments
alignments_dir = Path(args.data_dir) / "alignments" / "alignments"

for speaker, vid_filename in tqdm(files_to_process, desc="Processing videos"):
    vid_path = Path(vid_filename)
    video_id = vid_path.stem  # e.g., bbaf2n
    
    # Parse transcript from video ID (GRID utterance code)
    transcript = parse_grid_utterance(video_id)
    if not transcript:
        skipped_count += 1
        continue
    
    try:
        video_data = vid_dataloader.load_data(vid_filename)
        audio_data = aud_dataloader.load_data(vid_filename)
    except (UnboundLocalError, TypeError, OverflowError, AssertionError, Exception) as e:
        skipped_count += 1
        continue
    
    if video_data is None:
        skipped_count += 1
        continue
    
    # Create output filename: speaker/video_id.mp4
    dst_vid_filename = os.path.join(dst_vid_dir, speaker, f"{video_id}.mp4")
    dst_aud_filename = os.path.join(dst_vid_dir, speaker, f"{video_id}.wav")
    dst_txt_filename = os.path.join(dst_txt_dir, speaker, f"{video_id}.txt")
    
    # Use the whole video (GRID videos are short single utterances)
    video_length = len(video_data)
    audio_length = audio_data.size(1)
    
    # Basic quality check
    if video_length < 5 or audio_length < 1000:
        skipped_count += 1
        continue
    
    # Save video, audio, and text
    save_vid_aud_txt(
        dst_vid_filename,
        dst_aud_filename,
        dst_txt_filename,
        video_data,
        audio_data,
        transcript,
    )
    
    # Add to CSV data
    csv_entry = {
        'speaker': speaker,
        'video_id': video_id,
        'video_path': dst_vid_filename,
        'audio_path': dst_aud_filename,
        'text_path': dst_txt_filename,
        'transcript': transcript,
        'num_frames': video_length,
        'audio_length': audio_length,
        'detector': args.detector,
        'crop_type': args.crop_type,
    }
    
    csv_data_all.append(csv_entry)
    
    if speaker not in csv_data_by_speaker:
        csv_data_by_speaker[speaker] = []
    csv_data_by_speaker[speaker].append(csv_entry)
    
    processed_count += 1

print(f"\n✅ Processing complete!")
print(f"   Processed: {processed_count} videos")
print(f"   Skipped: {skipped_count} videos")

# Save CSV files
if csv_data_all:
    # Combined CSV
    df_all = pd.DataFrame(csv_data_all)
    csv_all_path = os.path.join(labels_dir, f"grid_all{csv_suffix}.csv")
    df_all.to_csv(csv_all_path, index=False)
    print(f"\n📊 Saved combined CSV: {csv_all_path} ({len(df_all)} entries)")
    
    # Individual speaker CSVs
    for speaker, data in csv_data_by_speaker.items():
        if data:
            df_speaker = pd.DataFrame(data)
            csv_speaker_path = os.path.join(labels_dir, f"grid_{speaker}{csv_suffix}.csv")
            df_speaker.to_csv(csv_speaker_path, index=False)
            print(f"   Saved {speaker} CSV: {csv_speaker_path} ({len(df_speaker)} entries)")

print("\n🎉 GRID preprocessing complete!")
