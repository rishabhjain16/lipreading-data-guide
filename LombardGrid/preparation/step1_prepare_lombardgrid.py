#!/usr/bin/env python3
"""
Lombard GRID Dataset Preprocessing Pipeline (RetinaFace)
========================================================

Preprocessing script for Lombard GRID - GRID corpus recorded in noisy conditions.
Uses RetinaFace detector for face detection and lip/face cropping.

Usage:
    python step1_prepare_lombardgrid.py \
        --data-dir /media/rishabhjain/SSD/lombardgrid \
        --root-dir /path/to/output \
        --crop-type lips \
        --view front

Note: Lombard GRID has 54 speakers with 3 conditions (normal, lombard, plain noise).
"""

import argparse
import json
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
parser = argparse.ArgumentParser(description="Lombard GRID Preprocessing with RetinaFace")
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="Directory where Lombard GRID dataset is stored",
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
    "--view",
    type=str,
    default="front",
    choices=["front", "side"],
    help="Camera view to process (default: front)",
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
    dst_vid_dir = os.path.join(args.root_dir, "lombardgrid_video_face_224x224")
    dst_txt_dir = os.path.join(args.root_dir, "lombardgrid_text_face_224x224")
    csv_suffix = "_face_224x224"
else:
    dst_vid_dir = os.path.join(args.root_dir, "lombardgrid_video")
    dst_txt_dir = os.path.join(args.root_dir, "lombardgrid_text")
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
            convert_gray=False,
        )
    else:  # lips
        video_process = VideoProcess(
            mean_face_path=str(mean_face_path),
            crop_width=96,
            crop_height=96,
            start_idx=48,
            stop_idx=68,
            window_margin=12,
            convert_gray=False,
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

def get_corrected_video_id(filename):
    """
    Get corrected video ID for files with WRONG annotations.
    s46_p_gbszs_WRONG_sgbszs.mov -> s46_p_sgbszs
    s27_l_bah2s_WRONG_#sil.mov -> s27_l_bah2s_WRONG_#sil (unchanged, will be ignored)
    """
    filename_no_ext = filename.replace('.mov', '').replace('.wav', '')
    parts = filename_no_ext.split('_')
    
    # Check if this is a WRONG annotation file
    if 'WRONG' in parts:
        wrong_index = parts.index('WRONG')
        if wrong_index + 1 < len(parts):
            correct_utterance = parts[wrong_index + 1]
            # Check if it's a silence marker
            if 'sil' in correct_utterance.lower() or '#sil' in correct_utterance:
                return filename_no_ext  # Keep original name for silence files
            else:
                # Rebuild with corrected utterance: s{speaker}_{condition}_{corrected_utterance}
                speaker = parts[0]
                condition = parts[1]
                return f"{speaker}_{condition}_{correct_utterance}"
    
    # Normal case
    return filename_no_ext

def extract_transcript_from_filename(filename):
    """
    Extract transcript from Lombard GRID filename.
    Format: s{speaker}_{condition}_{utterance}.mov
    Example: s10_l_bbat9p -> bin blue at t nine please
    
    Handle corrections for mislabeled files:
    - s46_p_gbszs_WRONG_sgbszs -> use sgbszs instead of gbszs
    - s27_l_bah2s_WRONG_#sil -> return None (silence, should be skipped)
    """
    parts = filename.split('_')
    
    # Check if this is a WRONG annotation file
    if 'WRONG' in parts:
        wrong_index = parts.index('WRONG')
        if wrong_index + 1 < len(parts):
            # Get the correct utterance after WRONG
            correct_utterance = parts[wrong_index + 1].replace('.mov', '').replace('.wav', '')
            # Check if it's a silence marker
            if 'sil' in correct_utterance.lower() or '#sil' in correct_utterance:
                return None  # Skip silence files
            else:
                return parse_grid_utterance(correct_utterance)
    
    # Normal case: s{speaker}_{condition}_{utterance}
    if len(parts) >= 3:
        utterance_code = parts[2].replace('.mov', '').replace('.wav', '')
        return parse_grid_utterance(utterance_code)
    
    return None

def find_video_files(data_dir, view):
    """Find all video files for specified view"""
    data_path = Path(data_dir)
    video_dir = data_path / view
    
    if not video_dir.exists():
        print(f"❌ Error: Video directory not found: {video_dir}")
        return {}
    
    # Find all video files
    video_files = list(video_dir.glob("*.mov"))
    
    # Group by speaker
    video_files_by_speaker = {}
    for video_path in video_files:
        # Extract speaker from filename: s{speaker}_{condition}_{utterance}
        speaker = video_path.stem.split('_')[0]
        if speaker not in video_files_by_speaker:
            video_files_by_speaker[speaker] = []
        video_files_by_speaker[speaker].append(video_path)
    
    for speaker in sorted(video_files_by_speaker.keys()):
        print(f"Found {len(video_files_by_speaker[speaker])} videos for speaker {speaker}")
    
    return video_files_by_speaker

# Find all video files
print(f"\nScanning Lombard GRID dataset at: {args.data_dir}")
print(f"View: {args.view}")
video_files_by_speaker = find_video_files(args.data_dir, args.view)

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

# Initialize CSV data collection
csv_data_all = []
csv_data_by_speaker = {}

processed_count = 0
skipped_count = 0
ignored_files = []  # Track files that were intentionally ignored

# Paths to audio and alignment directories
audio_dir = Path(args.data_dir) / "audio"
alignment_dir = Path(args.data_dir) / "alignment"

for speaker, vid_filename in tqdm(files_to_process, desc="Processing videos"):
    vid_path = Path(vid_filename)
    original_video_id = vid_path.stem  # e.g., s10_l_bbat9p or s46_p_gbszs_WRONG_sgbszs
    corrected_video_id = get_corrected_video_id(vid_path.name)  # e.g., s46_p_sgbszs
    
    # Find corresponding audio file (using original name)
    audio_path = audio_dir / f"{original_video_id}.wav"
    if not audio_path.exists():
        print(f"❌ Skipping {original_video_id}: Audio file not found")
        skipped_count += 1
        continue
    
    # Extract transcript from filename
    transcript = extract_transcript_from_filename(vid_path.name)
    if not transcript:
        # Check if this is an intentionally ignored file (silence)
        filename = vid_path.name
        if 'WRONG' in filename and ('sil' in filename.lower() or '#sil' in filename):
            print(f"🔇 Ignoring {original_video_id}: Silence file")
            ignored_files.append(original_video_id)
        else:
            print(f"❌ Skipping {original_video_id}: Failed to extract transcript")
        skipped_count += 1
        continue
    
    try:
        video_data = vid_dataloader.load_data(vid_filename)
        audio_data = aud_dataloader.load_data(str(audio_path))
    except (UnboundLocalError, TypeError, OverflowError, AssertionError, Exception) as e:
        print(f"❌ Skipping {original_video_id}: Error loading data - {type(e).__name__}")
        skipped_count += 1
        continue
    
    if video_data is None:
        print(f"❌ Skipping {original_video_id}: Video processing returned None")
        skipped_count += 1
        continue
    
    # Create output filename using corrected video ID: view/speaker/corrected_video_id.mp4
    dst_vid_filename = os.path.join(dst_vid_dir, args.view, speaker, f"{corrected_video_id}.mp4")
    dst_aud_filename = os.path.join(dst_vid_dir, args.view, speaker, f"{corrected_video_id}.wav")
    dst_txt_filename = os.path.join(dst_txt_dir, args.view, speaker, f"{corrected_video_id}.txt")
    
    # Log correction if name was changed
    if corrected_video_id != original_video_id:
        print(f"  🔧 Correcting filename: {original_video_id} -> {corrected_video_id}")
    
    # Use the whole video
    video_length = len(video_data)
    audio_length = audio_data.size(1)
    
    # Basic quality check
    if video_length < 5 or audio_length < 1000:
        if video_length < 5:
            print(f"❌ Skipping {original_video_id}: Video too short ({video_length} frames)")
        if audio_length < 1000:
            print(f"❌ Skipping {original_video_id}: Audio too short ({audio_length} samples)")
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
        'video_id': corrected_video_id,
        'condition': corrected_video_id.split('_')[1],  # l, p, or 1
        'video_path': dst_vid_filename,
        'audio_path': dst_aud_filename,
        'text_path': dst_txt_filename,
        'transcript': transcript,
        'num_frames': video_length,
        'audio_length': audio_length,
        'detector': args.detector,
        'crop_type': args.crop_type,
        'view': args.view,
        'original_filename': original_video_id if corrected_video_id != original_video_id else None,
    }
    
    csv_data_all.append(csv_entry)
    
    if speaker not in csv_data_by_speaker:
        csv_data_by_speaker[speaker] = []
    csv_data_by_speaker[speaker].append(csv_entry)
    
    # Log successful processing for corrected files
    if corrected_video_id != original_video_id:
        print(f"  ✅ Successfully processed corrected file: {corrected_video_id}")
    
    processed_count += 1

print(f"\n✅ Processing complete!")
print(f"   Processed: {processed_count} videos")
print(f"   Skipped: {skipped_count} videos")

# Report ignored files
if ignored_files:
    print(f"\n🔇 Ignored files (silence/empty content):")
    for ignored_file in ignored_files:
        print(f"   • {ignored_file}")
    print(f"   Total ignored: {len(ignored_files)}")
    print(f"   Actual failures: {skipped_count - len(ignored_files)}")

# Save CSV files
if csv_data_all:
    # Only save view-level CSV (front or side)
    df_all = pd.DataFrame(csv_data_all)
    csv_all_path = os.path.join(labels_dir, f"lombardgrid_{args.view}{csv_suffix}.csv")
    df_all.to_csv(csv_all_path, index=False)
    print(f"\n📊 Saved CSV: {csv_all_path} ({len(df_all)} entries)")

print("\n🎉 Lombard GRID preprocessing complete!")
