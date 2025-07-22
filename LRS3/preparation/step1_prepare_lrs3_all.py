#!/usr/bin/env python3
"""
Complete LRS3 Preprocessing with Flexible Cropping Options
One script to handle everything: lips, face, or full video cropping
"""

import argparse
import glob
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Complete LRS3 Preprocessing with Flexible Cropping")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory of original dataset")
    parser.add_argument("--detector", type=str, default="retinaface", help="Face detector (retinaface/mediapipe)")
    parser.add_argument("--landmarks-dir", type=str, default=None, help="Directory of landmarks")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of preprocessed dataset")
    parser.add_argument("--subset", type=str, required=True, help="Subset (train/val/test)")
    parser.add_argument("--dataset", type=str, default="lrs3", help="Dataset name (lrs3)")
    parser.add_argument("--seg-duration", type=int, default=16, help="Segment duration in seconds")
    parser.add_argument(
        "--crop-type", 
        type=str, 
        default="lips", 
        choices=["lips", "face", "full"],
        help="Cropping type: lips (96x96 mouth region), face (128x128 face region using landmarks), full (original size)"
    )
    parser.add_argument("--combine-av", action="store_true", help="Combine audio and video")
    parser.add_argument("--groups", type=int, default=1, help="Number of parallel groups")
    parser.add_argument("--job-index", type=int, default=0, help="Job index for parallel processing")
    
    args = parser.parse_args()
    
    # Set crop parameters based on type
    if args.crop_type == "lips":
        crop_width, crop_height = 96, 96
    elif args.crop_type == "face":
        crop_width, crop_height = 128, 128  # Smaller than original 224x224
    else:  # full
        crop_width, crop_height = None, None
    
    print(f"🎥 Crop type: {args.crop_type}")
    if crop_width:
        print(f"📐 Crop size: {crop_width}x{crop_height}")
    
    # Initialize
    seg_duration = args.seg_duration
    dataset = args.dataset
    text_transform = TextTransform()
    args.data_dir = os.path.normpath(args.data_dir)
    
    # Create data loaders
    if args.crop_type == "full":
        # For full, we need to bypass the normal video processing entirely
        # Create a custom loader that processes raw video
        class CustomVideoLoader:
            def __init__(self, detector, crop_type, crop_size=(128, 128)):
                self.detector = detector
                self.crop_type = crop_type
                self.crop_size = crop_size
                
            def load_data(self, filename, landmarks=None):
                import cv2
                import torch
                import numpy as np
                
                # Load video directly with OpenCV
                cap = cv2.VideoCapture(filename)
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Keep original size for full video and RGB format
                    # Don't convert to grayscale - keep RGB for torchvision compatibility
                    frames.append(frame)
                
                cap.release()
                
                if len(frames) == 0:
                    return None
                    
                return torch.tensor(np.array(frames))
        
        vid_dataloader = CustomVideoLoader(args.detector, args.crop_type, (crop_width, crop_height))
    elif args.crop_type == "face":
        # For face, use normal data loader but modify it to crop face region instead of lips
        vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, convert_gray=False)
        
        # Modify the video processor to crop face instead of lips
        # For face landmarks, use landmarks 0-16 (jaw line) instead of just mouth region (48-67)
        # This gives us the full face contour
        vid_dataloader.video_process.start_idx = 0   # Start from jaw landmarks
        vid_dataloader.video_process.stop_idx = 68   # Include all face landmarks (0-67)
        
        # Set face crop size (larger than lips)
        vid_dataloader.video_process.crop_width = crop_width
        vid_dataloader.video_process.crop_height = crop_height
    else:
        # For lips, use the normal data loader
        vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, convert_gray=False)
    aud_dataloader = AVSRDataLoader(modality="audio")
    
    # Setup output directories
    crop_suffix = f"_{args.crop_type}" if args.crop_type != "lips" else ""
    dst_vid_dir = os.path.join(args.root_dir, dataset, f"{dataset}_video_seg{seg_duration}s{crop_suffix}")
    dst_txt_dir = os.path.join(args.root_dir, dataset, f"{dataset}_text_seg{seg_duration}s")
    
    # Label file
    label_filename = os.path.join(
        args.root_dir, "labels",
        f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s{crop_suffix}.csv"
    )
    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    print(f"📁 Output video dir: {dst_vid_dir}")
    print(f"📄 Label file: {label_filename}")
    
    # Get file list for LRS3
    if args.subset == "test":
        filenames = glob.glob(os.path.join(args.data_dir, args.subset, "**", "*.mp4"), recursive=True)
    elif args.subset == "train":
        # For training, we need to exclude validation files to prevent data leakage
        filenames = glob.glob(os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True)
        filenames.extend(glob.glob(os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True))
        
        # Load validation IDs to exclude them from training
        # Try repo location first, then data directory
        possible_valid_files = [
            os.path.join(os.path.dirname(__file__), "..", "lrs3-valid.id"),  # LRS3 repo folder
            os.path.join(os.getcwd(), "lrs3-valid.id"),  # Current working directory
            os.path.join(args.data_dir, "lrs3-valid.id"),  # Data directory
            os.path.join(args.data_dir, "valid.id"),
            os.path.join(args.data_dir, "lrs3_valid.id")
        ]
        
        valid_id_file = None
        for vf in possible_valid_files:
            if os.path.exists(vf):
                valid_id_file = vf
                break
        
        if valid_id_file:
            print(f"📋 Loading validation IDs to exclude from training: {valid_id_file}")
            with open(valid_id_file, 'r') as f:
                valid_ids_raw = [line.strip() for line in f.readlines()]
            
            # Remove 'trainval/' prefix from validation IDs to match file paths
            valid_ids = set()
            for vid in valid_ids_raw:
                if vid.startswith('trainval/'):
                    valid_ids.add(vid[9:])  # Remove 'trainval/' prefix
                else:
                    valid_ids.add(vid)
            
            # Filter out validation files from training set
            original_count = len(filenames)
            filtered_filenames = []
            
            for file_path in filenames:
                # Check if this is a trainval file (pretrain files are always included)
                if "trainval" in file_path:
                    relative_path = os.path.relpath(file_path, os.path.join(args.data_dir, "trainval"))
                    file_id = relative_path.replace('.mp4', '')
                    
                    # Only include if NOT in validation set
                    if file_id not in valid_ids:
                        filtered_filenames.append(file_path)
                else:
                    # Include all pretrain files
                    filtered_filenames.append(file_path)
            
            filenames = filtered_filenames
            excluded_count = original_count - len(filenames)
            print(f"📊 Excluded {excluded_count} validation files from training set")
            print(f"📊 Training will use {len(filenames)} files")
        else:
            print(f"⚠️  Warning: Validation ID file not found!")
            print(f"   Training set may include validation files (data leakage risk)")
            print(f"   Download from: https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/data/lrs3-valid.id")
            
    elif args.subset == "val":
        # For LRS3, validation uses ID files (lrs3-valid.id) instead of path-based txt files
        # Try repo location first, then data directory
        possible_valid_files = [
            os.path.join(os.path.dirname(__file__), "..", "lrs3-valid.id"),  # LRS3 repo folder
            os.path.join(os.getcwd(), "lrs3-valid.id"),  # Current working directory
            os.path.join(args.data_dir, "lrs3-valid.id"),  # Data directory
            os.path.join(args.data_dir, "valid.id"), 
            os.path.join(args.data_dir, "lrs3_valid.id")
        ]
        
        valid_id_file = None
        for vf in possible_valid_files:
            if os.path.exists(vf):
                valid_id_file = vf
                break
        
        if valid_id_file:
            print(f"📋 Using validation ID file: {valid_id_file}")
            # Read validation IDs
            with open(valid_id_file, 'r') as f:
                valid_ids_raw = [line.strip() for line in f.readlines()]
            
            # Remove 'trainval/' prefix from validation IDs to match file paths
            valid_ids = set()
            for vid in valid_ids_raw:
                if vid.startswith('trainval/'):
                    valid_ids.add(vid[9:])  # Remove 'trainval/' prefix
                else:
                    valid_ids.add(vid)
            
            # Get all trainval files and filter by validation IDs
            all_trainval_files = glob.glob(os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True)
            filenames = []
            
            # Process all files to find matches
            for file_path in all_trainval_files:
                # Extract file ID from path (usually speaker_id/video_id format)
                relative_path = os.path.relpath(file_path, os.path.join(args.data_dir, "trainval"))
                file_id = relative_path.replace('.mp4', '')  # Remove extension
                
                # Check if this file ID is in validation set
                if file_id in valid_ids:
                    filenames.append(file_path)
            
            print(f"📊 Found {len(filenames)} validation files from {len(valid_ids)} IDs")
        else:
            print(f"❌ Error: Validation ID file not found!")
            print(f"   Looked for: lrs3-valid.id, valid.id, lrs3_valid.id in {args.data_dir}")
            print(f"   Please download from: https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/data/lrs3-valid.id")
            return
    
    # Handle parallel processing
    unit = math.ceil(len(filenames) / args.groups)
    filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]
    
    print(f"🎬 Processing {len(filenames)} files...")
    
    # Process files
    seg_vid_len = seg_duration * 25
    with open(label_filename, "w") as f:
        for data_filename in tqdm(filenames, desc=f"Processing {args.subset}"):
            # Load landmarks if available
            landmarks = None
            if args.landmarks_dir:
                landmarks_file = data_filename.replace(args.data_dir, args.landmarks_dir)[:-4] + ".pkl"
                if os.path.exists(landmarks_file):
                    landmarks = pickle.load(open(landmarks_file, "rb"))
            
            try:
                video_data = vid_dataloader.load_data(data_filename, landmarks)
                audio_data = aud_dataloader.load_data(data_filename)
            except Exception:
                continue
            
            if video_data is None or audio_data is None:
                continue
            
            # Get text content
            txt_file = data_filename[:-4] + ".txt"
            if not os.path.exists(txt_file):
                continue
            
            text_lines = open(txt_file).read().splitlines()
            if not text_lines:
                continue
            
            # Check if it's a test/trainval file (no segmentation needed) or pretrain (needs segmentation)
            path_parts = os.path.normpath(data_filename).split(os.sep)
            if "pretrain" in path_parts:
                # Process pretrain file with segmentation
                segments = split_file(txt_file, max_frames=seg_vid_len)
                for i, (content, start, end, duration) in enumerate(segments):
                    if len(segments) == 1:
                        trim_vid_data, trim_aud_data = video_data, audio_data
                    else:
                        start_idx, end_idx = int(start * 25), int(end * 25)
                        try:
                            trim_vid_data = video_data[start_idx:end_idx]
                            trim_aud_data = audio_data[:, start_idx * 640:end_idx * 640]
                        except Exception:
                            continue
                    
                    if trim_vid_data is None or trim_aud_data is None:
                        continue
                    if len(trim_vid_data) == 0 or trim_aud_data.size(1) == 0:
                        continue
                    
                    dst_vid_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
                    dst_aud_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
                    dst_txt_filename = f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}_{i:02d}.txt"
                    
                    # Save files
                    save_vid_aud_txt(
                        dst_vid_filename, dst_aud_filename, dst_txt_filename,
                        trim_vid_data, trim_aud_data, content,
                        video_fps=25, audio_sample_rate=16000
                    )
                    
                    # Combine AV if requested
                    if args.combine_av:
                        combine_audio_video(dst_vid_filename, dst_aud_filename)
                    
                    # Write to label file
                    basename = os.path.relpath(dst_vid_filename, start=os.path.join(args.root_dir, dataset))
                    token_ids = " ".join(str(t.item()) for t in text_transform.tokenize(content))
                    if token_ids:
                        f.write(f"{dataset},{basename},{trim_vid_data.shape[0]},{token_ids}\n")
                        
            else:
                # Process single file (test/trainval)
                text_line = " ".join(text_lines[0].split()[2:]) if len(text_lines[0].split()) > 2 else text_lines[0]
                content = text_line.replace("}", "").replace("{", "")
                
                dst_vid_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
                dst_aud_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.wav"
                dst_txt_filename = f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"
                
                # Save files
                save_vid_aud_txt(
                    dst_vid_filename, dst_aud_filename, dst_txt_filename,
                    video_data, audio_data, content,
                    video_fps=25, audio_sample_rate=16000
                )
                
                # Combine AV if requested
                if args.combine_av:
                    combine_audio_video(dst_vid_filename, dst_aud_filename)
                
                # Write to label file
                basename = os.path.relpath(dst_vid_filename, start=os.path.join(args.root_dir, dataset))
                token_ids = " ".join(str(t.item()) for t in text_transform.tokenize(content))
                f.write(f"{dataset},{basename},{video_data.shape[0]},{token_ids}\n")
    
    print(f"✅ Preprocessing completed!")
    print(f"📁 Output directory: {dst_vid_dir}")
    print(f"📄 Label file: {label_filename}")


def combine_audio_video(video_file, audio_file):
    """Combine audio and video files using ffmpeg"""
    try:
        in1 = ffmpeg.input(video_file)
        in2 = ffmpeg.input(audio_file)
        out = ffmpeg.output(
            in1["v"], in2["a"],
            video_file[:-4] + ".av.mp4",
            vcodec="copy", acodec="aac",
            strict="experimental", loglevel="panic"
        )
        out.run()
        shutil.move(video_file[:-4] + ".av.mp4", video_file)
        os.remove(audio_file)
    except Exception as e:
        print(f"Warning: Could not combine audio/video for {video_file}: {e}")


if __name__ == "__main__":
    main()
