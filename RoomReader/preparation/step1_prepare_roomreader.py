#!/usr/bin/env python3

import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings("ignore")

def clean_transcript(text):
    """Remove disfluency markers and non-English annotations from RoomReader transcripts"""
    if not text or text.strip() == "":
        return ""
    
    # Remove $ and $# markers with surrounding spaces (RoomReader specific)
    text = re.sub(r'\s*\$[#]?\s*', ' ', text)
    # Remove # markers with surrounding spaces (RoomReader specific)
    text = re.sub(r'\s*#\s*', ' ', text)
    
    # Remove common punctuation and non-English annotations (keeping apostrophes)
    # Remove: : , . ! ? ; - " ( ) [ ] { } @ % < >
    text = re.sub(r'[:,.!?;\-"()\[\]{}<>@%]', '', text)
    # Remove double dashes and ellipsis
    text = re.sub(r'--+|\.\.\.+', ' ', text)
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_roomreader_transcript(transcript_file):
    """Parse RoomReader transcript file and extract Check tier utterances"""
    utterances = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            
            # Filter for Check tiers with actual content
            if (len(parts) >= 12 and 
                parts[0].endswith('_Check') and 
                parts[11].strip() != ""):
                
                # Extract participant name from tier_id
                tier_id = parts[0]  # e.g., "P011_Christian_Check"
                participant = tier_id.replace('_Check', '')  # "P011_Christian"
                
                utterance = {
                    'participant': participant,
                    'start_time': float(parts[3]),
                    'end_time': float(parts[6]), 
                    'duration': float(parts[9]),
                    'text': clean_transcript(parts[11])
                }
                
                if utterance['text']:
                    utterances.append(utterance)
    
    return utterances

def create_speaker_mapping(participants):
    """Create mapping from participant names to anonymous speaker IDs"""
    unique_participants = sorted(set(participants))
    return {p: f"spk{i}" for i, p in enumerate(unique_participants)}

def process_roomreader_session(transcript_file, video_dir, output_path, text_path, landmarks_detector, video_process, args, output_size, combined_av_path=None):
    """Process a single RoomReader session with RetinaFace detection"""
    
    session_name = transcript_file.stem  # S01, S02, etc.
    print(f"Processing {session_name} with transcript {transcript_file.name}")
    
    # Parse transcript
    utterances = parse_roomreader_transcript(transcript_file)
    if not utterances:
        print(f"Warning: No valid utterances found in {transcript_file}")
        return []
    
    # Create speaker mapping for this session
    participants = [u['participant'] for u in utterances]
    speaker_mapping = create_speaker_mapping(participants)
    
    # Process each video
    video_files = list(video_dir.glob("*.mp4"))
    processed_data = []
    
    for video_file in video_files:
        # Extract participant name from video filename
        # Individual mode: S01_P026_Oliver.mp4
        # Conversational mode: S01_P026_Oliver_all.mp4
        video_name = video_file.stem  # S01_P026_Oliver or S01_P026_Oliver_all
        
        # Remove _all suffix for conversational mode
        if video_name.endswith('_all'):
            video_name = video_name[:-4]  # Remove '_all'
            
        if video_name.startswith(f"{session_name}_"):
            participant_name = video_name[len(session_name)+1:]  # P026_Oliver
        else:
            participant_name = video_name
            
        if participant_name not in speaker_mapping:
            continue
            
        print(f"  Processing video: {participant_name}")
        speaker_id = speaker_mapping[participant_name]
        
        # Find utterances for this participant
        participant_utterances = [u for u in utterances if u['participant'] == participant_name]
        
        for i, utterance in enumerate(participant_utterances):
            try:
                # Step 1: Extract video/audio segments based on timestamps first
                import torchvision
                import torchaudio
                
                # Load video segment directly using timestamps (memory efficient)
                video_segment, audio_segment, info = torchvision.io.read_video(
                    str(video_file),
                    start_pts=utterance['start_time'],
                    end_pts=utterance['end_time'],
                    pts_unit='sec'
                )
                
                if video_segment.size(0) == 0:
                    print(f"    Skipping utterance {i}: Empty video segment")
                    continue
                
                # Check audio segment size
                if audio_segment.size(0) == 0:
                    print(f"    Skipping utterance {i}: Empty audio segment")
                    continue
                
                print(f"    Audio segment shape: {audio_segment.shape}, Video segment shape: {video_segment.shape}")
                
                # Convert to numpy for RetinaFace processing
                video_np = video_segment.numpy()
                
                # Step 2: Apply RetinaFace detection and cropping on the small segment
                landmarks = landmarks_detector(video_np)
                processed_video = video_process(video_np, landmarks)
                
                if processed_video is None:
                    print(f"    Skipping utterance {i}: Face detection failed")
                    continue
                
                # Convert back to tensor
                video_data = torch.tensor(processed_video)
                
                # Process audio segment
                # audio_segment is already in (C, T) format from torchvision.io.read_video
                audio_data = audio_segment
                
                # Convert to mono if stereo (audio_data is in (C, T) format)
                if audio_data.size(0) > 1:
                    # Take mean across channels (dim 0) to convert stereo to mono
                    audio_data = torch.mean(audio_data, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                if 'audio_fps' in info and info['audio_fps'] != 16000:
                    audio_data = torchaudio.functional.resample(
                        audio_data, info['audio_fps'], 16000
                    )
                
                # Create unique ID for this utterance
                utterance_id = f"{session_name}_{speaker_id}_{i:03d}"
                
                # Create output filenames - video/audio in one folder, text in separate folder
                dst_vid_filename = output_path / f"{utterance_id}.mp4"
                dst_aud_filename = output_path / f"{utterance_id}.wav"
                dst_txt_filename = text_path / f"{utterance_id}.txt"
                
                # Save using the utils function
                from utils import save_vid_aud_txt
                save_vid_aud_txt(
                    str(dst_vid_filename),
                    str(dst_aud_filename),
                    str(dst_txt_filename),
                    video_data,
                    audio_data,
                    utterance['text'],
                    video_fps=info['video_fps'],  # Use original video FPS
                    audio_sample_rate=16000,
                )

                # Save combined AV file if requested (for sanity checking)
                if combined_av_path is not None:
                    combined_av_session_path = combined_av_path / session_name
                    combined_av_session_path.mkdir(parents=True, exist_ok=True)

                    dst_combined_filename = combined_av_session_path / f"{utterance_id}_av.mp4"
                    dst_combined_txt_filename = combined_av_session_path / f"{utterance_id}.txt"

                    torchvision.io.write_video(
                        str(dst_combined_filename),
                        video_data,
                        fps=info.get('video_fps', 30),
                        audio_array=audio_data,
                        audio_fps=16000,
                        audio_codec='aac'
                    )

                    # Save transcript alongside AV sanity-check clip
                    with open(dst_combined_txt_filename, 'w', encoding='utf-8') as f:
                        f.write(utterance['text'])
                
                # Add to processed data for CSV (LRS format)
                rel_video_path = f"{args.video_mode}/{session_name}/{utterance_id}.mp4"
                processed_data.append([
                    speaker_id,                    # speaker_id
                    rel_video_path,                # video_path (relative)
                    utterance['text'],             # transcript
                    len(utterance['text'].split()),# word_count
                    utterance_id,                  # unique_id
                    f"{session_name}_{i:03d}",     # transcript_id
                    args.detector,                 # detector
                    args.crop_type,                # crop_type
                    f"{output_size}x{output_size}" # resolution
                ])
                
                print(f"    ✅ Processed utterance {i}: {utterance['duration']:.1f}s - '{utterance['text']}'")
                
            except Exception as e:
                print(f"    ❌ Error processing {participant_name} utterance {i}: {e}")
                continue
    
    return processed_data

def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    # Create output directories with crop type and size suffixes like TCD-TIMIT
    crop_suffix = f"_{args.crop_type}"
    output_size = 96 if args.crop_type == "lips" else 224
    detector_suffix = f"_{args.detector}" if args.detector != "retinaface" else ""
    
    final_output_path = output_path / f"roomreader_video{crop_suffix}{detector_suffix}" / args.video_mode
    final_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create separate text directory like other datasets
    text_output_path = output_path / f"roomreader_text{crop_suffix}{detector_suffix}" / args.video_mode
    text_output_path.mkdir(parents=True, exist_ok=True)

    # Create combined AV directory if requested
    combined_av_path = None
    if args.save_combined_av:
        combined_av_path = output_path / f"roomreader_av{crop_suffix}{detector_suffix}" / args.video_mode
        combined_av_path.mkdir(parents=True, exist_ok=True)
        print(f"Combined AV files will be saved to: {combined_av_path}")
    
    # Initialize RetinaFace detector at module level like TCD-TIMIT
    if args.detector == "retinaface":
        from detectors.retinaface.detector import LandmarksDetector
        from detectors.retinaface.video_process import VideoProcess
        
        landmarks_detector = LandmarksDetector(device="cuda:0")
        
        # Configure crop settings based on crop_type
        if args.crop_type == "lips":
            start_idx, stop_idx = 48, 68  # Mouth landmarks
        elif args.crop_type == "face":
            start_idx, stop_idx = 17, 68  # Face landmarks (eyebrows to chin)
        
        video_process = VideoProcess(
            crop_width=output_size,
            crop_height=output_size,
            start_idx=start_idx,
            stop_idx=stop_idx,
            convert_gray=False
        )
    else:
        # Fallback for other detectors
        print(f"Detector {args.detector} not implemented")
        return
    
    # RoomReader has separate folders for transcripts and videos
    transcript_path = data_path / "annotations" / "transcriptions_txt"
    
    # Choose video folder based on mode
    if args.video_mode == "individual":
        video_path = data_path / "video" / "individual_participants" / "individual_participants_individual_audio"
    else:  # conversational mode
        video_path = data_path / "video" / "individual_participants" / "individual_participants_audio_all"
    
    if not transcript_path.exists():
        print(f"Error: Transcript path not found: {transcript_path}")
        return
    
    if not video_path.exists():
        print(f"Error: Video path not found: {video_path}")
        return
    
    # Find all transcript files
    transcript_files = list(transcript_path.glob("S*.txt"))
    
    print(f"Found {len(transcript_files)} sessions to process")
    print(f"Video mode: {args.video_mode}")
    print(f"Crop type: {args.crop_type} ({output_size}x{output_size})")
    print(f"Detector: {args.detector}")
    print(f"Combined AV: {'Yes (for sanity checking)' if args.save_combined_av else 'No'}")
    
    # Collect all processed data for CSV
    all_processed_data = []
    
    for transcript_file in tqdm(transcript_files, desc="Processing sessions"):
        try:
            session_name = transcript_file.stem  # S01, S02, etc.
            session_video_dir = video_path / session_name
            
            if not session_video_dir.exists():
                print(f"Warning: Video directory not found for {session_name}")
                continue
            
            # Create session output directories
            session_output_path = final_output_path / session_name
            session_output_path.mkdir(parents=True, exist_ok=True)
            
            session_text_path = text_output_path / session_name
            session_text_path.mkdir(parents=True, exist_ok=True)
                
            processed_data = process_roomreader_session(
                transcript_file, session_video_dir, session_output_path, session_text_path,
                landmarks_detector, video_process, args, output_size, combined_av_path
            )
            all_processed_data.extend(processed_data)
            
        except Exception as e:
            print(f"Error processing {session_name}: {e}")
            continue
    
    # Save CSV summary
    if all_processed_data:
        labels_dir = output_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Follow TCD-TIMIT/LRS naming pattern
        crop_suffix = f"_{args.crop_type}"
        detector_suffix = f"_{args.detector}" if args.detector != "retinaface" else ""
        csv_filename = f"roomreader_{args.video_mode}{crop_suffix}{detector_suffix}.csv"
        csv_path = labels_dir / csv_filename
        
        # Create DataFrame with LRS-compatible structure
        df = pd.DataFrame(all_processed_data, columns=[
            'speaker_id', 'video_path', 'transcript', 'word_count', 
            'unique_id', 'transcript_id', 'detector', 'crop_type', 'resolution'
        ])
        
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Saved CSV: {csv_path}")
        print(f"📊 Total: {len(df)} samples ({output_size}x{output_size}, {args.crop_type}, {args.detector})")
    
    print(f"✅ Processing complete! Output saved to: {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare RoomReader dataset with RetinaFace')
    parser.add_argument('--data-path', required=True, help='Path to RoomReader data')
    parser.add_argument('--output-path', required=True, help='Output directory')
    parser.add_argument('--video-mode', choices=['individual', 'conversational'], default='individual',
                        help='Video mode: individual (clean audio) or conversational (noisy audio)')
    parser.add_argument('--crop-type', choices=['lips', 'face'], default='lips',
                        help='Crop type: lips (96x96, mouth region) or face (224x224, full face)')
    parser.add_argument('--detector', default='retinaface', help='Face detector to use')
    parser.add_argument('--save-combined-av', action='store_true',
                        help='Save combined audio+video files in _av folder for sanity checking')
    args = parser.parse_args()
    
    main(args)
