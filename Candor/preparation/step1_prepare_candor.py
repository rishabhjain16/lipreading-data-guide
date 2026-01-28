#!/usr/bin/env python3
"""
Candor Dataset Preprocessing - Step 1: Video Processing with Face Detection

This script processes the Candor conversational dataset:
- Loads individual speaker videos and Speechmatics word-level transcripts
- Groups words into phrases (2-5 seconds) for optimal segment length
- Applies RetinaFace face detection and cropping
- Extracts synchronized audio segments
- Saves processed video, audio, and text files

Output format compatible with Auto-AVSR and AV-HuBERT training pipelines.
"""

import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
import torchvision
import warnings

warnings.filterwarnings("ignore")


def clean_transcript(text):
    """Clean transcript text by removing punctuation and normalizing"""
    if not text or text.strip() == "":
        return ""
    
    # Remove common punctuation (keeping apostrophes for contractions)
    text = re.sub(r'[:,.!?;\-"()\[\]{}<>@%]', '', text)
    # Remove double dashes and ellipsis
    text = re.sub(r'--+|\.\.\.+', ' ', text)
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_speechmatics_transcript(json_file):
    """Load Speechmatics JSON and extract word-level timestamps"""
    with open(json_file) as f:
        data = json.load(f)
    
    words = []
    for item in data.get("results", []):
        if item.get("type") == "word":
            alt = item["alternatives"][0]
            words.append({
                "word": alt["content"],
                "start": item["start_time"],
                "end": item["end_time"],
                "confidence": alt.get("confidence", 1.0)
            })
    
    return words


def group_words_into_phrases(words, min_duration=2.0, max_duration=5.0, max_gap=0.5):
    """
    Group words into phrases based on timing and punctuation.
    
    Args:
        words: List of word dictionaries with start/end times
        min_duration: Minimum phrase duration in seconds
        max_duration: Maximum phrase duration in seconds
        max_gap: Maximum gap between words to consider them in same phrase
    
    Returns:
        List of phrase dictionaries with start/end times and text
    """
    if not words:
        return []
    
    phrases = []
    current_phrase = []
    phrase_start = words[0]["start"]
    
    for i, word in enumerate(words):
        # Check if we should start a new phrase
        should_break = False
        
        if current_phrase:
            # Calculate current phrase duration
            phrase_duration = word["end"] - phrase_start
            
            # Calculate gap from previous word
            prev_word = words[i-1]
            gap = word["start"] - prev_word["end"]
            
            # Break conditions:
            # 1. Phrase is getting too long
            # 2. Large gap between words (silence/pause)
            # 3. Phrase already meets minimum and we hit a natural break
            if phrase_duration > max_duration:
                should_break = True
            elif gap > max_gap and phrase_duration >= min_duration:
                should_break = True
        
        if should_break:
            # Save current phrase
            phrase_text = " ".join([w["word"] for w in current_phrase])
            phrase_text = clean_transcript(phrase_text)
            
            if phrase_text:  # Only save non-empty phrases
                phrases.append({
                    "text": phrase_text,
                    "start": phrase_start,
                    "end": current_phrase[-1]["end"],
                    "duration": current_phrase[-1]["end"] - phrase_start,
                    "word_count": len(current_phrase)
                })
            
            # Start new phrase
            current_phrase = [word]
            phrase_start = word["start"]
        else:
            current_phrase.append(word)
    
    # Don't forget the last phrase
    if current_phrase:
        phrase_text = " ".join([w["word"] for w in current_phrase])
        phrase_text = clean_transcript(phrase_text)
        
        if phrase_text:
            phrases.append({
                "text": phrase_text,
                "start": phrase_start,
                "end": current_phrase[-1]["end"],
                "duration": current_phrase[-1]["end"] - phrase_start,
                "word_count": len(current_phrase)
            })
    
    return phrases


def get_speaker_info(session_id, candor_path, speechmatics_path):
    """Get speaker information and file paths for a session"""
    candor_path = Path(candor_path)
    speechmatics_path = Path(speechmatics_path)
    
    # Load channel map
    channel_map_file = candor_path / session_id / "processed" / "channel_map.json"
    with open(channel_map_file) as f:
        channel_map = json.load(f)
    
    return {
        "session_id": session_id,
        "speakers": [
            {
                "speaker_id": "spk0",
                "channel": "L",
                "user_id": channel_map["L"],
                "video_path": candor_path / session_id / "processed" / f"{channel_map['L']}.mp4",
                "transcript_path": speechmatics_path / f"{session_id}_0.json",
            },
            {
                "speaker_id": "spk1",
                "channel": "R",
                "user_id": channel_map["R"],
                "video_path": candor_path / session_id / "processed" / f"{channel_map['R']}.mp4",
                "transcript_path": speechmatics_path / f"{session_id}_1.json",
            }
        ]
    }


def process_candor_session(session_id, candor_path, speechmatics_path, output_path, 
                           text_path, landmarks_detector, video_process, args, output_size, 
                           combined_av_path=None):
    """Process a single Candor session"""
    
    print(f"\n{'='*70}")
    print(f"Processing session: {session_id}")
    print(f"{'='*70}")
    
    # Get speaker info
    try:
        speaker_info = get_speaker_info(session_id, candor_path, speechmatics_path)
    except Exception as e:
        print(f"❌ Error loading speaker info: {e}")
        return []
    
    processed_data = []
    filtered_stats = {
        'too_short_duration': 0,
        'too_few_words': 0,
        'filler_words': 0,
        'total_processed': 0
    }
    
    # Process each speaker
    for speaker in speaker_info["speakers"]:
        speaker_id = speaker["speaker_id"]
        video_path = speaker["video_path"]
        transcript_path = speaker["transcript_path"]
        
        print(f"\n🎤 Processing {speaker_id} ({speaker['user_id']})")
        print(f"   Video: {video_path.name}")
        print(f"   Transcript: {transcript_path.name}")
        
        # Check if files exist
        if not video_path.exists():
            print(f"   ⚠️ Video file not found, skipping")
            continue
        
        if not transcript_path.exists():
            print(f"   ⚠️ Transcript file not found, skipping")
            continue
        
        # Load transcript
        try:
            words = load_speechmatics_transcript(transcript_path)
            print(f"   📝 Loaded {len(words)} words")
        except Exception as e:
            print(f"   ❌ Error loading transcript: {e}")
            continue
        
        # Group into phrases
        phrases = group_words_into_phrases(
            words, 
            min_duration=args.min_phrase_duration,
            max_duration=args.max_phrase_duration,
            max_gap=args.max_word_gap
        )
        print(f"   📦 Grouped into {len(phrases)} phrases")
        
        if not phrases:
            print(f"   ⚠️ No valid phrases found, skipping")
            continue
        
        # Process each phrase
        for i, phrase in enumerate(tqdm(phrases, desc=f"   {speaker_id} phrases")):
            try:
                # Filter 1: Check minimum duration in milliseconds
                duration_ms = phrase["duration"] * 1000
                if duration_ms < args.min_duration_ms:
                    filtered_stats['too_short_duration'] += 1
                    continue
                
                # Filter 2: Check minimum word count
                if phrase["word_count"] < args.min_word_count:
                    filtered_stats['too_few_words'] += 1
                    continue
                
                # Filter 3: Check for filler words/noise (optional)
                if args.filter_fillers:
                    text_lower = phrase["text"].lower()
                    filler_words = ['uhm', 'uh', 'um', 'hmm', 'mhm', 'mm', 'hm', 'ah', 'eh', 'oh']
                    
                    # Skip if phrase is ONLY filler words
                    words_in_phrase = text_lower.split()
                    if all(word in filler_words for word in words_in_phrase):
                        filtered_stats['filler_words'] += 1
                        continue
                
                filtered_stats['total_processed'] += 1
                # Extract video/audio segment using timestamps
                video_segment, audio_segment, info = torchvision.io.read_video(
                    str(video_path),
                    start_pts=phrase["start"],
                    end_pts=phrase["end"],
                    pts_unit='sec'
                )
                
                # Validate segments
                if video_segment.size(0) == 0:
                    print(f"      ⚠️ Phrase {i}: Empty video segment")
                    continue
                
                if audio_segment.size(0) == 0:
                    print(f"      ⚠️ Phrase {i}: Empty audio segment")
                    continue
                
                # Convert video to numpy for RetinaFace
                video_np = video_segment.numpy()
                
                # Apply RetinaFace detection and cropping
                landmarks = landmarks_detector(video_np)
                processed_video = video_process(video_np, landmarks)
                
                if processed_video is None:
                    print(f"      ⚠️ Phrase {i}: Face detection failed")
                    continue
                
                # Convert back to tensor
                video_data = torch.tensor(processed_video)
                
                # Process audio
                audio_data = audio_segment
                
                # Convert to mono if stereo
                if audio_data.size(0) > 1:
                    audio_data = torch.mean(audio_data, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                if 'audio_fps' in info and info['audio_fps'] != 16000:
                    audio_data = torchaudio.functional.resample(
                        audio_data, info['audio_fps'], 16000
                    )
                
                # Create unique ID
                phrase_id = f"{session_id}_{speaker_id}_{i:04d}"
                
                # Create output paths (flat structure, no spk0/spk1 subfolders)
                dst_vid_filename = output_path / f"{phrase_id}.mp4"
                dst_aud_filename = output_path / f"{phrase_id}.wav"
                dst_txt_filename = text_path / f"{phrase_id}.txt"
                
                # Save files
                from utils import save_vid_aud_txt
                save_vid_aud_txt(
                    str(dst_vid_filename),
                    str(dst_aud_filename),
                    str(dst_txt_filename),
                    video_data,
                    audio_data,
                    phrase["text"],
                    video_fps=info.get('video_fps', 30),
                    audio_sample_rate=16000,
                )
                
                # Save combined AV file if requested (for sanity checking)
                if combined_av_path is not None:
                    combined_av_session_path = combined_av_path / session_id
                    combined_av_session_path.mkdir(parents=True, exist_ok=True)
                    
                    dst_combined_filename = combined_av_session_path / f"{phrase_id}_av.mp4"
                    
                    # Save video with audio embedded (using torchvision)
                    torchvision.io.write_video(
                        str(dst_combined_filename),
                        video_data,
                        fps=info.get('video_fps', 30),
                        audio_array=audio_data,
                        audio_fps=16000,
                        audio_codec='aac'
                    )
                
                # Add to processed data for CSV (flat structure)
                rel_video_path = f"{session_id}/{phrase_id}.mp4"
                processed_data.append([
                    speaker_id,                     # speaker_id
                    rel_video_path,                 # video_path (relative)
                    phrase["text"],                 # transcript
                    phrase["word_count"],           # word_count
                    phrase_id,                      # unique_id
                    f"{session_id}_{i:04d}",        # phrase_id
                    phrase["duration"],             # duration
                    phrase["start"],                # start_time
                    phrase["end"],                  # end_time
                    args.detector,                  # detector
                    args.crop_type,                 # crop_type
                    f"{output_size}x{output_size}", # resolution
                    speaker["user_id"],             # original_user_id
                ])
                
            except Exception as e:
                print(f"      ❌ Error processing phrase {i}: {e}")
                continue
        
        # Print filtering statistics for this speaker
        print(f"\n   📊 Filtering stats for {speaker_id}:")
        print(f"      ✅ Kept: {filtered_stats['total_processed']} phrases")
        print(f"      ⏱️ Too short duration: {filtered_stats['too_short_duration']}")
        print(f"      📝 Too few words: {filtered_stats['too_few_words']}")
        print(f"      🗣️ Filler words only: {filtered_stats['filler_words']}")
    
    return processed_data


def main(args):
    candor_path = Path(args.candor_path)
    speechmatics_path = Path(args.speechmatics_path)
    output_path = Path(args.output_path)
    
    # Create output directories with crop type and size suffixes
    crop_suffix = f"_{args.crop_type}" if args.crop_type != "lips" else ""
    output_size = 96 if args.crop_type == "lips" else 224
    size_suffix = f"_{output_size}x{output_size}" if output_size != 96 else ""
    detector_suffix = f"_{args.detector}" if args.detector != "retinaface" else ""
    
    final_output_path = output_path / f"candor_video{crop_suffix}{size_suffix}{detector_suffix}"
    final_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create separate text directory
    text_output_path = output_path / f"candor_text{crop_suffix}{size_suffix}{detector_suffix}"
    text_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create combined AV directory if requested
    combined_av_path = None
    if args.save_combined_av:
        combined_av_path = output_path / f"candor_video_av{crop_suffix}{size_suffix}{detector_suffix}"
        combined_av_path.mkdir(parents=True, exist_ok=True)
        print(f"📹 Combined AV files will be saved to: {combined_av_path}")
    
    # Initialize RetinaFace detector
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
        print(f"❌ Detector {args.detector} not implemented")
        return
    
    # Validate paths
    if not candor_path.exists():
        print(f"❌ Error: Candor path not found: {candor_path}")
        return
    
    if not speechmatics_path.exists():
        print(f"❌ Error: Speechmatics path not found: {speechmatics_path}")
        return
    
    # Get all session directories
    sessions = [d.name for d in candor_path.iterdir() if d.is_dir()]
    
    print(f"\n{'='*70}")
    print(f"CANDOR DATASET PREPROCESSING")
    print(f"{'='*70}")
    print(f"📁 Candor path: {candor_path}")
    print(f"📁 Speechmatics path: {speechmatics_path}")
    print(f"📁 Output path: {final_output_path}")
    print(f"✂️ Crop type: {args.crop_type} ({output_size}x{output_size})")
    print(f"🔍 Detector: {args.detector}")
    print(f"📊 Found {len(sessions)} sessions")
    print(f"⏱️ Phrase settings:")
    print(f"   Min duration: {args.min_phrase_duration}s")
    print(f"   Max duration: {args.max_phrase_duration}s")
    print(f"   Max word gap: {args.max_word_gap}s")
    print(f"🎬 Combined AV: {'Yes (for sanity checking)' if args.save_combined_av else 'No'}")
    
    # Collect all processed data for CSV
    all_processed_data = []
    
    # Process each session
    for session_id in tqdm(sessions, desc="Processing sessions"):
        try:
            # Create session output directories (flat structure, no spk0/spk1 subfolders)
            session_output_path = final_output_path / session_id
            session_output_path.mkdir(parents=True, exist_ok=True)
            
            session_text_path = text_output_path / session_id
            session_text_path.mkdir(parents=True, exist_ok=True)
            
            processed_data = process_candor_session(
                session_id, candor_path, speechmatics_path,
                session_output_path, session_text_path,
                landmarks_detector, video_process, args, output_size,
                combined_av_path
            )
            all_processed_data.extend(processed_data)
            
        except Exception as e:
            print(f"❌ Error processing session {session_id}: {e}")
            continue
    
    # Save CSV summary
    if all_processed_data:
        labels_dir = output_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        csv_filename = f"candor{crop_suffix}{size_suffix}{detector_suffix}.csv"
        csv_path = labels_dir / csv_filename
        
        # Create DataFrame
        df = pd.DataFrame(all_processed_data, columns=[
            'speaker_id', 'video_path', 'transcript', 'word_count',
            'unique_id', 'phrase_id', 'duration', 'start_time', 'end_time',
            'detector', 'crop_type', 'resolution', 'original_user_id'
        ])
        
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"📊 CSV saved: {csv_path}")
        print(f"📈 Total samples: {len(df)}")
        print(f"👥 Speakers: {df['speaker_id'].nunique()}")
        print(f"🎬 Sessions: {len(sessions)}")
        print(f"⏱️ Total duration: {df['duration'].sum():.1f}s ({df['duration'].sum()/60:.1f} min)")
        print(f"📝 Avg words per phrase: {df['word_count'].mean():.1f}")
        print(f"⏱️ Avg phrase duration: {df['duration'].mean():.1f}s")
        
        # Print per-speaker statistics
        print(f"\n📊 Per-speaker statistics:")
        for speaker_id in sorted(df['speaker_id'].unique()):
            spk_df = df[df['speaker_id'] == speaker_id]
            print(f"   {speaker_id}: {len(spk_df)} phrases, {spk_df['duration'].sum():.1f}s")
    
    print(f"\n✅ Processing complete! Output saved to: {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Candor Dataset Preprocessing - Step 1: Video Processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--candor-path', required=True,
                        help='Path to Candor_test directory')
    parser.add_argument('--speechmatics-path', required=True,
                        help='Path to candor_speechmatics directory')
    parser.add_argument('--output-path', required=True,
                        help='Output directory for processed files')
    parser.add_argument('--crop-type', choices=['lips', 'face'], default='lips',
                        help='Crop type: lips (96x96, mouth region) or face (224x224, full face)')
    parser.add_argument('--detector', default='retinaface',
                        help='Face detector to use')
    parser.add_argument('--min-phrase-duration', type=float, default=2.0,
                        help='Minimum phrase duration in seconds for grouping')
    parser.add_argument('--max-phrase-duration', type=float, default=5.0,
                        help='Maximum phrase duration in seconds for grouping')
    parser.add_argument('--max-word-gap', type=float, default=0.5,
                        help='Maximum gap between words to group in same phrase')
    parser.add_argument('--min-duration-ms', type=int, default=800,
                        help='Minimum duration in milliseconds to keep a phrase (filters very short clips)')
    parser.add_argument('--min-word-count', type=int, default=2,
                        help='Minimum number of words per phrase (filters out single-word fillers)')
    parser.add_argument('--filter-fillers', action='store_true',
                        help='Filter out phrases that are only filler words (uhm, uh, etc.)')
    parser.add_argument('--save-combined-av', action='store_true',
                        help='Save combined audio+video files in _av folder for sanity checking')
    
    args = parser.parse_args()
    
    main(args)
