#!/usr/bin/env python3
"""
RoomReader Step 2: Generate Training Manifests

This script generates the training manifests from processed RoomReader data:
- Generates train.tsv, valid.tsv, test.tsv files (video metadata)
- Generates train.wrd, valid.wrd, test.wrd files (transcripts)
- Counts frames and creates detailed metadata

Usage:
    python step2.py --roomreader-data-dir /path/to/processed/roomreader_video --metadata-dir /path/to/metadata --split-ratios 0.7,0.15,0.15
"""

import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random
import re

def clean_transcript(text):
    """Clean transcript text by removing punctuation and normalizing"""
    if not text or text.strip() == "":
        return ""
    
    # Remove common punctuation (keeping apostrophes)
    # Remove: : , . ! ? ; - " ( ) [ ] { } @ % < >
    text = re.sub(r'[:,.!?;\-"()\[\]{}<>@%]', '', text)
    # Remove double dashes and ellipsis
    text = re.sub(r'--+|\.\.\.+', ' ', text)
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_crop_type(data_dir):
    """Detect crop type from directory name suffix"""
    dir_name = os.path.basename(data_dir.rstrip('/'))
    
    if dir_name.endswith('_face'):
        return 'face'
    elif dir_name.endswith('_full'):
        return 'full'
    else:
        return 'lips'  # default

def load_csv_data(labels_dir, crop_suffix):
    """Load data from CSV files in the labels directory"""
    print(f"📁 Loading CSV data from: {labels_dir}")
    
    # Find CSV files
    csv_files = [f.name for f in labels_dir.glob('*.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {labels_dir}")
    
    print(f"📄 Found CSV files: {csv_files}")
    
    all_data = []
    conversational_data = []
    individual_data = []
    
    for csv_file in csv_files:
        csv_path = labels_dir / csv_file
        print(f"📊 Loading: {csv_path}")
        
        df = pd.read_csv(csv_path)
        records = df.to_dict('records')
        all_data.extend(records)
        
        # Separate by mode based on CSV filename or video_path
        if 'conversational' in csv_file.lower():
            conversational_data.extend(records)
        elif 'individual' in csv_file.lower():
            individual_data.extend(records)
        else:
            # Check video_path for mode identification
            for record in records:
                if 'conversational' in record.get('video_path', '').lower():
                    conversational_data.append(record)
                elif 'individual' in record.get('video_path', '').lower():
                    individual_data.append(record)
                else:
                    # Default to individual if unclear
                    individual_data.append(record)
    
    print(f"✅ Loaded {len(all_data)} records total")
    print(f"   📞 Conversational: {len(conversational_data)} records")
    print(f"   👤 Individual: {len(individual_data)} records")
    
    return all_data, conversational_data, individual_data

def split_data_by_speaker(data, split_ratios, seed=42, force_random=False):
    """Split data by speaker to ensure no speaker overlap between splits"""
    print(f"🎯 Splitting data by speaker with ratios: {split_ratios}")
    
    # Group by speaker
    speaker_data = defaultdict(list)
    for record in data:
        speaker_data[record['speaker_id']].append(record)
    
    speakers = list(speaker_data.keys())
    print(f"👥 Found {len(speakers)} speakers: {speakers}")
    
    # For small datasets with few speakers, optionally use random splitting
    if len(speakers) < 3 and not force_random:
        print(f"⚠️ Warning: Only {len(speakers)} speakers found. This will result in empty splits.")
        print(f"💡 Consider using --random-split for small test datasets")
    
    # Shuffle speakers for random assignment
    random.seed(seed)
    random.shuffle(speakers)
    
    # Calculate split indices
    n_speakers = len(speakers)
    train_end = int(n_speakers * split_ratios['train'])
    valid_end = train_end + int(n_speakers * split_ratios['valid'])
    
    train_speakers = speakers[:train_end]
    valid_speakers = speakers[train_end:valid_end]
    test_speakers = speakers[valid_end:]
    
    print(f"📊 Speaker splits:")
    print(f"  Train: {len(train_speakers)} speakers - {train_speakers}")
    print(f"  Valid: {len(valid_speakers)} speakers - {valid_speakers}")
    print(f"  Test: {len(test_speakers)} speakers - {test_speakers}")
    
    # Create splits
    splits = {
        'train': [],
        'valid': [],
        'test': []
    }
    
    for speaker in train_speakers:
        splits['train'].extend(speaker_data[speaker])
    for speaker in valid_speakers:
        splits['valid'].extend(speaker_data[speaker])
    for speaker in test_speakers:
        splits['test'].extend(speaker_data[speaker])
    
    print(f"📈 Sample counts:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Valid: {len(splits['valid'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    return splits

def split_data_randomly(data, split_ratios, seed=42):
    """Split data randomly (for testing with small datasets)"""
    print(f"🎲 Splitting data randomly with ratios: {split_ratios}")
    
    # Shuffle data
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    n_samples = len(shuffled_data)
    train_end = int(n_samples * split_ratios['train'])
    valid_end = train_end + int(n_samples * split_ratios['valid'])
    
    splits = {
        'train': shuffled_data[:train_end],
        'valid': shuffled_data[train_end:valid_end],
        'test': shuffled_data[valid_end:]
    }
    
    print(f"📈 Sample counts:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Valid: {len(splits['valid'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    return splits

def generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix):
    """Generate .tsv and .wrd files for training (following LRS format)"""
    print(f"📝 Generating training manifests (.tsv and .wrd files)...")
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        if not split_data:
            print(f"⚠️ Warning: No data for {split_name} split")
            continue
            
        print(f"🔄 Processing {split_name} split ({len(split_data)} files)...")
        
        tsv_path = metadata_dir / f"{split_name}.tsv"
        wrd_path = metadata_dir / f"{split_name}.wrd"
        
        valid_records = []
        
        # Process each record and count frames
        for record in tqdm(split_data, desc=f"Processing {split_name}"):
            video_path = data_dir / record['video_path']
            audio_path = video_path.with_suffix('.wav')
            
            if not video_path.exists():
                print(f"⚠️ Warning: Video file not found: {video_path}")
                continue
                
            if not audio_path.exists():
                print(f"⚠️ Warning: Audio file not found: {audio_path}")
                continue
            
            try:
                # Count frames using OpenCV
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                if frame_count > 0:
                    # For audio frame count, we approximate based on 16kHz sample rate
                    # This could be made more accurate by actually reading the audio file
                    duration = frame_count / fps if fps > 0 else 0
                    audio_frames = int(duration * 16000)  # 16kHz sample rate
                    
                    valid_records.append({
                        'file_id': record['unique_id'],
                        'video_path': str(video_path.absolute()),
                        'audio_path': str(audio_path.absolute()),
                        'frame_count': frame_count,
                        'audio_frames': audio_frames,
                        'transcript': clean_transcript(record['transcript'])
                    })
                else:
                    print(f"⚠️ Warning: Invalid frame count for {video_path}")
                    
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                continue
        
        # Write .tsv file (following LRS format)
        with open(tsv_path, 'w') as f:
            f.write('/\n')  # Header line
            for record in valid_records:
                f.write('\t'.join([
                    record['file_id'],
                    record['video_path'],
                    record['audio_path'],
                    str(record['frame_count']),
                    str(record['audio_frames'])
                ]) + '\n')
        
        # Write .wrd file
        with open(wrd_path, 'w') as f:
            for record in valid_records:
                f.write(f"{record['transcript']}\n")
        
        print(f"✅ {split_name}: {len(valid_records)}/{len(split_data)} valid files")
        print(f"   � Manifest: {tsv_path}")
        print(f"   📄 Words: {wrd_path}")
        
        # Print statistics
        total_frames = sum(r['frame_count'] for r in valid_records)
        total_audio = sum(r['audio_frames'] for r in valid_records)
        avg_duration = total_frames / (fps * len(valid_records)) if valid_records and fps > 0 else 0
        
        print(f"   📈 Stats: {total_frames:,} video frames, {total_audio:,} audio frames, {avg_duration:.1f}s avg")

def main():
    parser = argparse.ArgumentParser(
        description='RoomReader Step 2: Generate training manifests (.tsv/.wrd files)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--roomreader-data-dir', type=str, required=True,
                        help='RoomReader processed data directory (contains video files)')
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--split-ratios', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratios (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    parser.add_argument('--random-split', action='store_true',
                        help='Use random splitting instead of speaker-based splitting (for testing)')
    parser.add_argument('--create-mode-splits', action='store_true',
                        help='Create separate metadata folders for conversational and individual modes')
    
    args = parser.parse_args()
    
    # Parse split ratios
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.001:
        print("❌ Error: Split ratios must be three numbers that sum to 1.0")
        return 1
    
    split_ratios = {'train': ratios[0], 'valid': ratios[1], 'test': ratios[2]}
    
    # Validate input directory
    data_dir = Path(args.roomreader_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    
    if not data_dir.exists():
        print(f"❌ Error: RoomReader data directory not found: {data_dir}")
        return 1
    
    # Detect crop type and setup paths
    crop_type = detect_crop_type(str(data_dir))
    crop_suffix = f"_{crop_type}" if crop_type != "lips" else ""
    
    print(f"🎯 RoomReader Step 2 Processing")
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Metadata directory: {metadata_dir}")
    print(f"✂️ Crop type: {crop_type}")
    print(f"📈 Split ratios: {split_ratios}")
    
    # Find labels directory
    labels_dir = None
    for possible_label_dir in [data_dir / "labels", data_dir.parent / "labels"]:
        if possible_label_dir.exists():
            labels_dir = possible_label_dir
            break
    
    if not labels_dir:
        print(f"❌ Error: Labels directory not found. Looked in:")
        print(f"  - {data_dir / 'labels'}")
        print(f"  - {data_dir.parent / 'labels'}")
        return 1
    
    try:
        # Load CSV data and create splits
        print(f"\n🚀 Loading data and creating splits...")
        all_data, conversational_data, individual_data = load_csv_data(labels_dir, crop_suffix)
        
        if args.create_mode_splits:
            # Create separate metadata folders for each mode
            modes_data = {
                'conversational': conversational_data,
                'individual': individual_data
            }
            
            for mode_name, mode_data in modes_data.items():
                if not mode_data:
                    print(f"⚠️ Warning: No {mode_name} data found, skipping...")
                    continue
                
                print(f"\n🎯 Processing {mode_name} mode ({len(mode_data)} samples)...")
                mode_metadata_dir = metadata_dir.parent / f"metadata_{mode_name}"
                
                if args.random_split:
                    splits = split_data_randomly(mode_data, split_ratios, args.seed)
                else:
                    splits = split_data_by_speaker(mode_data, split_ratios, args.seed)
                
                # Generate training manifests for this mode
                generate_training_manifests(data_dir, splits, mode_metadata_dir, crop_suffix)
                
                print(f"✅ {mode_name.capitalize()} mode completed!")
                print(f"📁 Manifests created in: {mode_metadata_dir}")
        else:
            # Original behavior - process all data together
            if args.random_split:
                splits = split_data_randomly(all_data, split_ratios, args.seed)
            else:
                splits = split_data_by_speaker(all_data, split_ratios, args.seed)
            
            # Generate training manifests (.tsv and .wrd files)
            generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix)
        
        print(f"\n✅ RoomReader Step 2 completed successfully!")
        if args.create_mode_splits:
            print(f"📁 Mode-specific manifests created:")
            if conversational_data:
                print(f"   📞 Conversational: {metadata_dir.parent}/metadata_conversational/")
            if individual_data:
                print(f"   👤 Individual: {metadata_dir.parent}/metadata_individual/")
        else:
            print(f"📁 Combined manifests created in: {metadata_dir}")
            print(f"   • Manifests: train.tsv, valid.tsv, test.tsv")
            print(f"   • Word files: train.wrd, valid.wrd, test.wrd")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
