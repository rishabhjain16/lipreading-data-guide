#!/usr/bin/env python3
"""
Candor Step 2: Generate Training Manifests

This script generates training manifests from processed Candor data:
- Generates train.tsv, valid.tsv, test.tsv files (video metadata)
- Generates train.wrd, valid.wrd, test.wrd files (transcripts)
- Counts frames and creates detailed metadata

Usage:
    python step2_generate_file_lists.py \
        --candor-data-dir /path/to/processed/candor_video \
        --metadata-dir /path/to/metadata \
        --split-ratios 0.7,0.15,0.15
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

# Import TextTransform for tokenization
from transforms import TextTransform


def clean_transcript(text):
    """Clean transcript text by removing punctuation and normalizing"""
    if not text or text.strip() == "":
        return ""
    
    # Remove common punctuation (keeping apostrophes)
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
    
    if '_face' in dir_name:
        return 'face'
    else:
        return 'lips'  # default


def load_csv_data(labels_dir, crop_suffix):
    """Load data from CSV files in the labels directory"""
    print(f"📁 Loading CSV data from: {labels_dir}")
    
    # Find CSV files
    csv_files = [f for f in labels_dir.glob('*.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {labels_dir}")
    
    print(f"📄 Found CSV files: {[f.name for f in csv_files]}")
    
    all_data = []
    
    for csv_file in csv_files:
        print(f"📊 Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        records = df.to_dict('records')
        all_data.extend(records)
    
    print(f"✅ Loaded {len(all_data)} records total")
    
    return all_data


def load_official_splits(splits_dir):
    """Load official train/val/test splits from files"""
    splits_dir = Path(splits_dir)
    
    splits = {}
    for split_name in ['train', 'valid', 'test']:
        split_file = splits_dir / f"candor-{split_name}.id"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Official split file not found: {split_file}")
        
        with open(split_file) as f:
            sessions = [line.strip() for line in f if line.strip()]
        
        splits[split_name] = sessions
        print(f"📄 Loaded {split_name}: {len(sessions)} sessions")
    
    return splits


def split_data_by_official_splits(data, official_splits):
    """Split data using official session splits"""
    print(f"🎯 Using official splits")
    
    # Group by session
    session_data = defaultdict(list)
    for record in data:
        # Extract session ID from unique_id (format: {session_id}_{speaker_id}_{phrase_id})
        unique_id = record['unique_id']
        session_id = '_'.join(unique_id.split('_')[:-2])  # Remove speaker and phrase parts
        session_data[session_id].append(record)
    
    # Create splits based on official session lists
    splits = {
        'train': [],
        'valid': [],
        'test': []
    }
    
    for split_name, session_list in official_splits.items():
        for session_id in session_list:
            if session_id in session_data:
                splits[split_name].extend(session_data[session_id])
            else:
                print(f"⚠️  Warning: Session {session_id} in {split_name} split not found in data")
    
    print(f"📊 Split statistics:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Valid: {len(splits['valid'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    return splits


def split_data_by_session(data, split_ratios, seed=42):
    """Split data by session to ensure no session overlap between splits"""
    print(f"🎯 Splitting data by session with ratios: {split_ratios}")
    
    # Group by session (extract from unique_id or video_path)
    session_data = defaultdict(list)
    for record in data:
        # Extract session ID from unique_id (format: {session_id}_{speaker_id}_{phrase_id})
        unique_id = record['unique_id']
        session_id = '_'.join(unique_id.split('_')[:-2])  # Remove speaker and phrase parts
        session_data[session_id].append(record)
    
    sessions = list(session_data.keys())
    print(f"🎬 Found {len(sessions)} sessions")
    
    # Shuffle sessions for random assignment
    random.seed(seed)
    random.shuffle(sessions)
    
    # Calculate split indices
    n_sessions = len(sessions)
    train_end = int(n_sessions * split_ratios['train'])
    valid_end = train_end + int(n_sessions * split_ratios['valid'])
    
    train_sessions = sessions[:train_end]
    valid_sessions = sessions[train_end:valid_end]
    test_sessions = sessions[valid_end:]
    
    print(f"📊 Session splits:")
    print(f"  Train: {len(train_sessions)} sessions")
    print(f"  Valid: {len(valid_sessions)} sessions")
    print(f"  Test: {len(test_sessions)} sessions")
    
    # Create splits
    splits = {
        'train': [],
        'valid': [],
        'test': []
    }
    
    for session in train_sessions:
        splits['train'].extend(session_data[session])
    for session in valid_sessions:
        splits['valid'].extend(session_data[session])
    for session in test_sessions:
        splits['test'].extend(session_data[session])
    
    print(f"📈 Sample counts:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Valid: {len(splits['valid'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    return splits


def split_data_by_speaker(data, split_ratios, seed=42):
    """Split data by speaker to ensure no speaker overlap between splits"""
    print(f"🎯 Splitting data by speaker with ratios: {split_ratios}")
    
    # Group by speaker
    speaker_data = defaultdict(list)
    for record in data:
        speaker_data[record['speaker_id']].append(record)
    
    speakers = list(speaker_data.keys())
    print(f"👥 Found {len(speakers)} speakers")
    
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
    print(f"  Train: {len(train_speakers)} speakers")
    print(f"  Valid: {len(valid_speakers)} speakers")
    print(f"  Test: {len(test_speakers)} speakers")
    
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


def generate_split_csvs(splits, labels_dir, crop_suffix, all_data, spm_model_path=None):
    """Generate separate CSV files for each split (Auto-AVSR format with SentencePiece tokenization)"""
    print(f"📝 Generating split-specific CSV files (Auto-AVSR format)...")
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TextTransform for tokenization
    print(f"🔤 Loading SentencePiece tokenizer...")
    
    # Determine SPM model path
    if spm_model_path:
        # Use provided path
        model_path = spm_model_path
        dict_path = spm_model_path.replace('.model', '_units.txt')
    else:
        # Default: use spm1000 from parent directory
        parent_dir = Path(__file__).parent.parent
        model_path = parent_dir / 'spm' / 'unigram' / 'unigram1000.model'
        dict_path = parent_dir / 'spm' / 'unigram' / 'unigram1000_units.txt'
    
    try:
        text_transform = TextTransform(
            sp_model_path=str(model_path),
            dict_path=str(dict_path)
        )
        print(f"   ✅ Tokenizer loaded: {model_path.name if hasattr(model_path, 'name') else model_path}")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not load tokenizer: {e}")
        print(f"   Falling back to character-level tokenization")
        text_transform = None
    
    for split_name, split_data in splits.items():
        if not split_data:
            print(f"⚠️ Warning: No data for {split_name} split")
            continue
        
        csv_path = labels_dir / f"candor_{split_name}{crop_suffix}.csv"
        
        # Create Auto-AVSR format CSV
        # Format: dataset_name,video_path,duration_frames,token_id_1 token_id_2 ...
        with open(csv_path, 'w') as f:
            for record in split_data:
                # Get video path (relative)
                video_path = record['video_path']
                
                # Get duration in frames (approximate from duration in seconds)
                # Assuming 30 fps
                duration_frames = int(record.get('duration', 0) * 30)
                
                # Tokenize transcript
                if text_transform:
                    # Use SentencePiece tokenization (same as LRS3)
                    token_ids = " ".join(str(t.item()) for t in text_transform.tokenize(record['transcript']))
                else:
                    # Fallback: character-level tokenization
                    token_ids = " ".join(str(ord(c)) for c in record['transcript'])
                
                # Write in Auto-AVSR format
                f.write(f"candor,{video_path},{duration_frames},{token_ids}\n")
        
        print(f"✅ {split_name}: {len(split_data)} samples → {csv_path.name}")


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
                    # For audio frame count, approximate based on 16kHz sample rate
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
        print(f"   📊 Manifest: {tsv_path}")
        print(f"   📄 Words: {wrd_path}")
        
        # Print statistics
        if valid_records and fps > 0:
            total_frames = sum(r['frame_count'] for r in valid_records)
            total_audio = sum(r['audio_frames'] for r in valid_records)
            avg_duration = total_frames / (fps * len(valid_records))
            
            print(f"   📈 Stats: {total_frames:,} video frames, {total_audio:,} audio frames, {avg_duration:.1f}s avg")


def main():
    parser = argparse.ArgumentParser(
        description='Candor Step 2: Generate training manifests (.tsv/.wrd files)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--candor-data-dir', type=str, required=True,
                        help='Candor processed data directory (contains video files)')
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--split-ratios', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratios (comma-separated)')
    parser.add_argument('--split-by', type=str, default='session', choices=['session', 'speaker'],
                        help='Split by session or speaker')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    parser.add_argument('--use-official-splits', action='store_true',
                        help='Use official train/val/test splits (ensures reproducibility)')
    parser.add_argument('--splits-dir', type=str, default='./splits',
                        help='Directory containing official split files (candor-train.id, etc.)')
    parser.add_argument('--spm-model', type=str, default=None,
                        help='Path to SentencePiece model (default: uses spm1000 from parent directory)')
    
    args = parser.parse_args()
    
    # Parse split ratios
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.001:
        print("❌ Error: Split ratios must be three numbers that sum to 1.0")
        return 1
    
    split_ratios = {'train': ratios[0], 'valid': ratios[1], 'test': ratios[2]}
    
    # Validate input directory
    data_dir = Path(args.candor_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    
    if not data_dir.exists():
        print(f"❌ Error: Candor data directory not found: {data_dir}")
        return 1
    
    # Detect crop type
    crop_type = detect_crop_type(str(data_dir))
    crop_suffix = f"_{crop_type}" if crop_type != "lips" else ""
    
    print(f"🎯 Candor Step 2 Processing")
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Metadata directory: {metadata_dir}")
    print(f"✂️ Crop type: {crop_type}")
    print(f"📈 Split ratios: {split_ratios}")
    print(f"🎲 Split by: {args.split_by}")
    
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
        # Load CSV data
        print(f"\n🚀 Loading data and creating splits...")
        all_data = load_csv_data(labels_dir, crop_suffix)
        
        # Create splits
        if args.use_official_splits:
            # Load official splits
            print(f"\n📂 Loading official splits from: {args.splits_dir}")
            official_splits = load_official_splits(args.splits_dir)
            splits = split_data_by_official_splits(all_data, official_splits)
        else:
            # Create splits dynamically
            if args.split_by == 'session':
                splits = split_data_by_session(all_data, split_ratios, args.seed)
            else:
                splits = split_data_by_speaker(all_data, split_ratios, args.seed)
        
        # Generate training manifests (.tsv and .wrd files)
        generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix)
        
        # Generate split-specific CSV files (for Auto-AVSR)
        generate_split_csvs(splits, labels_dir, crop_suffix, all_data, args.spm_model)
        
        print(f"\n✅ Candor Step 2 completed successfully!")
        print(f"📁 Manifests created in: {metadata_dir}")
        print(f"   • Manifests: train.tsv, valid.tsv, test.tsv")
        print(f"   • Word files: train.wrd, valid.wrd, test.wrd")
        print(f"📁 Split CSVs created in: {labels_dir}")
        print(f"   • candor_train{crop_suffix}.csv (Auto-AVSR format with SentencePiece tokens)")
        print(f"   • candor_valid{crop_suffix}.csv (Auto-AVSR format with SentencePiece tokens)")
        print(f"   • candor_test{crop_suffix}.csv (Auto-AVSR format with SentencePiece tokens)")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
