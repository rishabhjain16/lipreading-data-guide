#!/usr/bin/env python3
"""
RoomReader Step 2: Generate Manifests

This script generates manifests from processed RoomReader data.

Default behavior (no --split-ratios):
- Creates test-only manifests in three folders: conversational/, individual/, combined/
- All data treated as test data (no train/val splits)

With --split-ratios:
- Creates train/valid/test splits with the specified ratios
- Can use --create-mode-splits to separate conversational and individual modes

Usage:
    # Default: Create test-only manifests for all three modes
    python step2.py \
        --roomreader-data-dir /path/to/processed/roomreader_video \
        --metadata-dir /path/to/metadata

    # With splits: Create train/val/test splits
    python step2.py \
        --roomreader-data-dir /path/to/processed/roomreader_video \
        --metadata-dir /path/to/metadata \
        --split-ratios 0.7,0.15,0.15 \
        --create-mode-splits
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

# Shared SPM tokenizer
from transforms import TextTransform


def write_tokens_and_label_csv(text_transform, data_dir, valid_records, out_dir, dataset_name="roomreader"):
    """Write tokens.txt and label.csv next to manifests.

    label.csv format (no header):
        dataset,video_path,token_ids(space-separated)
    """
    if not valid_records:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = out_dir / "tokens.txt"
    label_csv_path = out_dir / "label.csv"

    with open(tokens_path, "w") as ftok, open(label_csv_path, "w") as fc:
        for r in valid_records:
            transcript = r.get("transcript", "")
            token_ids = text_transform.tokenize(transcript)
            token_str = " ".join(str(t.item()) for t in token_ids)
            ftok.write(token_str + "\n")

            video_abs = str(Path(r["video_path"]).resolve())
            fc.write(f"{dataset_name},{video_abs},{token_str}\n")

    print(f"✅ Created tokenized labels: {tokens_path}")
    print(f"✅ Created label CSV: {label_csv_path}")

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

def generate_test_manifest(data_dir, data, metadata_dir, mode_name):
    """Generate test.tsv and test.wrd files (following LRS format with video frames first)"""
    print(f"\n{'='*60}")
    print(f"Creating test manifest for {mode_name} mode...")
    print(f"{'='*60}")
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if not data:
        print(f"⚠️ Warning: No data for {mode_name} mode")
        return
    
    print(f"🔄 Processing {len(data)} files...")
    
    tsv_path = metadata_dir / "test.tsv"
    wrd_path = metadata_dir / "test.wrd"
    
    valid_records = []
    
    # Process each record and count frames
    for record in tqdm(data, desc=f"Processing {mode_name}"):
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
    
    # Write .tsv file (following LRS format: id, video_path, audio_path, num_video_frames, num_audio_frames)
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
    
    print(f"✅ Created test manifest: {tsv_path} ({len(valid_records)} entries)")
    print(f"✅ Created word file: {wrd_path}")

    # Tokenization outputs (shared SPM)
    try:
        text_transform = TextTransform()
        write_tokens_and_label_csv(text_transform, data_dir, valid_records, metadata_dir, dataset_name="roomreader")
    except Exception as e:
        print(f"⚠️ Warning: Could not create tokens.txt/label.csv for {mode_name}: {e}")
    
    # Print statistics
    if valid_records:
        total_frames = sum(r['frame_count'] for r in valid_records)
        total_audio = sum(r['audio_frames'] for r in valid_records)
        fps = 25  # Approximate FPS
        avg_duration = total_frames / (fps * len(valid_records)) if valid_records else 0
        
        print(f"📈 Stats: {total_frames:,} video frames, {total_audio:,} audio frames, {avg_duration:.1f}s avg")


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

        # Tokenization outputs (shared SPM)
        try:
            text_transform = TextTransform()
            write_tokens_and_label_csv(text_transform, data_dir, valid_records, metadata_dir, dataset_name="roomreader")
        except Exception as e:
            print(f"⚠️ Warning: Could not create tokens.txt/label.csv for {split_name}: {e}")
        
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
    parser.add_argument('--split-ratios', type=str, default=None,
                        help='Train/val/test split ratios (comma-separated). If not provided, creates test-only manifests')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    parser.add_argument('--random-split', action='store_true',
                        help='Use random splitting instead of speaker-based splitting (for testing)')
    parser.add_argument('--create-mode-splits', action='store_true',
                        help='Create separate metadata folders for conversational and individual modes')
    
    args = parser.parse_args()
    
    # Parse split ratios (optional)
    split_ratios = None
    if args.split_ratios:
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
    if split_ratios:
        print(f"📈 Split ratios: {split_ratios}")
    else:
        print(f"📈 Mode: Test-only manifests (no train/val splits)")
    
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
        print(f"\n🚀 Loading data...")
        all_data, conversational_data, individual_data = load_csv_data(labels_dir, crop_suffix)
        
        # Default behavior: Create test-only manifests for three modes
        if not split_ratios:
            print(f"\n📋 Creating test-only manifests for all modes...")
            
            # Create conversational metadata
            if conversational_data:
                conversational_dir = metadata_dir / "conversational"
                generate_test_manifest(data_dir, conversational_data, conversational_dir, "conversational")
            
            # Create individual metadata
            if individual_data:
                individual_dir = metadata_dir / "individual"
                generate_test_manifest(data_dir, individual_data, individual_dir, "individual")
            
            # Create combined metadata
            combined_dir = metadata_dir / "combined"
            generate_test_manifest(data_dir, all_data, combined_dir, "combined")
            
            print(f"\n{'='*60}")
            print(f"✅ RoomReader Step 2 completed successfully!")
            print(f"{'='*60}")
            print(f"\nTest-only manifests created:")
            if conversational_data:
                print(f"  📞 Conversational ({len(conversational_data)} files): {metadata_dir}/conversational/")
            if individual_data:
                print(f"  👤 Individual ({len(individual_data)} files): {metadata_dir}/individual/")
            print(f"  🔗 Combined ({len(all_data)} files): {metadata_dir}/combined/")
        
        # With split ratios: Create train/val/test splits
        else:
            print(f"\n📋 Creating train/val/test splits...")
            
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
                
                print(f"\n✅ RoomReader Step 2 completed successfully!")
                print(f"📁 Mode-specific manifests created:")
                if conversational_data:
                    print(f"   📞 Conversational: {metadata_dir.parent}/metadata_conversational/")
                if individual_data:
                    print(f"   👤 Individual: {metadata_dir.parent}/metadata_individual/")
            else:
                # Original behavior - process all data together
                if args.random_split:
                    splits = split_data_randomly(all_data, split_ratios, args.seed)
                else:
                    splits = split_data_by_speaker(all_data, split_ratios, args.seed)
                
                # Generate training manifests (.tsv and .wrd files)
                generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix)
                
                print(f"\n✅ RoomReader Step 2 completed successfully!")
                print(f"📁 Combined manifests created in: {metadata_dir}")
                print(f"   • Manifests: train.tsv, valid.tsv, test.tsv")
                print(f"   • Word files: train.wrd, valid.wrd, test.wrd")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
