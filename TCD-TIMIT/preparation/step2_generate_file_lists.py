#!/usr/bin/env python3
"""
TCD-TIMIT Step 2: Generate File Lists

This script generates file.list and label.list from the processed TCD-TIMIT data.
Unlike LRS2/LRS3, TCD-TIMIT has a simpler structure where all speakers repeat
the same sentences, so we need to handle speaker-based splits.

Usage:
    python step2_generate_file_lists.py --tcd-data-dir /path/to/processed/tcd_timit/tcd_timit_video_seg16s
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random

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
    """Load data from CSV files created by Step 1"""
    print("📊 Loading data from CSV files...")
    
    fids, labels = [], []
    
    # New CSV format from updated step1
    csv_files = []
    for subset in ['volunteers', 'lipspeakers']:
        for crop in ['face', 'full', 'lips']:
            for size in ['96x96', '224x224']:
                csv_file = f'tcd_timit_{subset}_{crop}_{size}.csv'
                csv_path = labels_dir / csv_file
                if csv_path.exists():
                    csv_files.append(csv_file)
    
    # If no specific files found, try to find any CSV files
    if not csv_files:
        csv_files = [f.name for f in labels_dir.glob('*.csv')]
    
    for csv_file in csv_files:
        csv_path = labels_dir / csv_file
        if not csv_path.exists():
            continue
        
        print(f"  📝 Processing: {csv_file}")
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            speaker_id = row['speaker_id']
            video_path = row['video_path']
            transcript = row['transcript']
            transcript_id = row['transcript_id']
            
            # Extract relative path from video_path
            # video_path format: "tcd_timit_video_seg16s_face_224x224/volunteers/01M/sa1.mp4"
            # We want: "volunteers/01M/sa1"
            path_parts = video_path.split('/')
            if len(path_parts) >= 4:  # [dataset_dir, subset, speaker, file.mp4]
                relative_path = '/'.join(path_parts[1:])  # subset/speaker/file.mp4
                file_id = relative_path.replace('.mp4', '')  # subset/speaker/file
                fids.append(file_id)
                labels.append(transcript)
    
    print(f"  ✅ Loaded {len(fids)} files total")
    return fids, labels

def create_speaker_splits(fids, labels, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
    """
    Create train/val/test splits based on speakers for TCD-TIMIT
    This ensures no speaker appears in multiple splits
    Uses a fixed seed for reproducible splits
    """
    print("🔄 Creating speaker-based splits...")
    
    # Set seed for reproducible splits
    random.seed(seed)
    
    # Group files by speaker
    speaker_files = defaultdict(list)
    for i, fid in enumerate(fids):
        # Extract speaker ID from file path
        # Expected format: tcd_timit_video_seg16s/volunteers/speaker_id/transcript_id
        path_parts = fid.split('/')
        if len(path_parts) >= 3:
            subset = path_parts[1]  # volunteers or lipspeakers
            speaker_id = path_parts[2]
            speaker_key = f"{subset}_{speaker_id}"
            speaker_files[speaker_key].append(i)
    
    print(f"  📊 Found {len(speaker_files)} unique speakers")
    
    # Sort speakers for base consistency, then shuffle with seed for randomization
    speakers = sorted(speaker_files.keys())
    random.shuffle(speakers)  # This uses the seed set above
    
    # Calculate split indices
    n_speakers = len(speakers)
    train_end = int(n_speakers * split_ratios['train'])
    val_end = train_end + int(n_speakers * split_ratios['val'])
    
    train_speakers = speakers[:train_end]
    val_speakers = speakers[train_end:val_end]
    test_speakers = speakers[val_end:]
    
    print(f"  📊 Speaker split:")
    print(f"    • Train: {len(train_speakers)} speakers")
    print(f"    • Val: {len(val_speakers)} speakers")
    print(f"    • Test: {len(test_speakers)} speakers")
    
    # Create file indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for speaker in train_speakers:
        train_indices.extend(speaker_files[speaker])
    for speaker in val_speakers:
        val_indices.extend(speaker_files[speaker])
    for speaker in test_speakers:
        test_indices.extend(speaker_files[speaker])
    
    print(f"  📊 File split:")
    print(f"    • Train: {len(train_indices)} files")
    print(f"    • Val: {len(val_indices)} files")
    print(f"    • Test: {len(test_indices)} files")
    
    return train_indices, val_indices, test_indices

def main():
    parser = argparse.ArgumentParser(
        description='TCD-TIMIT File List Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tcd-data-dir', type=str, required=True,
                        help='TCD-TIMIT processed data directory (contains video files)')
    parser.add_argument('--split-ratios', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratios (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Parse split ratios
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.001:
        print("❌ Error: Split ratios must be three numbers that sum to 1.0")
        return 1
    
    split_ratios = {'train': ratios[0], 'val': ratios[1], 'test': ratios[2]}
    
    # Validate input directory
    tcd_data_dir = Path(args.tcd_data_dir).resolve()
    if not tcd_data_dir.exists():
        print(f"❌ Error: TCD-TIMIT data directory not found: {tcd_data_dir}")
        return 1
    
    # Detect crop type and setup paths
    crop_type = detect_crop_type(str(tcd_data_dir))
    crop_suffix = f"_{crop_type}" if crop_type != "lips" else ""
    
    # Find labels directory
    labels_dir = tcd_data_dir.parent / "labels"
    if not labels_dir.exists():
        print(f"❌ Error: Labels directory not found: {labels_dir}")
        print("💡 Make sure you've run step1_prepare_tcd_timit.py first")
        return 1
    
    print(f"📁 Data directory: {tcd_data_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"🎥 Detected crop type: {crop_type}")
    print(f"📊 Split ratios: Train={split_ratios['train']:.1%}, Val={split_ratios['val']:.1%}, Test={split_ratios['test']:.1%}")
    print("-" * 60)
    
    # Load data from CSV files
    fids, labels = load_csv_data(labels_dir, crop_suffix)
    
    if not fids:
        print("❌ Error: No data loaded from CSV files")
        return 1
    
    # Create speaker-based splits
    train_indices, val_indices, test_indices = create_speaker_splits(fids, labels, split_ratios, args.seed)
    
    # Write file.list and label.list
    file_list_path = tcd_data_dir / 'file.list'
    label_list_path = tcd_data_dir / 'label.list'
    
    print("📝 Writing file lists...")
    
    with open(file_list_path, 'w') as f:
        for fid in fids:
            f.write(f"{fid}\n")
    
    with open(label_list_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    # Write split files
    split_files = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    for split_name, indices in split_files.items():
        split_file_path = tcd_data_dir / f'{split_name}.txt'
        with open(split_file_path, 'w') as f:
            for idx in indices:
                # Write file ID (extract from full path)
                fid = fids[idx]
                # Get just the transcript_id part for split files
                transcript_id = fid.split('/')[-1]  # Last part after final /
                f.write(f"{transcript_id}\n")
        print(f"  ✅ Created {split_file_path} with {len(indices)} entries")
    
    print(f"✅ Created file.list with {len(fids)} entries: {file_list_path}")
    print(f"✅ Created label.list with {len(labels)} entries: {label_list_path}")
    
    print("\n🎉 TCD-TIMIT file list generation completed!")
    print(f"📊 Dataset summary:")
    print(f"   • Total files: {len(fids)}")
    print(f"   • Train: {len(train_indices)} files")
    print(f"   • Val: {len(val_indices)} files")  
    print(f"   • Test: {len(test_indices)} files")
    
    return 0

if __name__ == "__main__":
    exit(main())
