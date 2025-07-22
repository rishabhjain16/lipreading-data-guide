#!/usr/bin/env python3
"""
Minimal Step 2: Generate file.list and label.list from CSV files
"""

import os
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate file.list and label.list from LRS3 CSV files")
    parser.add_argument('--lrs3-data-dir', type=str, required=True,
                        help='LRS3 processed data directory (lrs3_video_seg16s[_face|_full])')
    parser.add_argument('--labels-dir', type=str, default=None,
                        help='Directory containing CSV files (default: auto-detect)')
    
    args = parser.parse_args()
    
    lrs3_data_dir = Path(args.lrs3_data_dir)
    
    # Detect crop type from directory name
    crop_type = "lips"  # default
    if "_face" in lrs3_data_dir.name:
        crop_type = "face"
    elif "_full" in lrs3_data_dir.name:
        crop_type = "full"
    
    # Auto-detect labels directory if not provided
    if args.labels_dir is None:
        possible_labels_dirs = [
            lrs3_data_dir.parent / 'labels',
            lrs3_data_dir.parent.parent / 'labels'
        ]
        
        labels_dir = None
        for ld in possible_labels_dirs:
            if ld.exists():
                labels_dir = ld
                break
        
        if labels_dir is None:
            print(f"❌ Error: Could not find labels directory")
            return 1
    else:
        labels_dir = Path(args.labels_dir)
    
    print(f"📁 Data directory: {lrs3_data_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"🎥 Detected crop type: {crop_type}")
    
    # Process CSV files (with crop type suffix for face/full)
    crop_suffix = f"_{crop_type}" if crop_type != "lips" else ""
    csv_files = {
        'train': f'lrs3_train_transcript_lengths_seg16s{crop_suffix}.csv',
        'val': f'lrs3_val_transcript_lengths_seg16s{crop_suffix}.csv',
        'test': f'lrs3_test_transcript_lengths_seg16s{crop_suffix}.csv'
    }
    
    fids, labels = [], []
    
    for split, csv_file in csv_files.items():
        csv_path = labels_dir / csv_file
        if not csv_path.exists():
            print(f"⚠️  {csv_file} not found, skipping {split}")
            continue
            
        print(f"📝 Processing {split}: {csv_file}")
        df = pd.read_csv(csv_path, header=None)
        
        for _, row in df.iterrows():
            video_path = row[1]    # path to video file
            token_ids = row[3]     # tokenized text
            
            # Use video path without .mp4 extension as file ID
            file_id = video_path.replace('.mp4', '')
            fids.append(file_id)
            
            # Extract the actual text from the corresponding text file
            # Convert video path to corresponding text file path
            # e.g., lrs3_video_seg16s_full/test/xTkKSJSqUSI/00009.mp4 -> test/xTkKSJSqUSI/00009.txt
            txt_file_path = video_path.replace('.mp4', '.txt')
            
            # Remove the video directory prefix (crop-type aware) to get the relative path
            video_dir_prefix = f'lrs3_video_seg16s{crop_suffix}/'
            if txt_file_path.startswith(video_dir_prefix):
                txt_file_path = txt_file_path[len(video_dir_prefix):]
            
            # Look in the lrs3_text_seg16s directory (sibling to lrs3_video_seg16s)
            text_dir = lrs3_data_dir.parent / 'lrs3_text_seg16s'
            full_txt_path = text_dir / txt_file_path
            
            actual_text = None
            if full_txt_path.exists():
                try:
                    with open(full_txt_path, 'r') as f:
                        line = f.readline().strip()
                        # The text file should contain the transcript directly
                        actual_text = line
                except Exception as e:
                    print(f"⚠️  Error reading {full_txt_path}: {e}")
            
            if actual_text is None:
                print(f"⚠️  Could not find transcript at {full_txt_path}")
                actual_text = f"MISSING_TRANSCRIPT_{len(labels)}"
            
            labels.append(actual_text)
    
    # Write file.list and label.list
    file_list_path = lrs3_data_dir / 'file.list'
    label_list_path = lrs3_data_dir / 'label.list'
    
    os.makedirs(lrs3_data_dir, exist_ok=True)
    
    with open(file_list_path, 'w') as f:
        for fid in fids:
            f.write(f"{fid}\n")
    
    with open(label_list_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"✅ Created file.list with {len(fids)} entries: {file_list_path}")
    print(f"✅ Created label.list with {len(labels)} entries: {label_list_path}")
    
    # Create split files for step 3
    val_ids, test_ids = set(), set()
    
    # Process each CSV to determine which files belong to which split
    for split_name, csv_filename in csv_files.items():
        csv_path = labels_dir / csv_filename
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path, names=['dataset', 'video_path', 'length', 'transcript'])
        for _, row in df.iterrows():
            video_path = row['video_path']
            file_id = video_path.replace('.mp4', '')
            
            # Remove lrs3_video_seg16s/ prefix to get clean ID
            clean_id = file_id
            if clean_id.startswith('lrs3_video_seg16s/'):
                clean_id = clean_id[len('lrs3_video_seg16s/'):]
            
            # Further clean the ID by removing directory prefix
            if 'trainval/' in clean_id:
                final_id = clean_id.replace('trainval/', '')
            elif 'test/' in clean_id:
                final_id = clean_id.replace('test/', '')
            elif 'pretrain/' in clean_id:
                final_id = clean_id.replace('pretrain/', '')
            else:
                final_id = clean_id
            
            if split_name == 'val':
                val_ids.add(final_id)
            elif split_name == 'test':
                test_ids.add(final_id)
    
    # Write split files
    if val_ids:
        val_file_path = lrs3_data_dir / 'val.txt'
        with open(val_file_path, 'w') as f:
            for vid in sorted(val_ids):
                f.write(f"{vid}\n")
        print(f"✅ Created val.txt with {len(val_ids)} entries: {val_file_path}")
    
    if test_ids:
        test_file_path = lrs3_data_dir / 'test.txt'
        with open(test_file_path, 'w') as f:
            for vid in sorted(test_ids):
                f.write(f"{vid}\n")
        print(f"✅ Created test.txt with {len(test_ids)} entries: {test_file_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
