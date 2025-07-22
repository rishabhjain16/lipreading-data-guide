#!/usr/bin/env python3
"""
LRS3 Data Preparation Script
Combines file generation, file movement, and dataset split copying into one step.
"""

import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def get_file_label(lrs3_root, target_dir):
    """Generate file.list and label.list directly in the target directory"""
    print("🔄 Generating file and label lists...")
    
    video_ids_total, labels_total = [], []
    csv_files = {
        'train': 'lrs3_train_transcript_lengths_seg16s.csv',
        'val': 'lrs3_val_transcript_lengths_seg16s.csv',
        'test': 'lrs3_test_transcript_lengths_seg16s.csv'
    }

    for split, csv_file in csv_files.items():
        print(f"  📝 Processing {split} split...")
        # Try multiple locations for labels directory
        csv_file_path = os.path.join(lrs3_root, 'labels', csv_file)
        if not os.path.exists(csv_file_path):
            # Check if labels is in the parent directory of target_dir
            parent_dir = Path(target_dir).parent
            csv_file_path = os.path.join(parent_dir, 'labels', csv_file)
        if not os.path.exists(csv_file_path):
            # Check if labels is in the grandparent directory of target_dir
            grandparent_dir = Path(target_dir).parent.parent
            csv_file_path = os.path.join(grandparent_dir, 'labels', csv_file)
        
        if not os.path.exists(csv_file_path):
            print(f"  ⚠️  Warning: {csv_file} not found, skipping {split}")
            continue
            
        df = pd.read_csv(csv_file_path, header=None)

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {split}"):
            video_file = row[1]
            parts = video_file.split('/')
            
            # LRS3 has different structure: trainval/speaker_id/video_id.mp4 or pretrain/...
            if 'pretrain' in video_file:
                speaker_id = parts[-2]
                file_name = parts[-1].replace('.mp4', '')
                relative_path = f"pretrain/{speaker_id}/{file_name}"
                txt_path = os.path.join(lrs3_root, 'lrs3', 'lrs3_text_seg16s', 'pretrain', speaker_id, f'{file_name}.txt')
                txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                if os.path.exists(txt_path_with_suffix):
                    txt_path = txt_path_with_suffix
                # Also try in parent directory structure
                if not os.path.exists(txt_path):
                    parent_dir = Path(target_dir).parent
                    txt_path = os.path.join(parent_dir, 'lrs3_text_seg16s', 'pretrain', speaker_id, f'{file_name}.txt')
                    txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                    if os.path.exists(txt_path_with_suffix):
                        txt_path = txt_path_with_suffix
                # Also try in grandparent directory structure
                if not os.path.exists(txt_path):
                    grandparent_dir = Path(target_dir).parent.parent
                    txt_path = os.path.join(grandparent_dir, 'lrs3_text_seg16s', 'pretrain', speaker_id, f'{file_name}.txt')
                    txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                    if os.path.exists(txt_path_with_suffix):
                        txt_path = txt_path_with_suffix
            else:
                # trainval or test files
                speaker_id = parts[-2]
                file_name = parts[-1].replace('.mp4', '')
                subset_name = "trainval" if "trainval" in video_file else "test"
                relative_path = f"{subset_name}/{speaker_id}/{file_name}"
                txt_path = os.path.join(lrs3_root, 'lrs3', 'lrs3_text_seg16s', subset_name, speaker_id, f'{file_name}.txt')
                # Also try in parent directory structure
                if not os.path.exists(txt_path):
                    parent_dir = Path(target_dir).parent
                    txt_path = os.path.join(parent_dir, 'lrs3_text_seg16s', subset_name, speaker_id, f'{file_name}.txt')
                # Also try in grandparent directory structure
                if not os.path.exists(txt_path):
                    grandparent_dir = Path(target_dir).parent.parent
                    txt_path = os.path.join(grandparent_dir, 'lrs3_text_seg16s', subset_name, speaker_id, f'{file_name}.txt')

            # Check if the text file exists and read the label
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as txt_file:
                    label = txt_file.readlines()[0].strip()
                    labels_total.append(label)
                    video_ids_total.append(relative_path)

    # Save files directly to target directory
    os.makedirs(target_dir, exist_ok=True)
    video_id_fn = os.path.join(target_dir, 'file.list')
    label_fn = os.path.join(target_dir, 'label.list')
    
    with open(video_id_fn, 'w') as fo:
        fo.write('\n'.join(video_ids_total)+'\n')
    with open(label_fn, 'w') as fo:
        fo.write('\n'.join(labels_total)+'\n')
    
    print(f"  ✅ Generated {len(video_ids_total)} entries")
    print(f"  📁 Saved to: {video_id_fn}")
    print(f"  📁 Saved to: {label_fn}")
    return video_id_fn, label_fn

def copy_dataset_splits(source_dir, target_dir):
    """Copy dataset split files to target directory"""
    print("🔄 Copying dataset split files...")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # List of specific files to copy for LRS3 (different from LRS2)
    files_to_copy = ["test.txt", "val.txt", "train.txt", "pretrain.txt", "trainval.txt"]
    copied_files = []
    
    # Copy each file if it exists
    for filename in files_to_copy:
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        
        # Also try in parent directory of target_dir if not found in source_dir
        if not os.path.exists(source_file):
            parent_dir = Path(target_dir).parent
            source_file = os.path.join(parent_dir, filename)
        # Also try in grandparent directory of target_dir
        if not os.path.exists(source_file):
            grandparent_dir = Path(target_dir).parent.parent
            source_file = os.path.join(grandparent_dir, filename)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            copied_files.append(filename)
            print(f"  ✅ Copied: {filename}")
        else:
            print(f"  ⚠️  Warning: {filename} not found in {source_dir} or {Path(target_dir).parent}")
    
    print(f"  📁 Copied {len(copied_files)} split files to: {target_dir}")
    return copied_files

def main():
    parser = argparse.ArgumentParser(
        description='LRS3 Data Preparation - Generate lists and copy dataset splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lrs3-root', type=str, required=True,
                        help='Root directory of LRS3 dataset (contains labels/, lrs3/ folders)')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Target directory for output files (usually lrs3_video_seg16s folder)')
    
    args = parser.parse_args()
    
    # Validate input directories
    lrs3_root = Path(args.lrs3_root).resolve()
    target_dir = Path(args.target_dir).resolve()
    
    if not lrs3_root.exists():
        print(f"❌ Error: LRS3 root directory not found: {lrs3_root}")
        return
    
    # Check for labels directory in multiple locations
    labels_found = False
    if (lrs3_root / 'labels').exists():
        labels_found = True
        print(f"✅ Found labels directory: {lrs3_root / 'labels'}")
    elif (target_dir.parent / 'labels').exists():
        labels_found = True
        print(f"✅ Found labels directory: {target_dir.parent / 'labels'}")
    elif (target_dir.parent.parent / 'labels').exists():
        labels_found = True
        print(f"✅ Found labels directory: {target_dir.parent.parent / 'labels'}")
    
    if not labels_found:
        print(f"❌ Error: Labels directory not found in:")
        print(f"  - {lrs3_root / 'labels'}")
        print(f"  - {target_dir.parent / 'labels'}")
        print(f"  - {target_dir.parent.parent / 'labels'}")
        return
    
    print(f"🚀 Starting LRS3 data preparation...")
    print(f"📁 Source: {lrs3_root}")
    print(f"📁 Target: {target_dir}")
    print("-" * 60)
    
    try:
        # Step 1: Generate file and label lists
        file_list, label_list = get_file_label(str(lrs3_root), str(target_dir))
        
        # Step 2: Copy dataset split files
        copied_splits = copy_dataset_splits(str(lrs3_root), str(target_dir))
        
        print("-" * 60)
        print("🎉 Data preparation completed successfully!")
        print(f"📊 Generated files:")
        print(f"   • file.list: {file_list}")
        print(f"   • label.list: {label_list}")
        if copied_splits:
            print(f"   • Split files: {', '.join(copied_splits)}")
        
    except Exception as e:
        print(f"❌ Error during preparation: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
