#!/usr/bin/env python3
"""
LRS2 Data Preparation Script
Combines file generation, file movement, and dataset split copying into one step.
"""

import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def detect_crop_type(data_dir):
    """Detect crop type from directory name suffix"""
    dir_name = os.path.basename(data_dir.rstrip('/'))
    
    if dir_name.endswith('_face'):
        return 'face'
    elif dir_name.endswith('_full'):
        return 'full'
    else:
        return 'lips'  # default

def get_file_label(lrs2_root, target_dir):
    """Generate file.list and label.list directly in the target directory"""
    print("🔄 Generating file and label lists...")
    
    # Detect crop type from target directory name
    crop_type = detect_crop_type(target_dir)
    crop_suffix = f"_{crop_type}" if crop_type != "lips" else ""
    print(f"🎥 Detected crop type: {crop_type}")
    
    video_ids_total, labels_total = [], []
    csv_files = {
        'train': f'lrs2_train_transcript_lengths_seg16s{crop_suffix}.csv',
        'val': f'lrs2_val_transcript_lengths_seg16s{crop_suffix}.csv',
        'test': f'lrs2_test_transcript_lengths_seg16s{crop_suffix}.csv'
    }

    for split, csv_file in csv_files.items():
        print(f"  📝 Processing {split} split...")
        # Try multiple locations for labels directory
        csv_file_path = os.path.join(lrs2_root, 'labels', csv_file)
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
            video_id = parts[-2]
            file_name = parts[-1].replace('.mp4', '')

            # Determine if the path belongs to main or pretrain based on file structure
            if 'pretrain' in video_file:
                relative_path = f"pretrain/{video_id}/{file_name}"
                txt_path = os.path.join(lrs2_root, 'lrs2', 'lrs2_text_seg16s', 'pretrain', video_id, f'{file_name}.txt')
                txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                if os.path.exists(txt_path_with_suffix):
                    txt_path = txt_path_with_suffix
                # Also try in parent directory structure
                if not os.path.exists(txt_path):
                    parent_dir = Path(target_dir).parent
                    txt_path = os.path.join(parent_dir, 'lrs2_text_seg16s', 'pretrain', video_id, f'{file_name}.txt')
                    txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                    if os.path.exists(txt_path_with_suffix):
                        txt_path = txt_path_with_suffix
                # Also try in grandparent directory structure
                if not os.path.exists(txt_path):
                    grandparent_dir = Path(target_dir).parent.parent
                    txt_path = os.path.join(grandparent_dir, 'lrs2_text_seg16s', 'pretrain', video_id, f'{file_name}.txt')
                    txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                    if os.path.exists(txt_path_with_suffix):
                        txt_path = txt_path_with_suffix
            else:
                relative_path = f"main/{video_id}/{file_name}"
                txt_path = os.path.join(lrs2_root, 'lrs2', 'lrs2_text_seg16s', 'main', video_id, f'{file_name}.txt')
                # Also try in parent directory structure
                if not os.path.exists(txt_path):
                    parent_dir = Path(target_dir).parent
                    txt_path = os.path.join(parent_dir, 'lrs2_text_seg16s', 'main', video_id, f'{file_name}.txt')
                # Also try in grandparent directory structure
                if not os.path.exists(txt_path):
                    grandparent_dir = Path(target_dir).parent.parent
                    txt_path = os.path.join(grandparent_dir, 'lrs2_text_seg16s', 'main', video_id, f'{file_name}.txt')

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
    
    # List of specific files to copy
    files_to_copy = ["test.txt", "val.txt", "train.txt", "pretrain.txt"]
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
        description='LRS2 Data Preparation - Generate lists and copy dataset splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lrs2-root', type=str, required=True,
                        help='Root directory of LRS2 dataset (contains labels/, lrs2/ folders)')
    parser.add_argument('--target-dir', type=str, required=True,
                        help='Target directory for output files (usually lrs2_video_seg24s folder)')
    
    args = parser.parse_args()
    
    # Validate input directories
    lrs2_root = Path(args.lrs2_root).resolve()
    target_dir = Path(args.target_dir).resolve()
    
    if not lrs2_root.exists():
        print(f"❌ Error: LRS2 root directory not found: {lrs2_root}")
        return
    
    # Check for labels directory in multiple locations
    labels_found = False
    if (lrs2_root / 'labels').exists():
        labels_found = True
        print(f"✅ Found labels directory: {lrs2_root / 'labels'}")
    elif (target_dir.parent / 'labels').exists():
        labels_found = True
        print(f"✅ Found labels directory: {target_dir.parent / 'labels'}")
    elif (target_dir.parent.parent / 'labels').exists():
        labels_found = True
        print(f"✅ Found labels directory: {target_dir.parent.parent / 'labels'}")
    
    if not labels_found:
        print(f"❌ Error: Labels directory not found in:")
        print(f"  - {lrs2_root / 'labels'}")
        print(f"  - {target_dir.parent / 'labels'}")
        print(f"  - {target_dir.parent.parent / 'labels'}")
        return
    
    print(f"🚀 Starting LRS2 data preparation...")
    print(f"📁 Source: {lrs2_root}")
    print(f"📁 Target: {target_dir}")
    print("-" * 60)
    
    try:
        # Step 1: Generate file and label lists
        file_list, label_list = get_file_label(str(lrs2_root), str(target_dir))
        
        # Step 2: Copy dataset split files
        copied_splits = copy_dataset_splits(str(lrs2_root), str(target_dir))
        
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
