#!/usr/bin/env python3
"""
LRS2 + LRS3 Dataset Combiner
Combines prepared LRS2 and LRS3 datasets with proper CSV path updates.
"""

import os
import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_directory_with_progress(src, dst, desc):
    """Copy directory with progress bar"""
    if not os.path.exists(src):
        print(f"⚠️  Source directory not found: {src}")
        return
    
    print(f"📁 Copying {desc}...")
    
    # Count total files for progress
    total_files = sum([len(files) for r, d, files in os.walk(src)])
    
    if total_files == 0:
        print(f"  ⚠️  No files found in {src}")
        return
    
    # Create destination directory
    os.makedirs(dst, exist_ok=True)
    
    # Copy with progress
    copied_files = 0
    with tqdm(total=total_files, desc=desc) as pbar:
        for root, dirs, files in os.walk(src):
            # Create corresponding directories in destination
            rel_path = os.path.relpath(root, src)
            dst_dir = os.path.join(dst, rel_path) if rel_path != '.' else dst
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy files
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)
                shutil.copy2(src_file, dst_file)
                copied_files += 1
                pbar.update(1)
    
    print(f"  ✅ Copied {copied_files} files")

def combine_csv_files(lrs2_labels_dir, lrs3_labels_dir, output_labels_dir):
    """Combine CSV files with updated paths"""
    print("📋 Combining CSV files with updated paths...")
    
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Process each split type
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"  📝 Processing {split} split...")
        
        combined_data = []
        
        # Process LRS2 CSV
        lrs2_csv_pattern = f"lrs2_{split}_transcript_lengths_seg16s*.csv"
        lrs2_csv_files = list(Path(lrs2_labels_dir).glob(lrs2_csv_pattern))
        
        if lrs2_csv_files:
            lrs2_csv_file = lrs2_csv_files[0]  # Take the first match
            print(f"    📄 Reading LRS2: {lrs2_csv_file.name}")
            
            df = pd.read_csv(lrs2_csv_file, header=None)
            for _, row in df.iterrows():
                # Update the video path to start with lrs_combined
                original_path = row[1]  # Assuming column 1 is the video path
                if original_path.startswith('lrs2_video_seg16s/'):
                    # Remove lrs2_video_seg16s/ and add lrs_combined_video_seg16s/lrs2/
                    clean_path = original_path.replace('lrs2_video_seg16s/', '')
                    updated_path = f"lrs_combined/lrs_combined_video_seg16s/lrs2/{clean_path}"
                else:
                    updated_path = f"lrs_combined/lrs_combined_video_seg16s/lrs2/{original_path}"
                
                # Create new row with updated path
                new_row = row.copy()
                new_row[1] = updated_path
                combined_data.append(new_row)
        
        # Process LRS3 CSV
        lrs3_csv_pattern = f"lrs3_{split}_transcript_lengths_seg16s*.csv"
        lrs3_csv_files = list(Path(lrs3_labels_dir).glob(lrs3_csv_pattern))
        
        if lrs3_csv_files:
            lrs3_csv_file = lrs3_csv_files[0]  # Take the first match
            print(f"    📄 Reading LRS3: {lrs3_csv_file.name}")
            
            df = pd.read_csv(lrs3_csv_file, header=None)
            for _, row in df.iterrows():
                # Update the video path to start with lrs_combined
                original_path = row[1]  # Assuming column 1 is the video path
                if original_path.startswith('lrs3_video_seg16s/'):
                    # Remove lrs3_video_seg16s/ and add lrs_combined_video_seg16s/lrs3/
                    clean_path = original_path.replace('lrs3_video_seg16s/', '')
                    updated_path = f"lrs_combined/lrs_combined_video_seg16s/lrs3/{clean_path}"
                else:
                    updated_path = f"lrs_combined/lrs_combined_video_seg16s/lrs3/{original_path}"
                
                # Create new row with updated path
                new_row = row.copy()
                new_row[1] = updated_path
                combined_data.append(new_row)
        
        # Save combined CSV
        if combined_data:
            combined_df = pd.DataFrame(combined_data)
            output_csv = os.path.join(output_labels_dir, f"lrs_combined_{split}_transcript_lengths_seg16s.csv")
            combined_df.to_csv(output_csv, header=False, index=False)
            print(f"    ✅ Saved {len(combined_data)} entries to lrs_combined_{split}_transcript_lengths_seg16s.csv")
        else:
            print(f"    ⚠️  No data found for {split} split")

def combine_datasets(lrs2_root, lrs3_root, output_dir):
    """Combine LRS2 and LRS3 datasets with proper structure"""
    print("🔄 Combining LRS2 and LRS3 datasets...")
    
    lrs2_root = Path(lrs2_root)
    lrs3_root = Path(lrs3_root)
    output_dir = Path(output_dir)
    
    # Validate structure
    lrs2_labels = lrs2_root / 'labels'
    lrs2_video = lrs2_root / 'lrs2' / 'lrs2_video_seg16s'
    lrs2_text = lrs2_root / 'lrs2' / 'lrs2_text_seg16s'
    
    lrs3_labels = lrs3_root / 'labels'
    lrs3_video = lrs3_root / 'lrs3' / 'lrs3_video_seg16s'
    lrs3_text = lrs3_root / 'lrs3' / 'lrs3_text_seg16s'
    
    if not lrs2_video.exists():
        raise ValueError(f"LRS2 video directory not found: {lrs2_video}")
    if not lrs3_video.exists():
        raise ValueError(f"LRS3 video directory not found: {lrs3_video}")
    
    print(f"📁 LRS2 structure validated: {lrs2_root}")
    print(f"📁 LRS3 structure validated: {lrs3_root}")
    
    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_labels = output_dir / 'labels'
    output_lrs_combined = output_dir / 'lrs_combined'
    output_video = output_lrs_combined / 'lrs_combined_video_seg16s'
    output_text = output_lrs_combined / 'lrs_combined_text_seg16s'
    
    output_labels.mkdir(exist_ok=True)
    output_lrs_combined.mkdir(exist_ok=True)
    output_video.mkdir(exist_ok=True)
    output_text.mkdir(exist_ok=True)
    
    # Copy video files
    copy_directory_with_progress(str(lrs2_video), str(output_video / 'lrs2'), "LRS2 videos")
    copy_directory_with_progress(str(lrs3_video), str(output_video / 'lrs3'), "LRS3 videos")
    
    # Copy text files
    if lrs2_text.exists():
        copy_directory_with_progress(str(lrs2_text), str(output_text / 'lrs2'), "LRS2 texts")
    if lrs3_text.exists():
        copy_directory_with_progress(str(lrs3_text), str(output_text / 'lrs3'), "LRS3 texts")
    
    # Combine CSV files with updated paths
    if lrs2_labels.exists() and lrs3_labels.exists():
        combine_csv_files(str(lrs2_labels), str(lrs3_labels), str(output_labels))
    
    # Create info file
    create_info_file(lrs2_root, lrs3_root, output_dir)
    
    return output_dir

def create_info_file(lrs2_root, lrs3_root, output_dir):
    """Create an info file about the combined dataset"""
    info_file = output_dir / 'dataset_info.txt'
    
    with open(info_file, 'w') as f:
        f.write("# Combined LRS2 + LRS3 Dataset\n")
        f.write("# Created by combining prepared datasets with proper path updates\n\n")
        f.write(f"Original LRS2 location: {lrs2_root}\n")
        f.write(f"Original LRS3 location: {lrs3_root}\n\n")
        f.write("Combined structure:\n")
        f.write("├── labels/\n")
        f.write("│   ├── lrs_combined_train_transcript_lengths_seg16s.csv\n")
        f.write("│   ├── lrs_combined_val_transcript_lengths_seg16s.csv\n")
        f.write("│   └── lrs_combined_test_transcript_lengths_seg16s.csv\n")
        f.write("└── lrs_combined/\n")
        f.write("    ├── lrs_combined_video_seg16s/\n")
        f.write("    │   ├── lrs2/ (LRS2 video files)\n")
        f.write("    │   └── lrs3/ (LRS3 video files)\n")
        f.write("    └── lrs_combined_text_seg16s/\n")
        f.write("        ├── lrs2/ (LRS2 text files)\n")
        f.write("        └── lrs3/ (LRS3 text files)\n\n")
    
    print(f"  ✅ Created info file: {info_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Combine prepared LRS2 and LRS3 datasets with proper CSV path updates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lrs2-root', type=str, required=True,
                        help='Root directory of prepared LRS2 dataset')
    parser.add_argument('--lrs3-root', type=str, required=True,
                        help='Root directory of prepared LRS3 dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for combined dataset')
    
    args = parser.parse_args()
    
    # Validate input directories
    lrs2_root = Path(args.lrs2_root).resolve()
    lrs3_root = Path(args.lrs3_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not lrs2_root.exists():
        print(f"❌ Error: LRS2 root directory not found: {lrs2_root}")
        return 1
    
    if not lrs3_root.exists():
        print(f"❌ Error: LRS3 root directory not found: {lrs3_root}")
        return 1
    
    print(f"🚀 Combining LRS2 and LRS3 datasets...")
    print(f"📁 LRS2 Root: {lrs2_root}")
    print(f"📁 LRS3 Root: {lrs3_root}")
    print(f"📁 Output: {output_dir}")
    print("-" * 60)
    
    try:
        # Combine the datasets
        combined_dir = combine_datasets(str(lrs2_root), str(lrs3_root), str(output_dir))
        
        print("-" * 60)
        print("🎉 Dataset combination completed successfully!")
        print(f"📊 Combined dataset created in: {combined_dir}")
        print("📋 Key features:")
        print("  • Video/audio files copied with lrs2/ and lrs3/ prefixes")
        print("  • Text files copied with proper structure")
        print("  • CSV files combined with updated paths")
        print("  • Ready for training with correct path references")
        
    except Exception as e:
        print(f"❌ Error during combination: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())