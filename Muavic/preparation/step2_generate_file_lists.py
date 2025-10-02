#!/usr/bin/env python3
"""
MuAViC Step 2: Generate File Lists

This script generates train/valid/test file lists from the processed MuAViC data.
Unlike TCD-TIMIT, MuAViC already has predefined splits from the original dataset.

Usage:
    python step2_generate_file_lists.py \
        --muavic-data-dir /path/to/processed/muavic/muavic_video \
        --language en
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description='Generate file lists for MuAViC dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--muavic-data-dir', type=str, required=True,
                       help='MuAViC processed data directory (contains video files)')
    parser.add_argument('--language', type=str, required=True,
                       help='Language code (en, es, fr, pt, it, el, ar, de, ru)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.muavic_data_dir)
    language = args.language
    
    print(f"Generating file lists for {language}...")
    
    # Check if language directory exists
    lang_dir = data_dir / language
    if not lang_dir.exists():
        print(f"Error: Language directory not found: {lang_dir}")
        return 1
    
    # Generate file lists for each split
    for split in ['train', 'valid', 'test']:
        split_dir = lang_dir / split
        
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            continue
        
        # Find all video files
        video_files = sorted(split_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"Warning: No video files found in {split_dir}")
            continue
        
        # Create file list (relative paths from data_dir)
        file_list_path = data_dir / f"{language}_{split}.txt"
        
        with open(file_list_path, 'w') as f:
            for video_file in video_files:
                # Get relative path from data_dir
                rel_path = video_file.relative_to(data_dir)
                # Remove .mp4 extension for file ID
                file_id = str(rel_path.with_suffix(''))
                f.write(f"{file_id}\n")
        
        print(f"✅ Created {split} file list: {file_list_path} ({len(video_files)} files)")
    
    print(f"\n✅ File lists generated successfully!")
    print(f"   Location: {data_dir}")
    print(f"   Files: {language}_train.txt, {language}_valid.txt, {language}_test.txt")
    
    return 0

if __name__ == "__main__":
    exit(main())
