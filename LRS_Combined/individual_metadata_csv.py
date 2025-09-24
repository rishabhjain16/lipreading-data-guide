#!/usr/bin/env python3
"""
Extract Dataset CSV Files
Creates separate train/val/test CSV files for LRS2 or LRS3 from the combined dataset.
"""

import os
import argparse
import pandas as pd
from pathlib import Path

def extract_dataset_csv(combined_dir, dataset_name):
    """Extract CSV files for a specific dataset (lrs2 or lrs3) from combined dataset"""
    print(f"🔄 Extracting {dataset_name.upper()} CSV files from combined dataset...")
    
    combined_dir = Path(combined_dir)
    
    # Validate combined directory structure
    labels_dir = combined_dir / 'labels'
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Output will be in the same labels directory
    output_dir = labels_dir
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"  📝 Processing {split} split...")
        
        # Find the combined CSV file
        combined_csv = labels_dir / f"lrs_combined_{split}_transcript_lengths_seg16s.csv"
        
        if not combined_csv.exists():
            print(f"    ⚠️  Combined CSV not found: {combined_csv}")
            continue
        
        # Read the combined CSV
        df = pd.read_csv(combined_csv, header=None)
        
        # Filter rows for the specific dataset
        dataset_rows = []
        
        for _, row in df.iterrows():
            video_path = row[1]  # Assuming column 1 is the video path
            
            # Check if this row belongs to the requested dataset
            if f"/{dataset_name}/" in video_path:
                # Keep the same format: lrs_combined as dataset name, path within that folder
                dataset_rows.append(row)
        
        if dataset_rows:
            # Create output CSV
            output_csv = output_dir / f"{dataset_name}_{split}_transcript_lengths_seg16s.csv"
            output_df = pd.DataFrame(dataset_rows)
            output_df.to_csv(output_csv, header=False, index=False)
            
            print(f"    ✅ Created {output_csv.name} with {len(dataset_rows)} entries")
        else:
            print(f"    ⚠️  No {dataset_name.upper()} entries found in {split} split")
    

    
    return output_dir

def extract_both_datasets(combined_dir):
    """Extract CSV files for both LRS2 and LRS3 datasets"""
    print("🔄 Extracting both LRS2 and LRS3 CSV files...")
    
    # Extract LRS2
    extract_dataset_csv(combined_dir, 'lrs2')
    
    # Extract LRS3  
    extract_dataset_csv(combined_dir, 'lrs3')
    
    print("  ✅ Extracted CSV files for both datasets")

def main():
    parser = argparse.ArgumentParser(
        description='Extract separate CSV files for LRS2 or LRS3 from combined dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--combined-dir', type=str, required=True,
                        help='Path to the combined dataset directory')
    parser.add_argument('--dataset', type=str, choices=['lrs2', 'lrs3', 'both'], default='both',
                        help='Dataset to extract (lrs2, lrs3, or both). Default: both')
    
    args = parser.parse_args()
    
    # Validate input directory
    combined_dir = Path(args.combined_dir).resolve()
    
    if not combined_dir.exists():
        print(f"❌ Error: Combined dataset directory not found: {combined_dir}")
        return 1
    
    print(f"🚀 Extracting CSV files...")
    print(f"� Cotmbined Dataset: {combined_dir}")
    print(f"🎯 Target Dataset: {args.dataset.upper()}")
    print("-" * 60)
    
    try:
        if args.dataset == 'both':
            extract_both_datasets(str(combined_dir))
        else:
            extract_dataset_csv(str(combined_dir), args.dataset)
        
        labels_dir = combined_dir / 'labels'
        print("-" * 60)
        print("🎉 CSV extraction completed successfully!")
        print(f"📊 CSV files created in: {labels_dir}")
        print("📋 Files created:")
        if args.dataset in ['lrs2', 'both']:
            print("  • lrs2_train_transcript_lengths_seg16s.csv")
            print("  • lrs2_val_transcript_lengths_seg16s.csv") 
            print("  • lrs2_test_transcript_lengths_seg16s.csv")
        if args.dataset in ['lrs3', 'both']:
            print("  • lrs3_train_transcript_lengths_seg16s.csv")
            print("  • lrs3_val_transcript_lengths_seg16s.csv")
            print("  • lrs3_test_transcript_lengths_seg16s.csv")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())