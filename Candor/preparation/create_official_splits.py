#!/usr/bin/env python3
"""
Create official train/val/test splits for Candor dataset.

This script creates fixed split files that can be distributed with the dataset
to ensure everyone uses the same train/val/test splits.

Usage:
    python create_official_splits.py \
        --candor-data-dir /path/to/candor_video \
        --output-dir ./splits \
        --split-ratios 0.7,0.15,0.15 \
        --seed 42
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import random


def get_all_sessions(candor_data_dir):
    """Get all session IDs from the processed data directory"""
    data_dir = Path(candor_data_dir)
    
    # Get all session directories
    sessions = [d.name for d in data_dir.iterdir() if d.is_dir()]
    sessions.sort()  # Ensure consistent ordering
    
    return sessions


def create_splits(sessions, split_ratios, seed=42):
    """Create train/val/test splits from sessions"""
    
    # Shuffle sessions with fixed seed for reproducibility
    random.seed(seed)
    shuffled_sessions = sessions.copy()
    random.shuffle(shuffled_sessions)
    
    # Calculate split indices
    n_sessions = len(shuffled_sessions)
    train_end = int(n_sessions * split_ratios['train'])
    valid_end = train_end + int(n_sessions * split_ratios['valid'])
    
    splits = {
        'train': shuffled_sessions[:train_end],
        'valid': shuffled_sessions[train_end:valid_end],
        'test': shuffled_sessions[valid_end:]
    }
    
    return splits


def save_split_files(splits, output_dir, format='txt'):
    """Save split files in specified format"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, session_list in splits.items():
        if format == 'txt':
            # Simple text format (one session ID per line)
            output_file = output_dir / f"candor-{split_name}.id"
            with open(output_file, 'w') as f:
                for session_id in session_list:
                    f.write(f"{session_id}\n")
            print(f"✅ Created: {output_file}")
        
        elif format == 'json':
            # JSON format (more structured)
            output_file = output_dir / f"candor-{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'split': split_name,
                    'sessions': session_list,
                    'count': len(session_list)
                }, f, indent=2)
            print(f"✅ Created: {output_file}")


def print_split_statistics(splits):
    """Print statistics about the splits"""
    
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    total = sum(len(sessions) for sessions in splits.values())
    
    for split_name in ['train', 'valid', 'test']:
        sessions = splits[split_name]
        count = len(sessions)
        percentage = (count / total) * 100
        
        print(f"\n{split_name.upper()}:")
        print(f"  Sessions: {count} ({percentage:.1f}%)")
        print(f"  Session IDs: {', '.join(sessions[:3])}{'...' if count > 3 else ''}")
    
    print(f"\nTotal sessions: {total}")


def verify_splits(splits):
    """Verify that splits don't overlap"""
    
    all_sessions = []
    for sessions in splits.values():
        all_sessions.extend(sessions)
    
    # Check for duplicates
    if len(all_sessions) != len(set(all_sessions)):
        print("⚠️  WARNING: Duplicate sessions found across splits!")
        return False
    
    print("✅ Verification passed: No overlapping sessions")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create official train/val/test splits for Candor dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--candor-data-dir', type=str, required=True,
                        help='Candor processed data directory')
    parser.add_argument('--output-dir', type=str, default='./splits',
                        help='Output directory for split files')
    parser.add_argument('--split-ratios', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratios (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--format', type=str, default='txt', choices=['txt', 'json', 'both'],
                        help='Output format for split files')
    
    args = parser.parse_args()
    
    # Parse split ratios
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.001:
        print("❌ Error: Split ratios must be three numbers that sum to 1.0")
        return 1
    
    split_ratios = {'train': ratios[0], 'valid': ratios[1], 'test': ratios[2]}
    
    print("="*60)
    print("CREATING OFFICIAL CANDOR SPLITS")
    print("="*60)
    print(f"Data directory: {args.candor_data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: {split_ratios}")
    print(f"Random seed: {args.seed}")
    print(f"Format: {args.format}")
    
    # Get all sessions
    print("\n📁 Scanning for sessions...")
    sessions = get_all_sessions(args.candor_data_dir)
    print(f"Found {len(sessions)} sessions")
    
    # Create splits
    print("\n🎲 Creating splits...")
    splits = create_splits(sessions, split_ratios, args.seed)
    
    # Verify splits
    print("\n🔍 Verifying splits...")
    verify_splits(splits)
    
    # Print statistics
    print_split_statistics(splits)
    
    # Save split files
    print("\n💾 Saving split files...")
    if args.format in ['txt', 'both']:
        save_split_files(splits, args.output_dir, format='txt')
    if args.format in ['json', 'both']:
        save_split_files(splits, args.output_dir, format='json')
    
    print("\n" + "="*60)
    print("✅ OFFICIAL SPLITS CREATED SUCCESSFULLY")
    print("="*60)
    print(f"\nSplit files saved to: {args.output_dir}")
    print("\nTo use these splits in preprocessing:")
    print(f"  python step2_generate_file_lists.py \\")
    print(f"    --candor-data-dir {args.candor_data_dir} \\")
    print(f"    --metadata-dir ./metadata \\")
    print(f"    --use-official-splits \\")
    print(f"    --splits-dir {args.output_dir}")
    
    print("\nTo distribute these splits:")
    print(f"  1. Share the files in {args.output_dir}/")
    print(f"  2. Users place them in their splits/ directory")
    print(f"  3. Users run preprocessing with --use-official-splits")
    
    return 0


if __name__ == "__main__":
    exit(main())
