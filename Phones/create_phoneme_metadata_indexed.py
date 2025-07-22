#!/usr/bin/env python3
"""
LRS3 Phoneme Metadata Generator (Index-based Dictionary)

This script converts word-level transcripts to phoneme-level transcripts using G2P (Grapheme-to-Phoneme) 
conversion. It creates index-based dictionaries where each phoneme is mapped to a unique index starting from 0.

Usage:
    python create_phoneme_metadata_indexed.py --metadata-dir /path/to/lrs3/metadata [options]

The script will:
1. Convert word transcripts (.wrd files) to phoneme transcripts (.phn files)
2. Create an index-based phoneme dictionary (dict.phn.txt) with format: <token> <index>
3. Generate statistics about the phoneme conversion process

Author: LRS3 Data Preparation Pipeline
"""

import os
import argparse
from pathlib import Path
from g2p_en import G2p
import tqdm
import re
from collections import Counter

def remove_stress_markers(phoneme):
    """Remove stress markers (digits) from phonemes"""
    return re.sub(r'[0-9]', '', phoneme)

def convert_words_to_phonemes(metadata_dir, splits, remove_stress=True, remove_punctuation=True):
    """
    Convert word transcripts to phoneme transcripts
    
    Args:
        metadata_dir (str): Directory containing .wrd files from LRS3 preprocessing
        splits (list): List of dataset splits to process
        remove_stress (bool): Whether to remove stress markers from phonemes
        remove_punctuation (bool): Whether to remove punctuation from phonemes
    
    Returns:
        dict: Phoneme counts for statistics
    """
    print("🔄 Initializing G2P converter...")
    g2p = G2p()
    
    # Set of all phonemes encountered
    phoneme_counts = Counter()
    punctuation = ["'", ",", "."]
    
    # Process each split
    for split in splits:
        input_file = Path(metadata_dir) / f"{split}.wrd"
        output_file = Path(metadata_dir) / f"{split}.phn"
        
        if not input_file.exists():
            print(f"⚠️  Warning: {input_file} does not exist, skipping {split}")
            continue
            
        print(f"📝 Processing {split} split: {input_file}")
        total_sequences = 0
        total_phonemes = 0
        
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in tqdm.tqdm(f_in, desc=f"Converting {split}"):
                words = line.strip().split()
                if not words:  # Skip empty lines
                    f_out.write('\n')
                    continue
                    
                phonemes = []
                for word in words:
                    # Convert word to phonemes
                    word_phonemes = g2p(word)
                    
                    # Remove stress markers if requested
                    if remove_stress:
                        word_phonemes = [remove_stress_markers(p) for p in word_phonemes]
                    
                    # Filter out punctuation if requested
                    if remove_punctuation:
                        word_phonemes = [p for p in word_phonemes if p not in punctuation]
                    
                    # Filter out empty strings and whitespace-only phonemes
                    word_phonemes = [p for p in word_phonemes if p and p.strip()]
                    
                    # Update phoneme counts
                    phoneme_counts.update(word_phonemes)
                    phonemes.extend(word_phonemes)
                
                # Write phoneme sequence to output file
                f_out.write(' '.join(phonemes) + '\n')
                total_sequences += 1
                total_phonemes += len(phonemes)
        
        print(f"  ✅ Created {output_file}")
        print(f"     📊 {total_sequences:,} sequences, {total_phonemes:,} phonemes")
    
    return phoneme_counts

def create_phoneme_dictionary_indexed(metadata_dir, phoneme_counts):
    """
    Create index-based phoneme dictionary in the format: <token> <index>
    
    Args:
        metadata_dir (str): Directory to save the dictionary
        phoneme_counts (Counter): Phoneme frequency counts (used for sorting)
    """
    # Write dictionary in index format: <token> <index>
    dict_file = Path(metadata_dir) / "dict.phn.txt"
    print(f"📋 Creating index-based phoneme dictionary: {dict_file}")
    
    # Sort phonemes by frequency (descending) then alphabetically for consistent ordering
    sorted_phonemes = sorted(phoneme_counts.items(), key=lambda x: (-x[1], x[0]))
    
    with open(dict_file, 'w') as f:
        # Add special blank token at index 0
        f.write("<blank> 0\n")
        
        # Add phonemes starting from index 1
        for idx, (phoneme, count) in enumerate(sorted_phonemes, 1):
            if phoneme and phoneme.strip() and phoneme != "<blank>":  # Skip empty, whitespace, and duplicate blank
                f.write(f"{phoneme} {idx}\n")
    
    total_phonemes = sum(phoneme_counts.values())
    unique_phonemes = len(phoneme_counts)
    
    print(f"  ✅ Dictionary created with {unique_phonemes + 1} tokens (including <blank>)")
    print(f"     📊 Total phoneme tokens in data: {total_phonemes:,}")
    print(f"     📊 Unique phonemes: {unique_phonemes}")
    print(f"     📊 Index range: 0 to {unique_phonemes}")
    
    # Show top 10 most frequent phonemes with their indices
    print("\n📈 Top 10 most frequent phonemes:")
    print("  Rank Token      Count      Index")
    print("  ---- --------  --------   -----")
    print(f"   1.  <blank>   (special)    0")
    
    for i, (phoneme, count) in enumerate(sorted_phonemes[:9], 2):
        print(f"  {i:2d}.  {phoneme:<8}  {count:8,}    {i-1}")
    
    return dict_file

def validate_files(metadata_dir, splits):
    """Validate that all required input files exist"""
    missing_files = []
    
    for split in splits:
        wrd_file = Path(metadata_dir) / f"{split}.wrd"
        if not wrd_file.exists():
            missing_files.append(str(wrd_file))
    
    if missing_files:
        print("❌ Error: Missing required .wrd files:")
        for file in missing_files:
            print(f"   {file}")
        print("\n💡 Make sure you've run the LRS3 preprocessing pipeline (step3_metadata_prep.py) first.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert LRS3 word transcripts to phoneme transcripts with index-based dictionary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with LRS3 metadata directory
    python create_phoneme_metadata_indexed.py --metadata-dir /path/to/lrs3/metadata
    
    # Process only train and test splits
    python create_phoneme_metadata_indexed.py --metadata-dir /path/to/lrs3/metadata --splits train,test
    
    # Keep stress markers and punctuation
    python create_phoneme_metadata_indexed.py --metadata-dir /path/to/lrs3/metadata --keep-stress --keep-punctuation

Output files:
    - train.phn, valid.phn, test.phn: Phoneme transcripts
    - dict.phn.txt: Index-based phoneme dictionary (<token> <index>)
      Format: <blank> 0, AH 1, T 2, etc.
        """
    )
    
    parser.add_argument(
        "--metadata-dir", 
        required=True, 
        help="Directory containing .wrd files from LRS3 step3_metadata_prep.py output"
    )
    parser.add_argument(
        "--splits", 
        default="train,valid,test", 
        help="Comma-separated list of dataset splits to process (default: train,valid,test)"
    )
    parser.add_argument(
        "--keep-stress", 
        action="store_true", 
        help="Keep stress markers (0,1,2) in phonemes (default: remove them)"
    )
    parser.add_argument(
        "--keep-punctuation", 
        action="store_true", 
        help="Keep punctuation symbols in phoneme set (default: remove them)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        print(f"❌ Error: Metadata directory does not exist: {metadata_dir}")
        return 1
    
    splits = [s.strip() for s in args.splits.split(',')]
    remove_stress = not args.keep_stress
    remove_punctuation = not args.keep_punctuation
    
    # Validate input files
    if not validate_files(metadata_dir, splits):
        return 1
    
    print("🚀 Starting LRS3 phoneme metadata generation (index-based)...")
    print(f"📁 Metadata directory: {metadata_dir}")
    print(f"📋 Processing splits: {', '.join(splits)}")
    print(f"🔧 Remove stress markers: {remove_stress}")
    print(f"🔧 Remove punctuation: {remove_punctuation}")
    print(f"📊 Dictionary format: <token> <index> (starting from 0)")
    print("-" * 60)
    
    try:
        # Convert words to phonemes
        phoneme_counts = convert_words_to_phonemes(
            metadata_dir, splits, remove_stress, remove_punctuation
        )
        
        if not phoneme_counts:
            print("❌ Error: No phonemes were generated. Check your input files.")
            return 1
        
        # Create index-based phoneme dictionary
        dict_file = create_phoneme_dictionary_indexed(metadata_dir, phoneme_counts)
        
        print("-" * 60)
        print("🎉 Index-based phoneme metadata generation completed successfully!")
        print(f"📁 Output directory: {metadata_dir}")
        print(f"📋 Generated files:")
        
        # List generated files
        for split in splits:
            phn_file = metadata_dir / f"{split}.phn"
            if phn_file.exists():
                print(f"   • {split}.phn")
        print(f"   • dict.phn.txt (index-based: <token> <index>)")
        
        print(f"\n💡 Dictionary format: Each phoneme is mapped to a unique index starting from 0")
        print(f"💡 <blank> token is always at index 0 for CTC/attention models")
        print(f"💡 You can now use these phoneme files for training!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
