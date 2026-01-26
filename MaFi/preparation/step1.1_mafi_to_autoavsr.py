#!/usr/bin/env python3

import pandas as pd
import sentencepiece as spm
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True) 
    parser.add_argument("--spm-model-path", 
                        default="/home/rishabhjain/Desktop/Experiments/auto_avsr/spm/unigram/unigram5000.model",
                        help="Path to SentencePiece model")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} records")
    
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model_path)
    print(f"Loaded SentencePiece model with {sp.vocab_size()} vocab")
    
    # Process each row
    records = []
    for _, row in df.iterrows():
        # Use the original video path from the dataset
        video_path = row['video_path']
        
        # Tokenize the word using SentencePiece (convert to uppercase and remove hyphens)
        word = row['word'].upper().replace('-', '')
        tokens = sp.encode(word, out_type=int)
        tokenized = ' '.join(map(str, tokens))
        
        # Extract ID from video path (filename without extension)
        import os
        filename = os.path.basename(row['video_path'])
        video_id = os.path.splitext(filename)[0]
        
        record = {
            'dataset': 'mafi',
            'video_path': video_path,
            'num_frames': row['num_frames'],
            'tokenized': tokenized
        }
        records.append(record)
    
    # Save output
    output_df = pd.DataFrame(records)
    output_df.to_csv(args.output_csv, index=False, header=False)
    print(f"Saved {len(output_df)} records to {args.output_csv}")
    print(f"Sample records:")
    print(output_df.head(3))

if __name__ == "__main__":
    main()
