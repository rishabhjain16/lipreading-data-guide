#!/usr/bin/env python3
"""
TCD-TIMIT MLF Parser

This script parses .mlf (Master Label File) format transcripts commonly used in TCD-TIMIT.
MLF format typically contains phoneme or word-level transcriptions with timing information.

Usage:
    python parse_mlf.py --mlf-file /path/to/transcripts.mlf [--output-dir /path/to/output]
"""

import os
import argparse
import re
from pathlib import Path
from collections import defaultdict

def parse_mlf_file(mlf_file_path):
    """
    Parse MLF (Master Label File) format
    
    MLF format typically looks like:
    #!MLF!#
    "*/filename.rec"
    start_time end_time phoneme
    start_time end_time phoneme
    .
    "*/next_filename.rec"
    ...
    
    Returns:
        dict: {filename: [(start, end, label), ...]}
    """
    print(f"Parsing MLF file: {mlf_file_path}")
    
    transcripts = defaultdict(list)
    current_file = None
    
    with open(mlf_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # MLF header
        if line == "#!MLF!#":
            continue
            
        # File specification line (usually starts with " and ends with ")
        if line.startswith('"') and line.endswith('"'):
            # Extract filename from path
            file_spec = line[1:-1]  # Remove quotes
            # Common patterns: "*/filename.rec", "./filename.lab", etc.
            if '/' in file_spec:
                current_file = file_spec.split('/')[-1]
            else:
                current_file = file_spec
            # Remove extensions like .rec, .lab
            current_file = re.sub(r'\.(rec|lab|txt)$', '', current_file)
            continue
            
        # End of file marker
        if line == ".":
            current_file = None
            continue
            
        # Transcript line
        if current_file is not None:
            # Try to parse timing + label format
            parts = line.split()
            
            if len(parts) >= 3:
                # Format: start_time end_time label
                try:
                    start_time = int(parts[0])
                    end_time = int(parts[1])
                    label = ' '.join(parts[2:])
                    transcripts[current_file].append((start_time, end_time, label))
                except ValueError:
                    # If timing parsing fails, treat as label-only
                    label = line
                    transcripts[current_file].append((None, None, label))
            elif len(parts) == 1:
                # Format: just label
                label = parts[0]
                transcripts[current_file].append((None, None, label))
            else:
                print(f"⚠️  Warning: Unrecognized format on line {line_num}: {line}")
    
    print(f"  ✅ Parsed {len(transcripts)} files with transcripts")
    return dict(transcripts)

def analyze_transcripts(transcripts):
    """Analyze the parsed transcripts to understand the data structure"""
    print("\nAnalyzing transcripts...")
    
    total_files = len(transcripts)
    total_segments = sum(len(segments) for segments in transcripts.values())
    
    print(f"   • Total files: {total_files}")
    print(f"   • Total segments: {total_segments}")
    print(f"   • Average segments per file: {total_segments/total_files:.1f}")
    
    # Analyze label types
    all_labels = []
    has_timing = 0
    
    for file_id, segments in transcripts.items():
        for start, end, label in segments:
            all_labels.append(label)
            if start is not None and end is not None:
                has_timing += 1
    
    unique_labels = set(all_labels)
    print(f"   • Unique labels: {len(unique_labels)}")
    print(f"   • Segments with timing: {has_timing}/{total_segments}")
    
    # Show most common labels
    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\n📈 Most common labels:")
    for label, count in label_counts.most_common(10):
        print(f"   • {label}: {count}")
    
    # Show sample files
    print(f"\n📝 Sample files:")
    for i, (file_id, segments) in enumerate(list(transcripts.items())[:5]):
        print(f"   • {file_id}: {len(segments)} segments")
        if segments:
            sample_segment = segments[0]
            if sample_segment[0] is not None:
                print(f"     - Example: {sample_segment[0]}-{sample_segment[1]} '{sample_segment[2]}'")
            else:
                print(f"     - Example: '{sample_segment[2]}'")

def save_transcripts(transcripts, output_dir, format_type="text"):
    """Save transcripts in various formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving transcripts to: {output_path}")
    
    if format_type == "text":
        # Save as individual text files
        for file_id, segments in transcripts.items():
            # Extract just the text content
            text_content = []
            for start, end, label in segments:
                # Skip silence markers and special tokens
                if label.upper() not in ['SIL', 'SP', 'SPN', '<s>', '</s>']:
                    text_content.append(label)
            
            if text_content:
                output_file = output_path / f"{file_id}.txt"
                with open(output_file, 'w') as f:
                    f.write(' '.join(text_content) + '\n')
        
        print(f"  ✅ Saved {len(transcripts)} text files")
    
    elif format_type == "csv":
        # Save as CSV for processing pipeline
        import csv
        csv_file = output_path / "transcripts.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_id', 'transcript', 'num_segments', 'has_timing'])
            
            for file_id, segments in transcripts.items():
                # Extract text
                text_content = []
                has_timing = any(s[0] is not None for s in segments)
                
                for start, end, label in segments:
                    if label.upper() not in ['SIL', 'SP', 'SPN', '<s>', '</s>']:
                        text_content.append(label)
                
                transcript = ' '.join(text_content)
                writer.writerow([file_id, transcript, len(segments), has_timing])
        
        print(f"  ✅ Saved CSV file: {csv_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Parse TCD-TIMIT .mlf transcript files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mlf-file", 
        type=str, 
        required=True,
        help="Path to .mlf transcript file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Directory to save parsed transcripts (optional)"
    )
    parser.add_argument(
        "--format", 
        choices=["text", "csv"], 
        default="text",
        help="Output format for transcripts"
    )
    
    args = parser.parse_args()
    
    if not Path(args.mlf_file).exists():
        print(f"Error: MLF file not found: {args.mlf_file}")
        return 1
        
    print("TCD-TIMIT MLF Parser")
    print("-" * 40)    # Parse MLF file
    transcripts = parse_mlf_file(args.mlf_file)
    
    if not transcripts:
        print("❌ No transcripts found in MLF file")
        return 1
    
    # Analyze transcripts
    analyze_transcripts(transcripts)
    
    # Save transcripts if output directory specified
    if args.output_dir:
        save_transcripts(transcripts, args.output_dir, args.format)
    
    print("\n🎉 MLF parsing completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
