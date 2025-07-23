#!/usr/bin/env python3
"""
TCD-TIMIT Dataset Structure Explorer

This script explores the TCD-TIMIT dataset structure to understand:
- Directory organization (volunteers vs lipspeakers)
- Video file formats and naming conventions  
- Transcript files (.mlf format)
- Speaker organization
- Available data splits

Usage:
    python explore_tcd_timit.py --data-dir /path/to/TCD-TIMIT
"""

import os
import argparse
from pathlib import Path
import glob

def explore_directory_structure(data_dir):
    """Explore the overall directory structure"""
    print("Exploring TCD-TIMIT directory structure...")
    print(f"Dataset root: {data_dir}")
    print("-" * 60)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        return False
    
    # List top-level directories
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Top-level directories ({len(subdirs)}):")
    for subdir in sorted(subdirs):
        print(f"   • {subdir.name}")
    
    return True

def explore_speaker_structure(data_dir):
    """Explore speaker organization within volunteers/lipspeakers"""
    print("\nExploring speaker structure...")
    
    data_path = Path(data_dir)
    
    # Check volunteers
    volunteers_dir = data_path / "volunteers"
    if volunteers_dir.exists():
        print(f"\nVolunteers directory:")
        speakers = [d for d in volunteers_dir.iterdir() if d.is_dir()]
        print(f"   • Number of volunteer speakers: {len(speakers)}")
        if speakers:
            print(f"   • Sample speakers: {', '.join([s.name for s in sorted(speakers)[:5]])}")
            
            # Explore one speaker's structure
            sample_speaker = speakers[0]
            print(f"\nSample speaker structure ({sample_speaker.name}):")
            files = list(sample_speaker.glob("*"))
            video_files = [f for f in files if f.suffix.lower() in ['.mp4', '.avi', '.mov']]
            other_files = [f for f in files if f not in video_files]
            
            print(f"   • Video files: {len(video_files)}")
            if video_files:
                print(f"     - Sample: {video_files[0].name}")
            print(f"   • Other files: {len(other_files)}")
            for f in other_files[:3]:
                print(f"     - {f.name}")
    
    # Check lipspeakers
    lipspeakers_dir = data_path / "lipspeakers"
    if lipspeakers_dir.exists():
        print(f"\nLipspeakers directory:")
        speakers = [d for d in lipspeakers_dir.iterdir() if d.is_dir()]
        print(f"   • Number of lipspeakers: {len(speakers)}")
        if speakers:
            print(f"   • Sample speakers: {', '.join([s.name for s in sorted(speakers)[:5]])}")

def explore_transcript_files(data_dir):
    """Explore .mlf transcript files"""
    print("\n📝 Exploring transcript files...")
    
    data_path = Path(data_dir)
    
    # Find .mlf files
    mlf_files = list(data_path.glob("**/*.mlf"))
    print(f"Found {len(mlf_files)} .mlf files:")
    
    for mlf_file in mlf_files:
        print(f"   • {mlf_file.relative_to(data_path)}")
        
        # Read a sample of the .mlf file
        try:
            with open(mlf_file, 'r') as f:
                lines = f.readlines()[:20]  # First 20 lines
            
            print(f"     - Size: {len(lines)} lines (showing first 10)")
            for i, line in enumerate(lines[:10], 1):
                print(f"     {i:2d}: {line.strip()}")
            print("     ...")
        except Exception as e:
            print(f"     - Error reading file: {e}")

def explore_video_files(data_dir):
    """Explore video file patterns"""
    print("\n🎥 Exploring video files...")
    
    data_path = Path(data_dir)
    
    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    all_videos = []
    
    for ext in video_extensions:
        videos = list(data_path.glob(f"**/{ext}"))
        all_videos.extend(videos)
    
    print(f"🎬 Found {len(all_videos)} video files")
    
    if all_videos:
        # Group by extension
        by_ext = {}
        for video in all_videos:
            ext = video.suffix.lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1

        print("📊 Video formats:")
        for ext, count in by_ext.items():
            print(f"   • {ext}: {count} files")

        # Sample file names
        print("\n📝 Sample video filenames:")
        for video in sorted(all_videos)[:10]:
            rel_path = video.relative_to(data_path)
            print(f"   • {rel_path}")

        # Analyze video properties for all videos
        print(f"\nAnalyzing video properties for all {len(all_videos)} videos...")
        analyze_video_properties(all_videos, data_path)

def analyze_video_properties(video_files, data_path):
    """Analyze video properties like resolution, duration, fps, etc."""
    import cv2
    
    total_duration = 0
    resolutions = {}
    fps_values = []
    file_sizes = []
    
    for video_path in video_files:
        try:
            # Get file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            file_sizes.append(file_size_mb)

            # Open video with OpenCV
            cap = cv2.VideoCapture(str(video_path))

            if cap.isOpened():
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Calculate duration
                duration = frame_count / fps if fps > 0 else 0
                total_duration += duration

                # Store resolution
                resolution = f"{width}x{height}"
                resolutions[resolution] = resolutions.get(resolution, 0) + 1

                # Store FPS
                if fps > 0:
                    fps_values.append(fps)

                cap.release()

        except Exception as e:
            print(f"   ⚠️  Could not analyze {video_path.name}: {e}")
            continue

    # Print analysis results
    if total_duration > 0:
        hours = total_duration / 3600
        print(f"⏱️  Total duration (all videos): {total_duration:.1f}s ({hours:.2f} hours)")

    if resolutions:
        print("📺 Video resolutions found:")
        for res, count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {res}: {count} videos")

    if fps_values:
        avg_fps = sum(fps_values) / len(fps_values)
        print(f"🎬 Frame rates: {min(fps_values):.1f}-{max(fps_values):.1f} fps (avg: {avg_fps:.1f})")

    if file_sizes:
        avg_size = sum(file_sizes) / len(file_sizes)
        total_size_gb = sum(file_sizes) / 1024
        print(f"💾 File sizes: {min(file_sizes):.1f}-{max(file_sizes):.1f} MB (avg: {avg_size:.1f} MB)")
        print(f"💾 Total size: {total_size_gb:.2f} GB")

def analyze_dataset_splits(data_dir):
    """Analyze potential dataset splits"""
    print("\n📊 Analyzing potential dataset splits...")
    
    data_path = Path(data_dir)
    
    # Check if there are predefined split files
    split_patterns = ['*train*', '*test*', '*val*', '*split*']
    split_files = []
    
    for pattern in split_patterns:
        files = list(data_path.glob(f"**/{pattern}"))
        split_files.extend(files)
    
    if split_files:
        print(f"📋 Found potential split files:")
        for split_file in split_files:
            print(f"   • {split_file.relative_to(data_path)}")
    else:
        print("📋 No predefined split files found")
        print("💡 Will need to create splits based on speakers or sentences")

def main():
    parser = argparse.ArgumentParser(
        description="Explore TCD-TIMIT dataset structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Path to TCD-TIMIT dataset directory"
    )
    
    args = parser.parse_args()
    
    print("🚀 TCD-TIMIT Dataset Structure Explorer")
    print("=" * 60)
    
    # Explore dataset structure
    if not explore_directory_structure(args.data_dir):
        return 1
    
    explore_speaker_structure(args.data_dir)
    explore_transcript_files(args.data_dir)
    explore_video_files(args.data_dir)
    analyze_dataset_splits(args.data_dir)
    
    print("\n" + "=" * 60)
    print("🎉 Dataset exploration completed!")
    print("💡 Use this information to design the preprocessing pipeline")
    
    return 0

if __name__ == "__main__":
    exit(main())
