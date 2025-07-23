#!/usr/bin/env python3
"""
Video Quality Checker for TCD-TIMIT Preprocessing
=================================================

Quick script to check if the jitter/shaking has been reduced in processed videos.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def analyze_video_stability(video_path):
    """
    Analyze video stability by measuring frame-to-frame differences
    Lower values indicate more stable video (less jitter)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    prev_frame = None
    frame_diffs = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
        
        prev_frame = gray
        frame_count += 1
    
    cap.release()
    
    if not frame_diffs:
        return None
    
    # Calculate stability metrics
    mean_diff = np.mean(frame_diffs)
    std_diff = np.std(frame_diffs)
    max_diff = np.max(frame_diffs)
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_diff': max_diff,
        'frame_count': frame_count,
        'stability_score': mean_diff + std_diff  # Lower is better
    }

def compare_videos(original_dir, processed_dir, output_pattern="*.mp4"):
    """Compare stability between original and processed videos"""
    
    original_path = Path(original_dir)
    processed_path = Path(processed_dir)
    
    original_videos = list(original_path.rglob(output_pattern))
    processed_videos = list(processed_path.rglob(output_pattern))
    
    print("🔍 Video Stability Analysis")
    print("=" * 60)
    print("📊 Analyzing frame-to-frame differences (lower = more stable)")
    print()
    
    for orig_video in original_videos[:5]:  # Test first 5 videos
        # Find corresponding processed video
        rel_path = orig_video.relative_to(original_path)
        
        # Look for processed version (might have different directory structure)
        processed_candidates = [v for v in processed_videos if v.name == orig_video.name]
        
        if not processed_candidates:
            print(f"⚠️  No processed version found for {orig_video.name}")
            continue
        
        proc_video = processed_candidates[0]
        
        print(f"📹 Analyzing: {orig_video.name}")
        
        # Analyze original
        orig_stats = analyze_video_stability(orig_video)
        if orig_stats is None:
            print(f"   ❌ Could not analyze original video")
            continue
        
        # Analyze processed
        proc_stats = analyze_video_stability(proc_video)
        if proc_stats is None:
            print(f"   ❌ Could not analyze processed video")
            continue
        
        # Compare results
        improvement = ((orig_stats['stability_score'] - proc_stats['stability_score']) / 
                      orig_stats['stability_score']) * 100
        
        print(f"   📈 Original stability score: {orig_stats['stability_score']:.2f}")
        print(f"   📉 Processed stability score: {proc_stats['stability_score']:.2f}")
        
        if improvement > 0:
            print(f"   ✅ Improvement: {improvement:.1f}% more stable")
        else:
            print(f"   ⚠️  Change: {abs(improvement):.1f}% {'less' if improvement < 0 else 'more'} stable")
        
        print(f"   🎬 Frames: {orig_stats['frame_count']} → {proc_stats['frame_count']}")
        print()

def quick_visual_check(video_path):
    """Quick visual check - play video for manual inspection"""
    print(f"🎬 Playing video: {video_path}")
    print("Press 'q' to quit, SPACE to pause/resume")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for display if too large
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imshow('Video Check', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to pause
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Check video quality and stability")
    parser.add_argument("--processed-dir", type=str, required=True, 
                       help="Directory with processed videos")
    parser.add_argument("--original-dir", type=str, default=None,
                       help="Directory with original videos for comparison")
    parser.add_argument("--visual-check", action="store_true",
                       help="Show videos for visual inspection")
    
    args = parser.parse_args()
    
    processed_path = Path(args.processed_dir)
    
    if not processed_path.exists():
        print(f"❌ Processed directory not found: {args.processed_dir}")
        return 1
    
    # Find processed videos
    processed_videos = list(processed_path.rglob("*.mp4"))
    
    if not processed_videos:
        print(f"❌ No videos found in: {args.processed_dir}")
        return 1
    
    print(f"✅ Found {len(processed_videos)} processed videos")
    
    if args.original_dir and Path(args.original_dir).exists():
        # Compare with originals
        compare_videos(args.original_dir, args.processed_dir)
    else:
        # Just analyze processed videos
        print("\n🔍 Analyzing processed videos only:")
        print("=" * 40)
        
        for video in processed_videos[:5]:
            stats = analyze_video_stability(video)
            if stats:
                print(f"📹 {video.name}")
                print(f"   📊 Stability score: {stats['stability_score']:.2f}")
                print(f"   🎬 Frames: {stats['frame_count']}")
    
    if args.visual_check:
        print("\n👁️  Visual inspection:")
        for video in processed_videos[:3]:
            quick_visual_check(video)
    
    print("\n💡 Stability Tips:")
    print("   • Lower stability scores = less jitter")
    print("   • Consistent frame differences = smooth video")
    print("   • Use temporal smoothing for best results")

if __name__ == "__main__":
    exit(main())
