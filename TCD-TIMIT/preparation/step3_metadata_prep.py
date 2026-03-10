#!/usr/bin/env python3
"""
TCD-TIMIT Step 3: Metadata Preparation

This script counts frames and creates manifest files for TCD-TIMIT dataset.

Default behavior:
- Creates test-only manifests for multiple configurations:
  - lipspeakers_30degcam/, lipspeakers_straightcam/, lipspeakers/ (combined cameras)
  - volunteers_30degcam/, volunteers_straightcam/, volunteers/ (combined cameras)
  - volunteers_30degcam_lipcompare/, volunteers_straightcam_lipcompare/ (female volunteers matched to lipspeakers data size)
  - combined/ (all data)
- All data treated as test data (no train/val splits)

With --use-splits flag:
- Uses the splits created by step2 (train.txt, val.txt, test.txt)
- Creates train/valid/test manifests

Usage:
    # Default: Create test-only manifests for all configurations
    python step3_metadata_prep.py \
      --tcd-data-dir /path/to/processed/tcd_timit/tcd_timit_video \
      --metadata-dir /path/to/processed/tcd_timit/metadata \
      --vocab-size 1000
    
    # With splits: Use existing train/val/test splits from step2
    python step3_metadata_prep.py \
      --tcd-data-dir /path/to/processed/tcd_timit/tcd_timit_video \
      --metadata-dir /path/to/processed/tcd_timit/metadata \
      --vocab-size 1000 \
      --use-splits
"""

import os
import cv2
import shutil
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile

# Import from LRS3 preparation (reuse the vocabulary generation)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "LRS3" / "preparation"))
from gen_subword import gen_vocab

def count_frames(fids, base_dir):
    """Count frames in audio and video files"""
    print("Counting frames in audio and video files...")
    total_num_frames = []
    
    for fid in tqdm(fids, desc="Counting frames"):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        
        if not os.path.exists(wav_fn):
            print(f"Warning: Missing audio file: {wav_fn}")
            continue
        if not os.path.exists(video_fn):
            print(f"Warning: Missing video file: {video_fn}")
            continue
            
        try:
            num_frames_audio = len(wavfile.read(wav_fn)[1])
            cap = cv2.VideoCapture(video_fn)
            num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_num_frames.append([num_frames_audio, num_frames_video])
        except Exception as e:
            print(f"Warning: Error processing {fid}: {str(e)}")
            continue
    
    print(f"  Successfully counted frames for {len(total_num_frames)} files")
    return total_num_frames

def create_test_manifest(tcd_data_dir, fids, labels, nfs_audio, nfs_video, metadata_dir, config_name, vocab_size):
    """Create test-only manifest for a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Creating test manifest for {config_name}...")
    print(f"{'='*60}")
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if not fids:
        print(f"⚠️ Warning: No data for {config_name}")
        return
    
    print(f"🔄 Processing {len(fids)} files...")
    
    # Generate vocabulary
    print(f"Generating sentencepiece vocabulary...")
    vocab_dir = metadata_dir / f"spm{vocab_size}"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    spm_filename_prefix = f"spm_unigram{vocab_size}"
    
    # Adjust vocab size if dataset is small
    total_words = len(set(" ".join(labels).lower().split()))
    if len(labels) < 10 or total_words < vocab_size:
        print(f"Warning: Small dataset detected: {len(labels)} samples, {total_words} unique words")
        print(f"Adjusting vocabulary size from {vocab_size} to {min(total_words, max(5, vocab_size//2))}")
        vocab_size = min(total_words, max(5, vocab_size//2))
        vocab_dir = metadata_dir / f"spm{vocab_size}"
        vocab_dir.mkdir(parents=True, exist_ok=True)
        spm_filename_prefix = f"spm_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w", delete=False) as f:
        for label in labels:
            f.write(label.lower() + "\n")
        temp_file = f.name
    
    gen_vocab(Path(temp_file), vocab_dir / spm_filename_prefix, 'unigram', vocab_size)
    os.unlink(temp_file)
    
    vocab_path = (vocab_dir / spm_filename_prefix).as_posix() + '.txt'
    print(f"✅ Created vocabulary: {vocab_path}")
    
    # Create test manifest
    tsv_path = metadata_dir / "test.tsv"
    wrd_path = metadata_dir / "test.wrd"
    
    with open(tsv_path, 'w') as f:
        f.write('/\n')  # Header line
        for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
            video_path = os.path.abspath(f"{tcd_data_dir}/{fid}.mp4")
            audio_path = os.path.abspath(f"{tcd_data_dir}/{fid}.wav")
            
            # Format: id, video_path, audio_path, num_video_frames, num_audio_frames
            f.write('\t'.join([
                fid,
                video_path,
                audio_path,
                str(nf_video),
                str(nf_audio)
            ]) + '\n')
    
    with open(wrd_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    # Copy vocabulary as dictionary
    dict_path = metadata_dir / "dict.wrd.txt"
    shutil.copyfile(vocab_path, str(dict_path))
    
    print(f"✅ Created test manifest: {tsv_path} ({len(fids)} entries)")
    print(f"✅ Created word file: {wrd_path}")
    print(f"✅ Created dictionary: {dict_path}")


def check_missing_files(fids, base_dir):
    """Check for missing audio/video files"""
    print("Checking for missing files...")
    missing = []
    
    for fid in tqdm(fids, desc="Checking files"):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        is_file = os.path.isfile(wav_fn) and os.path.isfile(video_fn)
        
        if not is_file:
            if not os.path.isfile(wav_fn):
                print(f"  Missing audio: {wav_fn}")
            if not os.path.isfile(video_fn):
                print(f"  Missing video: {video_fn}")
            missing.append(fid)
    
    if len(missing) > 0:
        print(f"  Found {len(missing)} files with missing audio/video")
    else:
        print(f"  All files present")
    
    return missing

def create_frame_files(tcd_data_dir, fids, num_frames):
    """Create nframes.audio and nframes.video files"""
    print("Creating frame count files...")
    
    audio_num_frames = [x[0] for x in num_frames]
    video_num_frames = [x[1] for x in num_frames]

    nframes_audio_path = os.path.join(tcd_data_dir, 'nframes.audio')
    nframes_video_path = os.path.join(tcd_data_dir, 'nframes.video')
    
    with open(nframes_audio_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
    with open(nframes_video_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in video_num_frames]))

    print(f"  Created: {nframes_audio_path}")
    print(f"  Created: {nframes_video_path}")
    
    return nframes_audio_path, nframes_video_path

def create_manifest_files(tcd_data_dir, metadata_dir, vocab_size):
    """Create manifest files for training"""
    print("Creating manifest files...")
    
    # Required files
    file_list = os.path.join(tcd_data_dir, "file.list")
    label_list = os.path.join(tcd_data_dir, "label.list")
    nframes_audio_file = os.path.join(tcd_data_dir, "nframes.audio")
    nframes_video_file = os.path.join(tcd_data_dir, "nframes.video")
    
    # Check if all required files exist
    required_files = [file_list, label_list, nframes_audio_file, nframes_video_file]
    for req_file in required_files:
        if not os.path.exists(req_file):
            print(f"Error: Required file not found: {req_file}")
            return None
    
    # Generate vocabulary
    print("Generating sentencepiece vocabulary...")
    vocab_dir = (Path(metadata_dir) / f"spm{vocab_size}").absolute()
    vocab_dir.mkdir(parents=True, exist_ok=True)
    smp_filename_prefix = f"spm_unigram{vocab_size}"
    
    # Read all label text
    label_text = [ln.strip() for ln in open(label_list).readlines()]
    
    # Check if we have enough data for the requested vocab size
    total_words = len(set(" ".join(label_text).lower().split()))
    
    if len(label_text) < 10 or total_words < vocab_size:
        print(f"Warning: Small dataset detected: {len(label_text)} samples, {total_words} unique words")
        print(f"Adjusting vocabulary size from {vocab_size} to {min(total_words, max(5, vocab_size//2))}")
        vocab_size = min(total_words, max(5, vocab_size//2))
        vocab_dir = (Path(metadata_dir) / f"smp{vocab_size}").absolute()
        vocab_dir.mkdir(parents=True, exist_ok=True)
        smp_filename_prefix = f"smp_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w") as f:
        for t in label_text:
            f.write(t.lower() + "\n")
        f.flush()  # Ensure data is written before training
        gen_vocab(Path(f.name), vocab_dir/smp_filename_prefix, 'unigram', vocab_size)
    
    vocab_path = (vocab_dir/smp_filename_prefix).as_posix() + '.txt'
    print(f"  Created vocabulary: {vocab_path}")

    def setup_target(target_dir, train, valid, test):
        """Setup target directory with train/valid/test splits"""
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        for name, data in zip(["train", "valid", "test"], [train, valid, test]):
            if not data:
                continue
                
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')  # Header line
                for fid, label, nf_audio, nf_video in data:
                    # Convert file ID to full paths
                    video_path = os.path.abspath(f"{tcd_data_dir}/{fid}.mp4")
                    audio_path = os.path.abspath(f"{tcd_data_dir}/{fid}.wav")
                    
                    # Format: id, video_path, audio_path, num_video_frames, num_audio_frames
                    fo.write('\t'.join([
                        fid,
                        video_path,
                        audio_path,
                        str(nf_video),
                        str(nf_audio)
                    ])+'\n')
            
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")
        print(f"  Copied vocabulary to: {target_dir}/dict.wrd.txt")

    # Read all data
    fids = [x.strip() for x in open(file_list).readlines()]
    labels = [x.strip().lower() for x in open(label_list).readlines()]
    nfs_audio = [x.strip() for x in open(nframes_audio_file).readlines()]
    nfs_video = [x.strip() for x in open(nframes_video_file).readlines()]

    # Read dataset splits
    train, valid, test = [], [], []
    
    # Load split files created by step 2
    train_split_file = f"{tcd_data_dir}/train.txt"
    val_split_file = f"{tcd_data_dir}/val.txt"
    test_split_file = f"{tcd_data_dir}/test.txt"
    
    train_ids = set()
    valid_ids = set()
    test_ids = set()
    
    if os.path.exists(train_split_file):
        train_ids = set(line.strip() for line in open(train_split_file))
    if os.path.exists(val_split_file):
        valid_ids = set(line.strip() for line in open(val_split_file))
    if os.path.exists(test_split_file):
        test_ids = set(line.strip() for line in open(test_split_file))
    
    # Assign files to splits
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        # Extract transcript ID from file path for comparison
        transcript_id = fid.split('/')[-1]  # Last part after final /
        
        data_item = [fid, label, nf_audio, nf_video]
        
        if transcript_id in train_ids:
            train.append(data_item)
        elif transcript_id in valid_ids:
            valid.append(data_item)
        elif transcript_id in test_ids:
            test.append(data_item)
        else:
            # Default to train if not in any split
            train.append(data_item)

    output_dir = metadata_dir
    print(f"Setting up metadata directory: {output_dir}")
    setup_target(output_dir, train, valid, test)
    
    print(f"  Dataset splits:")
    print(f"    Train: {len(train)} samples")
    print(f"    Valid: {len(valid)} samples") 
    print(f"    Test: {len(test)} samples")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description='TCD-TIMIT Processing - Count frames and create manifest files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tcd-data-dir', type=str, required=True,
                        help='TCD-TIMIT data directory (contains file.list, label.list, and video files)')
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size for sentencepiece')
    parser.add_argument('--use-splits', action='store_true',
                        help='Use existing train/val/test split files from step2 (default: create test-only manifests)')
    
    args = parser.parse_args()
    
    # Validate input directory
    tcd_data_dir = Path(args.tcd_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    
    if not tcd_data_dir.exists():
        print(f"Error: TCD-TIMIT data directory not found: {tcd_data_dir}")
        return 1
    
    file_list_path = tcd_data_dir / 'file.list'
    if not file_list_path.exists():
        print(f"Error: file.list not found in: {tcd_data_dir}")
        print("Try running step2_generate_file_lists.py first to generate file.list and label.list")
        return 1
    
    print(f"Starting TCD-TIMIT processing...")
    print(f"Data directory: {tcd_data_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"Vocabulary size: {args.vocab_size}")
    print("-" * 60)
    
    try:
        # Read file list
        with open(file_list_path, 'r') as f:
            fids = [ln.strip() for ln in f.readlines()]
        print(f"Found {len(fids)} files to process")
        
        # Check if user wants to use existing splits
        if args.use_splits:
            train_split_file = tcd_data_dir / "train.txt"
            if not train_split_file.exists():
                print("Error: --use-splits specified but no split files found")
                print(f"Expected: {train_split_file}")
                print("Run step2_generate_file_lists.py first to create split files")
                return 1
            
            print("Using existing split files from step2 - creating train/val/test manifests")
            
            # Step 1: Check for missing files
            missing_fids = check_missing_files(fids, str(tcd_data_dir))
            
            if len(missing_fids) > 0:
                missing_list_path = tcd_data_dir / 'missing.list'
                with open(missing_list_path, 'w') as fo:
                    fo.write('\n'.join(missing_fids) + '\n')
                print(f"Some audio/video files are missing. See: {missing_list_path}")
                print("Please resolve missing files before proceeding.")
                return 1
            
            # Step 2: Count frames
            num_frames = count_frames(fids, str(tcd_data_dir))
            
            if len(num_frames) == 0:
                print("No valid files found for frame counting")
                return 1
            
            # Step 3: Create frame count files
            nframes_audio_path, nframes_video_path = create_frame_files(
                str(tcd_data_dir), fids, num_frames
            )
            
            # Step 4: Create manifest files with splits
            output_dir = create_manifest_files(str(tcd_data_dir), str(metadata_dir), args.vocab_size)
            
            print("-" * 60)
            print("Processing completed successfully!")
            print(f"Output directory: {output_dir}")
            print("Generated files:")
            print(f"   Frame counts: {nframes_audio_path}, {nframes_video_path}")
            print(f"   Manifests: train.tsv, valid.tsv, test.tsv")
            print(f"   Word files: train.wrd, valid.wrd, test.wrd")
            print(f"   Dictionary: dict.wrd.txt")
        
        else:
            print("Creating test-only manifests for all configurations (9 folders)")
            print("Use --use-splits flag to create train/val/test splits instead")
            
            # Read labels
            label_list_path = tcd_data_dir / 'label.list'
            if not label_list_path.exists():
                print(f"Error: label.list not found in: {tcd_data_dir}")
                return 1
            
            with open(label_list_path, 'r') as f:
                labels = [ln.strip() for ln in f.readlines()]
            
            # Count frames for all files
            num_frames = count_frames(fids, str(tcd_data_dir))
            
            if len(num_frames) == 0:
                print("No valid files found for frame counting")
                return 1
            
            # Extract frame counts
            nfs_audio = [str(x[0]) for x in num_frames]
            nfs_video = [str(x[1]) for x in num_frames]
            
            # Group files by subset and camera
            configs = {
                'lipspeakers_30degcam': [],
                'lipspeakers_straightcam': [],
                'lipspeakers': [],
                'volunteers_30degcam': [],
                'volunteers_straightcam': [],
                'volunteers': [],
                'volunteers_30degcam_lipcompare': [],
                'volunteers_straightcam_lipcompare': [],
                'combined': []
            }
            
            # Track speakers and counts for lipcompare subset
            volunteer_speakers_30deg = {}  # speaker -> list of indices
            volunteer_speakers_straight = {}  # speaker -> list of indices
            lipspeakers_30deg_count = 0
            lipspeakers_straight_count = 0
            
            for i, fid in enumerate(fids):
                parts = fid.split('/')
                # Expected format: subset/speaker/Clips/camera/filename
                # Example: lipspeakers/Lipspkr1/Clips/30degcam/Lipspkr1_Clips_30degcam_sa1
                
                if len(parts) >= 4:
                    subset = parts[0]  # volunteers or lipspeakers
                    speaker = parts[1]  # speaker ID
                    camera = parts[3]  # 30degcam or straightcam (after Clips)
                    
                    # Add to combined
                    configs['combined'].append(i)
                    
                    # Add to subset-specific
                    if subset in ['volunteers', 'lipspeakers']:
                        configs[subset].append(i)
                        
                        # Add to camera-specific
                        if camera in ['30degcam', 'straightcam']:
                            config_key = f"{subset}_{camera}"
                            if config_key in configs:
                                configs[config_key].append(i)
                            
                            # Count lipspeakers for target size
                            if subset == 'lipspeakers':
                                if camera == '30degcam':
                                    lipspeakers_30deg_count += 1
                                elif camera == 'straightcam':
                                    lipspeakers_straight_count += 1
                            
                            # For volunteers, track speakers and their files
                            elif subset == 'volunteers':
                                if camera == '30degcam':
                                    if speaker not in volunteer_speakers_30deg:
                                        volunteer_speakers_30deg[speaker] = []
                                    volunteer_speakers_30deg[speaker].append(i)
                                elif camera == 'straightcam':
                                    if speaker not in volunteer_speakers_straight:
                                        volunteer_speakers_straight[speaker] = []
                                    volunteer_speakers_straight[speaker].append(i)
            
            # Select female volunteers to match lipspeakers data size
            def select_volunteers_to_match_size(volunteer_speakers_dict, target_count):
                """Select female volunteers to exactly match target count"""
                # Filter for female speakers (ending with 'F')
                female_speakers = {s: files for s, files in volunteer_speakers_dict.items() if s.endswith('F')}
                
                # Sort by speaker ID for consistency
                sorted_speakers = sorted(female_speakers.keys())
                
                selected_indices = []
                selected_speakers = []
                
                for speaker in sorted_speakers:
                    speaker_files = female_speakers[speaker]
                    
                    # If adding all files from this speaker would exceed target, only add what we need
                    if len(selected_indices) + len(speaker_files) > target_count:
                        remaining = target_count - len(selected_indices)
                        selected_indices.extend(speaker_files[:remaining])
                        selected_speakers.append(f"{speaker}(partial:{remaining}/{len(speaker_files)})")
                        break
                    else:
                        selected_indices.extend(speaker_files)
                        selected_speakers.append(speaker)
                    
                    # Stop when we've exactly matched target
                    if len(selected_indices) >= target_count:
                        break
                
                return selected_indices, selected_speakers
            
            # Select volunteers for each camera
            lipcompare_30deg_indices, lipcompare_30deg_speakers = select_volunteers_to_match_size(
                volunteer_speakers_30deg, lipspeakers_30deg_count
            )
            lipcompare_straight_indices, lipcompare_straight_speakers = select_volunteers_to_match_size(
                volunteer_speakers_straight, lipspeakers_straight_count
            )
            
            configs['volunteers_30degcam_lipcompare'] = lipcompare_30deg_indices
            configs['volunteers_straightcam_lipcompare'] = lipcompare_straight_indices
            
            print(f"\nLipspeakers data size:")
            print(f"  30degcam: {lipspeakers_30deg_count} files")
            print(f"  straightcam: {lipspeakers_straight_count} files")
            print(f"\nSelected female volunteers for lipcompare:")
            print(f"  30degcam: {len(lipcompare_30deg_indices)} files from {len(lipcompare_30deg_speakers)} speakers {lipcompare_30deg_speakers}")
            print(f"  straightcam: {len(lipcompare_straight_indices)} files from {len(lipcompare_straight_speakers)} speakers {lipcompare_straight_speakers}")
            
            # Create test manifests for each configuration
            for config_name, indices in configs.items():
                if not indices:
                    continue
                
                config_fids = [fids[i] for i in indices]
                config_labels = [labels[i] for i in indices]
                config_nfs_audio = [nfs_audio[i] for i in indices]
                config_nfs_video = [nfs_video[i] for i in indices]
                
                config_metadata_dir = metadata_dir / config_name
                create_test_manifest(
                    str(tcd_data_dir),
                    config_fids,
                    config_labels,
                    config_nfs_audio,
                    config_nfs_video,
                    config_metadata_dir,
                    config_name,
                    args.vocab_size
                )
            
            print("\n" + "="*60)
            print("✅ Processing completed successfully!")
            print("="*60)
            print(f"\nTest-only manifests created:")
            for config_name, indices in configs.items():
                if indices:
                    print(f"  {config_name}: {len(indices)} files → {metadata_dir}/{config_name}/")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
