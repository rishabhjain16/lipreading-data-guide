#!/usr/bin/env python3
"""
LRS2 Processing Script
Combines frame counting and manifest creation into one step.
"""

import os
import cv2
import shutil
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from gen_subword import gen_vocab
from tempfile import NamedTemporaryFile

def count_frames(fids, base_dir):
    """Count frames in audio and video files"""
    print("🔄 Counting frames in audio and video files...")
    total_num_frames = []
    
    for fid in tqdm(fids, desc="Counting frames"):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        
        if not os.path.exists(wav_fn):
            print(f"⚠️  Missing audio file: {wav_fn}")
            continue
        if not os.path.exists(video_fn):
            print(f"⚠️  Missing video file: {video_fn}")
            continue
            
        try:
            num_frames_audio = len(wavfile.read(wav_fn)[1])
            cap = cv2.VideoCapture(video_fn)
            num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_num_frames.append([num_frames_audio, num_frames_video])
        except Exception as e:
            print(f"⚠️  Error processing {fid}: {str(e)}")
            continue
    
    print(f"  ✅ Successfully counted frames for {len(total_num_frames)} files")
    return total_num_frames

def check_missing_files(fids, base_dir):
    """Check for missing audio/video files"""
    print("🔍 Checking for missing files...")
    missing = []
    
    for fid in tqdm(fids, desc="Checking files"):
        wav_fn = os.path.join(base_dir, fid + ".wav")
        video_fn = os.path.join(base_dir, fid + ".mp4")
        is_file = os.path.isfile(wav_fn) and os.path.isfile(video_fn)
        
        if not is_file:
            if not os.path.isfile(wav_fn):
                print(f"  ❌ Missing audio: {wav_fn}")
            if not os.path.isfile(video_fn):
                print(f"  ❌ Missing video: {video_fn}")
            missing.append(fid)
    
    if len(missing) > 0:
        print(f"  ⚠️  Found {len(missing)} files with missing audio/video")
    else:
        print(f"  ✅ All files present")
    
    return missing

def create_frame_files(lrs2_data_dir, fids, num_frames):
    """Create nframes.audio and nframes.video files"""
    print("📝 Creating frame count files...")
    
    audio_num_frames = [x[0] for x in num_frames]
    video_num_frames = [x[1] for x in num_frames]

    nframes_audio_path = os.path.join(lrs2_data_dir, 'nframes.audio')
    nframes_video_path = os.path.join(lrs2_data_dir, 'nframes.video')
    
    with open(nframes_audio_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
    with open(nframes_video_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in video_num_frames]))
    
    print(f"  ✅ Created: {nframes_audio_path}")
    print(f"  ✅ Created: {nframes_video_path}")
    
    return nframes_audio_path, nframes_video_path

def create_manifest_files(lrs2_data_dir, metadata_dir, vocab_size):
    """Create manifest TSV files for train, valid, test"""
    print("📋 Creating manifest files...")
    
    # Validate required files exist
    file_list = os.path.join(lrs2_data_dir, "file.list")
    label_list = os.path.join(lrs2_data_dir, "label.list")
    nframes_audio_file = os.path.join(lrs2_data_dir, "nframes.audio")
    nframes_video_file = os.path.join(lrs2_data_dir, "nframes.video")
    
    for required_file in [file_list, label_list, nframes_audio_file, nframes_video_file]:
        if not os.path.isfile(required_file):
            raise FileNotFoundError(f"Required file not found: {required_file}")

    print("🔤 Generating sentencepiece vocabulary...")
    vocab_dir = (Path(metadata_dir) / f"spm{vocab_size}").absolute()
    vocab_dir.mkdir(parents=True, exist_ok=True)
    smp_filename_prefix = f"spm_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w") as f:
        label_text = [ln.strip() for ln in open(label_list).readlines()]
        for t in label_text:
            f.write(t.lower() + "\n")
        gen_vocab(Path(f.name), vocab_dir/smp_filename_prefix, 'unigram', vocab_size)
    
    vocab_path = (vocab_dir/smp_filename_prefix).as_posix() + '.txt'
    print(f"  ✅ Created vocabulary: {vocab_path}")

    def setup_target(target_dir, train, valid, test):
        """Setup target directory with train/valid/test splits"""
        os.makedirs(target_dir, exist_ok=True)
        
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            print(f"  📝 Creating {name}.tsv ({len(data)} samples)")
            
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, _, nf_audio, nf_video in data:
                    # Determine if the file is in 'main' or 'pretrain'
                    if 'pretrain' in fid:
                        prefix = 'pretrain'
                    else:
                        prefix = 'main'
                    clean_fid = fid.replace('main/', '').replace('pretrain/', '')
                    fo.write('\t'.join([
                        clean_fid,
                        os.path.abspath(f"{lrs2_data_dir}/{prefix}/{clean_fid}.mp4"),
                        os.path.abspath(f"{lrs2_data_dir}/{prefix}/{clean_fid}.wav"),
                        str(nf_video),
                        str(nf_audio)
                    ])+'\n')
            
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")
        print(f"  ✅ Copied vocabulary to: {target_dir}/dict.wrd.txt")

    # Read all data
    fids = [x.strip() for x in open(file_list).readlines()]
    labels = [x.strip().lower() for x in open(label_list).readlines()]
    nfs_audio = [x.strip() for x in open(nframes_audio_file).readlines()]
    nfs_video = [x.strip() for x in open(nframes_video_file).readlines()]

    # Read dataset splits
    valid_ids = set(line.strip().split()[0] for line in open(f"{lrs2_data_dir}/val.txt"))
    test_ids = set(line.strip().split()[0] for line in open(f"{lrs2_data_dir}/test.txt"))

    train, valid, test = [], [], []
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        clean_fid = fid.replace('main/', '').replace('pretrain/', '')
        data_item = [fid, label, nf_audio, nf_video]
        
        if clean_fid in test_ids:
            test.append(data_item)
        elif clean_fid in valid_ids:
            valid.append(data_item)
        else:
            train.append(data_item)

    output_dir = metadata_dir
    print(f"📁 Setting up metadata directory: {output_dir}")
    setup_target(output_dir, train, valid, test)
    
    print(f"  📊 Dataset splits:")
    print(f"    • Train: {len(train)} samples")
    print(f"    • Valid: {len(valid)} samples") 
    print(f"    • Test: {len(test)} samples")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description='LRS2 Processing - Count frames and create manifest files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lrs2-data-dir', type=str, required=True,
                        help='LRS2 data directory (contains file.list, label.list, and video files)')
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size for sentencepiece')
    
    args = parser.parse_args()
    
    # Validate input directory
    lrs2_data_dir = Path(args.lrs2_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    
    if not lrs2_data_dir.exists():
        print(f"❌ Error: LRS2 data directory not found: {lrs2_data_dir}")
        return 1
    
    file_list_path = lrs2_data_dir / 'file.list'
    if not file_list_path.exists():
        print(f"❌ Error: file.list not found in: {lrs2_data_dir}")
        return 1
    
    print(f"🚀 Starting LRS2 processing...")
    print(f"📁 Data directory: {lrs2_data_dir}")
    print(f"📁 Metadata directory: {metadata_dir}")
    print(f"📝 Vocabulary size: {args.vocab_size}")
    print("-" * 60)
    
    try:
        # Read file list
        with open(file_list_path, 'r') as f:
            fids = [ln.strip() for ln in f.readlines()]
        print(f"📊 Found {len(fids)} files to process")
        
        # Step 1: Check for missing files
        missing_fids = check_missing_files(fids, str(lrs2_data_dir))
        
        if len(missing_fids) > 0:
            missing_list_path = lrs2_data_dir / 'missing.list'
            with open(missing_list_path, 'w') as fo:
                fo.write('\n'.join(missing_fids) + '\n')
            print(f"❌ Some audio/video files are missing. See: {missing_list_path}")
            print("Please resolve missing files before proceeding.")
            return 1
        
        # Step 2: Count frames
        num_frames = count_frames(fids, str(lrs2_data_dir))
        
        if len(num_frames) == 0:
            print("❌ No valid files found for frame counting")
            return 1
        
        # Step 3: Create frame count files
        nframes_audio_path, nframes_video_path = create_frame_files(
            str(lrs2_data_dir), fids, num_frames
        )
        
        # Step 4: Create manifest files
        output_dir = create_manifest_files(str(lrs2_data_dir), str(metadata_dir), args.vocab_size)
        
        print("-" * 60)
        print("🎉 Processing completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print("📋 Generated files:")
        print(f"   • Frame counts: {nframes_audio_path}, {nframes_video_path}")
        print(f"   • Manifests: train.tsv, valid.tsv, test.tsv")
        print(f"   • Word files: train.wrd, valid.wrd, test.wrd")
        print(f"   • Dictionary: dict.wrd.txt")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
