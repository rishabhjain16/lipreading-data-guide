#!/usr/bin/env python3
"""
LRS3 Processing Script
Combines frame counting and manifest creation into one step.
"""

import os
import cv2
import shutil
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
from gen_subword import gen_vocab
from tempfile import NamedTemporaryFile

def load_data_from_csv_files(labels_dir, lrs3_data_dir):
    """Load file IDs and labels from CSV files created by Step 1"""
    print("📊 Loading data from CSV files...")
    
    fids, labels = [], []
    csv_files = {
        'train': 'lrs3_train_transcript_lengths_seg16s.csv',
        'val': 'lrs3_val_transcript_lengths_seg16s.csv', 
        'test': 'lrs3_test_transcript_lengths_seg16s.csv'
    }
    
    for split, csv_file in csv_files.items():
        csv_path = os.path.join(labels_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  ⚠️  Warning: {csv_file} not found, skipping {split}")
            continue
            
        print(f"  📝 Processing {split}: {csv_file}")
        df = pd.read_csv(csv_path, header=None)
        
        for _, row in df.iterrows():
            dataset_name = row[0]  # lrs3
            video_path = row[1]    # path to video file
            length = row[2]        # video length
            token_ids = row[3]     # tokenized text
            
            # Convert token IDs back to text (simplified approach)
            # For now, we'll use the token IDs as placeholder text
            # In a full implementation, you'd need the tokenizer to decode
            
            fids.append(video_path.replace('.mp4', ''))
            labels.append(f"placeholder_text_for_{video_path}")  # Simplified
    
    print(f"  ✅ Loaded {len(fids)} files total")
    return fids, labels

def count_frames(fids, base_dir):
    """Count frames in audio and video files"""
    print("🔄 Counting frames in audio and video files...")
    total_num_frames = []
    
    for fid in tqdm(fids, desc="Counting frames"):
        # Remove lrs3_video_seg16s/ prefix if present to get the relative path
        clean_fid = fid
        if clean_fid.startswith('lrs3_video_seg16s/'):
            clean_fid = clean_fid[len('lrs3_video_seg16s/'):]
        
        wav_fn = os.path.join(base_dir, clean_fid + ".wav")
        video_fn = os.path.join(base_dir, clean_fid + ".mp4")
        
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
        # Remove lrs3_video_seg16s/ prefix if present to get the relative path
        clean_fid = fid
        if clean_fid.startswith('lrs3_video_seg16s/'):
            clean_fid = clean_fid[len('lrs3_video_seg16s/'):]
        
        wav_fn = os.path.join(base_dir, clean_fid + ".wav")
        video_fn = os.path.join(base_dir, clean_fid + ".mp4")
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

def create_frame_files(lrs3_data_dir, fids, num_frames):
    """Create nframes.audio and nframes.video files"""
    print("📝 Creating frame count files...")
    
    audio_num_frames = [x[0] for x in num_frames]
    video_num_frames = [x[1] for x in num_frames]

    nframes_audio_path = os.path.join(lrs3_data_dir, 'nframes.audio')
    nframes_video_path = os.path.join(lrs3_data_dir, 'nframes.video')
    
    with open(nframes_audio_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
    with open(nframes_video_path, 'w') as fo:
        fo.write(''.join([f"{x}\n" for x in video_num_frames]))
    
    print(f"  ✅ Created: {nframes_audio_path}")
    print(f"  ✅ Created: {nframes_video_path}")
    
    return nframes_audio_path, nframes_video_path

def create_manifest_files(lrs3_data_dir, metadata_dir, vocab_size):
    """Create manifest TSV files for train, valid, test"""
    print("📋 Creating manifest files...")
    
    # Validate required files exist
    file_list = os.path.join(lrs3_data_dir, "file.list")
    label_list = os.path.join(lrs3_data_dir, "label.list")
    nframes_audio_file = os.path.join(lrs3_data_dir, "nframes.audio")
    nframes_video_file = os.path.join(lrs3_data_dir, "nframes.video")
    
    for required_file in [file_list, label_list, nframes_audio_file, nframes_video_file]:
        if not os.path.isfile(required_file):
            raise FileNotFoundError(f"Required file not found: {required_file}")

    print("🔤 Generating sentencepiece vocabulary...")
    vocab_dir = (Path(metadata_dir) / f"spm{vocab_size}").absolute()
    vocab_dir.mkdir(parents=True, exist_ok=True)
    smp_filename_prefix = f"spm_unigram{vocab_size}"
    
    with NamedTemporaryFile(mode="w", delete=False) as f:
        label_text = [ln.strip() for ln in open(label_list).readlines()]
        print(f"  📊 Processing {len(label_text)} labels for vocabulary")
        for t in label_text:
            f.write(t.lower() + "\n")
        f.flush()  # Ensure all data is written to the file
        print(f"  📄 Temporary file created: {f.name}")
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
                    # Handle our file ID format: lrs3_video_seg16s/trainval/speaker/video
                    clean_fid = fid
                    if clean_fid.startswith('lrs3_video_seg16s/'):
                        clean_fid = clean_fid[len('lrs3_video_seg16s/'):]
                    
                    # Determine the subset based on path (trainval, test, or pretrain)
                    if 'pretrain' in clean_fid:
                        prefix = 'pretrain'
                        final_fid = clean_fid.replace('pretrain/', '')
                    elif 'test' in clean_fid:
                        prefix = 'test' 
                        final_fid = clean_fid.replace('test/', '')
                    else:  # trainval
                        prefix = 'trainval'
                        final_fid = clean_fid.replace('trainval/', '')
                    
                    fo.write('\t'.join([
                        final_fid,
                        os.path.abspath(f"{lrs3_data_dir}/{prefix}/{final_fid}.mp4"),
                        os.path.abspath(f"{lrs3_data_dir}/{prefix}/{final_fid}.wav"),
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

    # Read dataset splits (LRS3 specific)
    train, valid, test = [], [], []
    
    # For LRS3, we need to handle different split files
    val_split_file = f"{lrs3_data_dir}/val.txt"
    test_split_file = f"{lrs3_data_dir}/test.txt"
    
    valid_ids = set()
    test_ids = set()
    
    if os.path.exists(val_split_file):
        valid_ids = set(line.strip().split()[0] for line in open(val_split_file))
    if os.path.exists(test_split_file):
        test_ids = set(line.strip().split()[0] for line in open(test_split_file))
    
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        # Extract clean file ID for comparison with split files
        # First remove lrs3_video_seg16s/ prefix if present
        clean_fid = fid
        if clean_fid.startswith('lrs3_video_seg16s/'):
            clean_fid = clean_fid[len('lrs3_video_seg16s/'):]
        
        # Then remove directory prefix to get speaker/video format  
        if 'pretrain/' in clean_fid:
            final_fid = clean_fid.replace('pretrain/', '')
        elif 'test/' in clean_fid:
            final_fid = clean_fid.replace('test/', '')
        elif 'trainval/' in clean_fid:
            final_fid = clean_fid.replace('trainval/', '')
        else:
            final_fid = clean_fid
        
        data_item = [fid, label, nf_audio, nf_video]
        
        if 'test' in fid or final_fid in test_ids:
            test.append(data_item)
        elif final_fid in valid_ids:
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
        description='LRS3 Processing - Count frames and create manifest files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lrs3-data-dir', type=str, required=True,
                        help='LRS3 data directory (contains file.list, label.list, and video files)')
    parser.add_argument('--metadata-dir', type=str, required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size for sentencepiece')
    
    args = parser.parse_args()
    
    # Validate input directory
    lrs3_data_dir = Path(args.lrs3_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    
    if not lrs3_data_dir.exists():
        print(f"❌ Error: LRS3 data directory not found: {lrs3_data_dir}")
        return 1
    
    file_list_path = lrs3_data_dir / 'file.list'
    if not file_list_path.exists():
        print(f"❌ Error: file.list not found in: {lrs3_data_dir}")
        print("💡 Try running step2_simple.py first to generate file.list and label.list")
        return 1
    
    print(f"🚀 Starting LRS3 processing...")
    print(f"📁 Data directory: {lrs3_data_dir}")
    print(f"📁 Metadata directory: {metadata_dir}")
    print(f"📝 Vocabulary size: {args.vocab_size}")
    print("-" * 60)
    
    try:
        # Read file list
        with open(file_list_path, 'r') as f:
            fids = [ln.strip() for ln in f.readlines()]
        print(f"📊 Found {len(fids)} files to process")
        
        # Step 1: Check for missing files
        missing_fids = check_missing_files(fids, str(lrs3_data_dir))
        
        if len(missing_fids) > 0:
            missing_list_path = lrs3_data_dir / 'missing.list'
            with open(missing_list_path, 'w') as fo:
                fo.write('\n'.join(missing_fids) + '\n')
            print(f"❌ Some audio/video files are missing. See: {missing_list_path}")
            print("Please resolve missing files before proceeding.")
            return 1
        
        # Step 2: Count frames
        num_frames = count_frames(fids, str(lrs3_data_dir))
        
        if len(num_frames) == 0:
            print("❌ No valid files found for frame counting")
            return 1
        
        # Step 3: Create frame count files
        nframes_audio_path, nframes_video_path = create_frame_files(
            str(lrs3_data_dir), fids, num_frames
        )
        
        # Step 4: Create manifest files
        output_dir = create_manifest_files(str(lrs3_data_dir), str(metadata_dir), args.vocab_size)
        
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
