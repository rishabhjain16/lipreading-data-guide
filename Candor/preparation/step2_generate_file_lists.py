#!/usr/bin/env python3
"""
Candor Step 2: Generate Training Manifests (SentencePiece compatible with TextTransform)

This version:
✓ trains SentencePiece
✓ auto-creates *_units.txt
✓ keeps TextTransform() unchanged
✓ supports official / session / speaker splits
✓ same behavior as LRS2/Auto-AVSR pipeline
"""

import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random
import re
import shutil
from tempfile import NamedTemporaryFile

from gen_subword import gen_vocab
from transforms import TextTransform


# =========================================================
# helpers
# =========================================================

def create_units_file(spm_txt_path, units_path):
    """
    Convert sentencepiece .txt vocab into *_units.txt format.

    SentencePiece .txt:
        token  score
    units.txt:
        token id
    """
    print("📖 Creating *_units.txt for TextTransform...")
    with open(spm_txt_path) as fin, open(units_path, "w") as fout:
        for idx, line in enumerate(fin):
            token = line.split()[0]
            fout.write(f"{token} {idx}\n")
    print(f"   ✅ Created: {units_path}")


def clean_transcript(text):
    if not text or text.strip() == "":
        return ""
    text = re.sub(r'[:,.!?;\-"()\[\]{}<>@%]', '', text)
    text = re.sub(r'--+|\.\.\.+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def generate_dict_file(vocab_path, output_path):
    shutil.copyfile(vocab_path, output_path)
    return output_path


def detect_crop_type(data_dir):
    return 'face' if '_face' in os.path.basename(data_dir.rstrip('/')) else 'lips'


def load_csv_data(labels_dir):
    csv_file = labels_dir / 'candor.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"candor.csv not found in {labels_dir}")
    df = pd.read_csv(csv_file)
    return df.to_dict('records')


def load_official_splits(splits_dir):
    splits_dir = Path(splits_dir)
    splits = {}
    for split_name in ['train', 'valid', 'test']:
        split_file = splits_dir / f"candor-{split_name}.id"
        if not split_file.exists():
            raise FileNotFoundError(f"Official split file not found: {split_file}")
        with open(split_file) as f:
            sessions = [line.strip() for line in f if line.strip()]
        splits[split_name] = sessions
        print(f"📄 Loaded {split_name}: {len(sessions)} sessions")
    return splits


def split_data_by_official_splits(data, official_splits):
    print("🎯 Using official session splits")
    session_data = defaultdict(list)
    for record in data:
        unique_id = record['unique_id']
        session_id = '_'.join(unique_id.split('_')[:-2])
        session_data[session_id].append(record)

    splits = {'train': [], 'valid': [], 'test': []}
    for split_name, session_list in official_splits.items():
        for session_id in session_list:
            if session_id in session_data:
                splits[split_name].extend(session_data[session_id])
            else:
                print(f"⚠️  Session {session_id} in {split_name} split not found in data")

    for k in splits:
        print(f"  {k}: {len(splits[k])} samples")
    return splits


def split_data_by_session(data, split_ratios, seed=42):
    print(f"🎯 Splitting data by session with ratios: {split_ratios}")
    session_data = defaultdict(list)
    for record in data:
        unique_id = record['unique_id']
        session_id = '_'.join(unique_id.split('_')[:-2])
        session_data[session_id].append(record)

    sessions = list(session_data.keys())
    random.seed(seed)
    random.shuffle(sessions)

    n = len(sessions)
    train_end = int(n * split_ratios['train'])
    valid_end = train_end + int(n * split_ratios['valid'])

    train_sessions = sessions[:train_end]
    valid_sessions = sessions[train_end:valid_end]
    test_sessions = sessions[valid_end:]

    splits = {'train': [], 'valid': [], 'test': []}
    for s in train_sessions:
        splits['train'].extend(session_data[s])
    for s in valid_sessions:
        splits['valid'].extend(session_data[s])
    for s in test_sessions:
        splits['test'].extend(session_data[s])

    print(f"📊 Session splits: train={len(train_sessions)}, valid={len(valid_sessions)}, test={len(test_sessions)}")
    print(f"📈 Sample counts: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    return splits


def split_data_by_speaker(data, split_ratios, seed=42):
    print(f"🎯 Splitting data by speaker with ratios: {split_ratios}")
    speaker_data = defaultdict(list)
    for record in data:
        speaker_data[record['speaker_id']].append(record)

    speakers = list(speaker_data.keys())
    random.seed(seed)
    random.shuffle(speakers)

    n = len(speakers)
    train_end = int(n * split_ratios['train'])
    valid_end = train_end + int(n * split_ratios['valid'])

    train_speakers = speakers[:train_end]
    valid_speakers = speakers[train_end:valid_end]
    test_speakers = speakers[valid_end:]

    splits = {'train': [], 'valid': [], 'test': []}
    for spk in train_speakers:
        splits['train'].extend(speaker_data[spk])
    for spk in valid_speakers:
        splits['valid'].extend(speaker_data[spk])
    for spk in test_speakers:
        splits['test'].extend(speaker_data[spk])

    print(f"📊 Speaker splits: train={len(train_speakers)}, valid={len(valid_speakers)}, test={len(test_speakers)}")
    print(f"📈 Sample counts: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    return splits


# =========================================================
# Tokenized CSVs (TextTransform only)
# =========================================================
def generate_split_csvs(splits, labels_dir, crop_suffix):
    print("📝 Generating split CSVs (Auto-AVSR format with TextTransform)...")
    labels_dir.mkdir(parents=True, exist_ok=True)

    text_transform = TextTransform()  # uses *_units.txt automatically

    for split_name, split_data in splits.items():
        if not split_data:
            print(f"⚠️ No data for {split_name} split")
            continue

        csv_path = labels_dir / f"candor_{split_name}{crop_suffix}.csv"
        with open(csv_path, 'w') as f:
            for record in split_data:
                duration_frames = int(record.get('duration', 0) * 30)
                token_ids = " ".join(
                    str(t.item()) for t in text_transform.tokenize(record['transcript'])
                )
                f.write(f"candor,{record['video_path']},{duration_frames},{token_ids}\n")

        print(f"   ✅ {csv_path.name}")


# =========================================================
# Training manifests
# =========================================================
def generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix):
    print("📝 Generating .tsv/.wrd manifests...")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    skip_log_path = metadata_dir / 'skip.log'
    skipped = []

    for split_name, split_data in splits.items():
        if not split_data:
            print(f"⚠️ No data for {split_name} split")
            continue

        tsv_path = metadata_dir / f"{split_name}.tsv"
        wrd_path = metadata_dir / f"{split_name}.wrd"

        valid_records = []

        for record in tqdm(split_data, desc=f"{split_name}"):
            video_path = data_dir / record['video_path']
            audio_path = video_path.with_suffix('.wav')

            if not video_path.exists():
                skipped.append(f"Missing video: {video_path}")
                continue
            if not audio_path.exists():
                skipped.append(f"Missing audio: {audio_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if frame_count <= 0 or fps <= 0:
                skipped.append(f"Invalid frame/fps: {video_path}")
                continue

            duration = frame_count / fps
            audio_frames = int(duration * 16000)

            valid_records.append(
                (record['unique_id'], video_path, audio_path, frame_count, audio_frames,
                 clean_transcript(record['transcript']))
            )

        with open(tsv_path, 'w') as f:
            f.write('/\n')
            for r in valid_records:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n")

        with open(wrd_path, 'w') as f:
            for r in valid_records:
                f.write(r[5] + "\n")

        print(f"   ✅ {split_name}: {len(valid_records)}/{len(split_data)} valid samples")

    if skipped:
        with open(skip_log_path, 'w') as f:
            for s in skipped:
                f.write(s + "\n")
        print(f"⚠️ {len(skipped)} files skipped – see {skip_log_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description='Candor Step 2: Generate training manifests (.tsv/.wrd + Auto-AVSR CSVs)'
    )
    parser.add_argument('--candor-data-dir', required=True,
                        help='Candor processed data directory (contains video files)')
    parser.add_argument('--metadata-dir', required=True,
                        help='Directory where metadata files will be created')
    parser.add_argument('--split-ratios', type=str, default='0.7,0.15,0.15',
                        help='Train/val/test split ratios (comma-separated)')
    parser.add_argument('--split-by', type=str, default='session', choices=['session', 'speaker'],
                        help='Split by session or speaker')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    parser.add_argument('--use-official-splits', action='store_true',
                        help='Use official train/val/test splits')
    parser.add_argument('--splits-dir', type=str, default='./splits',
                        help='Directory containing candor-train.id, candor-valid.id, candor-test.id')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size for SentencePiece model')

    args = parser.parse_args()

    # parse split ratios
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-3:
        print("❌ Split ratios must be three numbers that sum to 1.0")
        return 1
    split_ratios = {'train': ratios[0], 'valid': ratios[1], 'test': ratios[2]}

    data_dir = Path(args.candor_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    if not data_dir.exists():
        print(f"❌ Candor data directory not found: {data_dir}")
        return 1

    crop_type = detect_crop_type(str(data_dir))
    crop_suffix = f"_{crop_type}" if crop_type != "lips" else ""

    # locate labels dir
    labels_dir = None
    for cand in [data_dir / "labels", data_dir.parent / "labels"]:
        if cand.exists():
            labels_dir = cand
            break
    if labels_dir is None:
        print("❌ Labels directory not found (looked in data_dir/labels and parent/labels)")
        return 1

    print("🎯 Candor Step 2")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"📁 Metadata directory: {metadata_dir}")
    print(f"✂️ Crop type: {crop_type}")

    # load data
    all_data = load_csv_data(labels_dir)

    # create splits
    if args.use_official_splits:
        print(f"\n📂 Loading official splits from: {args.splits_dir}")
        official_splits = load_official_splits(args.splits_dir)
        splits = split_data_by_official_splits(all_data, official_splits)
    else:
        print(f"\n📈 Creating splits (split-by={args.split_by}, ratios={split_ratios})")
        if args.split_by == 'session':
            splits = split_data_by_session(all_data, split_ratios, args.seed)
        else:
            splits = split_data_by_speaker(all_data, split_ratios, args.seed)

    # =====================================================
    # Train SentencePiece
    # =====================================================
    print("\n🔤 Training SentencePiece...")
    vocab_dir = (metadata_dir / f"spm{args.vocab_size}").absolute()
    vocab_dir.mkdir(parents=True, exist_ok=True)
    prefix = vocab_dir / f"spm_unigram{args.vocab_size}"

    with NamedTemporaryFile("w", delete=False) as f:
        for r in splits['train']:
            t = clean_transcript(r['transcript'])
            if t:
                f.write(t + "\n")
        f.flush()
        print(f"  📊 Training on {len(splits['train'])} transcripts")
        gen_vocab(Path(f.name), prefix, 'unigram', args.vocab_size)

    vocab_txt = prefix.with_suffix(".txt")
    spm_model = prefix.with_suffix(".model")
    units_path = prefix.with_name(prefix.name + "_units.txt")

    create_units_file(vocab_txt, units_path)
    generate_dict_file(vocab_txt, metadata_dir / "dict.wrd.txt")

    print(f"   ✅ SentencePiece model: {spm_model}")
    print(f"   ✅ Vocab txt: {vocab_txt}")
    print(f"   ✅ Units txt: {units_path}")

    # =====================================================
    # Manifests + tokenized CSVs
    # =====================================================
    generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix)
    generate_split_csvs(splits, labels_dir, crop_suffix)

    print("\n✅ Candor Step 2 completed")
    print(f"   • train/valid/test.tsv + .wrd in {metadata_dir}")
    print(f"   • dict.wrd.txt in {metadata_dir}")
    print(f"   • SentencePiece files in {vocab_dir}")
    print(f"   • candor_*{crop_suffix}.csv in {labels_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
