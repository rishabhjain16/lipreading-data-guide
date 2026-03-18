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

import sentencepiece as spm

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
    print("📝 Generating split CSVs (Auto-AVSR format using shared SentencePiece)...")
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Use the repo-wide shared SentencePiece model so token ids match other datasets
    sp, sp_model_path, _ = _load_shared_spm()
    print(f"   Using shared SPM: {sp_model_path}")

    for split_name, split_data in splits.items():
        if not split_data:
            print(f"⚠️ No data for {split_name} split")
            continue

        csv_path = labels_dir / f"candor_{split_name}{crop_suffix}.csv"
        with open(csv_path, 'w', encoding='utf8') as f:
            for record in split_data:
                duration_frames = int(record.get('duration', 0) * 30)
                # SentencePiece in this repo expects uppercased text
                ids = sp.EncodeAsIds((record.get('transcript') or "").upper())
                token_ids = " ".join(str(i) for i in ids)
                f.write(f"candor,{record['video_path']},{duration_frames},{token_ids}\n")

        print(f"   ✅ {csv_path.name}")


def _load_shared_spm():
    """Load the repo-wide shared SentencePiece model.

    Note: shared vocab is uppercase, so callers must normalize text to .upper() before encoding.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sp_model_path = repo_root / "spm" / "unigram" / "unigram5000.model"
    units_path = repo_root / "spm" / "unigram" / "unigram5000_units.txt"
    if not sp_model_path.exists():
        raise FileNotFoundError(f"Shared SPM model not found: {sp_model_path}")
    if not units_path.exists():
        raise FileNotFoundError(f"Shared SPM units not found: {units_path}")
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    return sp, sp_model_path, units_path


def write_inference_files_from_manifests(
    metadata_dir: Path,
    data_dir: Path,
    crop_suffix: str,
    dataset_name: str = "candor",
):
    """Create shared-SPM CSVs in the same 4-column format as LRS2 Auto-AVSR.

    Writes for each split in {train,valid,test} where files exist:
      - <split>.tokens.txt
      - candor_<train|val|test>_transcript_lengths_seg16s{crop_suffix}.csv  (no header)

    CSV format:
      dataset,rel_video_path,input_length(nframes_video),token_ids
    """
    sp, sp_model_path, _ = _load_shared_spm()
    print(f"Using shared SPM model: {sp_model_path}")

    split_to_csvname = {"train": "train", "valid": "val", "test": "test"}

    for split in ["train", "valid", "test"]:
        tsv_in = metadata_dir / f"{split}.tsv"
        wrd_in = metadata_dir / f"{split}.wrd"
        if not tsv_in.exists() or not wrd_in.exists():
            continue

        # Read video paths from TSV (skip first line '/\\n')
        with open(tsv_in, "r", encoding="utf8") as ftsv:
            _root = ftsv.readline()
            tsv_lines = [ln.rstrip("\n") for ln in ftsv]
        video_paths = []
        nframes_video = []
        for ln in tsv_lines:
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                raise ValueError(f"Malformed TSV line in {tsv_in}: {ln[:200]}")
            video_paths.append(parts[1])
            nframes_video.append(int(parts[3]))

        with open(wrd_in, "r", encoding="utf8") as fwrd:
            wrd_lines = [ln.rstrip("\n") for ln in fwrd]

        if len(video_paths) != len(wrd_lines):
            raise ValueError(
                f"Line count mismatch for {split}: {tsv_in} has {len(video_paths)} examples, "
                f"but {wrd_in} has {len(wrd_lines)} lines"
            )

        tokens_out = metadata_dir / f"{split}.tokens.txt"
        csv_split = split_to_csvname[split]
        csv_out = metadata_dir / f"{dataset_name}_{csv_split}_transcript_lengths_seg16s{crop_suffix}.csv"
        with open(tokens_out, "w", encoding="utf8") as ftok, open(csv_out, "w", encoding="utf8") as fc:
            for vid, nf, text in zip(video_paths, nframes_video, wrd_lines):
                # Use SentencePiece ids as-is (0-based) and let the units-file mapping
                # (used by TextTransform) define the 1-based IDs.
                ids = sp.EncodeAsIds((text or "").upper())
                token_str = " ".join(str(i) for i in ids)
                ftok.write(token_str + "\n")
                video_abs = os.path.abspath(vid)
                fc.write(f"{dataset_name},{video_abs},{nf},{token_str}\n")

        print(f"   ✅ {tokens_out.name}")
        print(f"   ✅ {csv_out.name}")


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
    parser.add_argument('--split-ratios', type=str, default='0.8,0.1,0.1',
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
    parser.add_argument('--write-legacy-avsr-csv', action='store_true',
                        help='Also write legacy candor_{split}.csv using TextTransform (4 columns with duration).')
    # (No additional filelist outputs by default)
    parser.add_argument('--write-filelists', action='store_true', dest='write_filelists',
                        help='Also write GRID/TCD-style file.list and label.list to the candor data directory')

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
    # NOTE: We no longer train a Candor-specific SentencePiece model here.
    # We use the repo-wide shared SPM (spm/unigram/unigram5000.model) so token ids
    # are consistent across datasets.
    # =====================================================

    # =====================================================
    # Manifests + tokenized CSVs
    # =====================================================
    generate_training_manifests(data_dir, splits, metadata_dir, crop_suffix)

    # Inference-ready artifacts next to TSV/WRD
    write_inference_files_from_manifests(
        metadata_dir,
        data_dir=data_dir,
        crop_suffix=crop_suffix,
        dataset_name="candor",
    )

    # Optional legacy CSV generation (uses TextTransform + per-dataset units)
    if args.write_legacy_avsr_csv:
        generate_split_csvs(splits, labels_dir, crop_suffix)

    # Optional: write GRID/TCD-style file.list and label.list
    if args.write_filelists:
        def create_file_label_lists(metadata_dir, data_dir):
            """Create file.list and label.list in data_dir using TSV/WRD manifests.

            - file.list: relative paths (relative to data_dir) to video files, one per line
            - label.list: cleaned transcript per line (matching order)
            """
            all_video_paths = []
            all_transcripts = []

            for split in ['train', 'valid', 'test']:
                tsv_in = metadata_dir / f"{split}.tsv"
                wrd_in = metadata_dir / f"{split}.wrd"
                if not tsv_in.exists() or not wrd_in.exists():
                    continue

                with open(tsv_in, 'r', encoding='utf8') as ftsv:
                    _ = ftsv.readline()  # header '/'
                    tsv_lines = [ln.rstrip('\n') for ln in ftsv if ln.strip()]

                with open(wrd_in, 'r', encoding='utf8') as fwrd:
                    wrd_lines = [ln.rstrip('\n') for ln in fwrd]

                if len(tsv_lines) != len(wrd_lines):
                    print(f"⚠️  Line-count mismatch for {split}: {len(tsv_lines)} tsv vs {len(wrd_lines)} wrd")

                for ln, wrd in zip(tsv_lines, wrd_lines):
                    parts = ln.split('\t')
                    if len(parts) < 2:
                        continue
                    video_path = parts[1]
                    # try to relativize against data_dir when possible
                    try:
                        rel = os.path.relpath(video_path, start=str(data_dir))
                    except Exception:
                        rel = video_path
                    all_video_paths.append(rel)
                    all_transcripts.append(clean_transcript(wrd))

            # write to data_dir
            file_list_path = data_dir / 'file.list'
            label_list_path = data_dir / 'label.list'

            with open(file_list_path, 'w', encoding='utf8') as f:
                for p in all_video_paths:
                    f.write(p + '\n')

            with open(label_list_path, 'w', encoding='utf8') as f:
                for t in all_transcripts:
                    f.write(t + '\n')

            print(f"   ✅ Wrote file.list ({len(all_video_paths)} entries) to {file_list_path}")
            print(f"   ✅ Wrote label.list ({len(all_transcripts)} entries) to {label_list_path}")

        create_file_label_lists(metadata_dir, data_dir)

    print("\n✅ Candor Step 2 completed")
    print(f"   • train/valid/test.tsv + .wrd in {metadata_dir}")
    print(f"   • train/valid/test.tokens.txt in {metadata_dir}")
    print(f"   • candor_*_transcript_lengths_seg16s{crop_suffix}.csv in {metadata_dir} (shared SPM, 4 cols)")
    if args.write_legacy_avsr_csv:
        print(f"   • candor_*{crop_suffix}.csv in {labels_dir} (legacy TextTransform CSVs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
