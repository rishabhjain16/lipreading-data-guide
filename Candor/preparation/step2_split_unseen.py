#!/usr/bin/env python3
"""
Candor Step 2: Reproducible hour-budget splits.

Splits:
  - train   : all remaining sessions
  - valid   : ~10h random non-test sessions
  - test    : ~60h unseen speakers (single-session users only, deduplicated by session)

Canonical split files written/read from --splits-dir:
  - candor-train.id
  - candor-valid.id
  - candor-test.id

Reproducibility:
  - Deterministic selection via sorted candidates + seeded shuffle for valid
  - Split definition persisted as .id files
  - Other researchers can reuse .id files via --use-existing-splits
"""

import os
import cv2
import argparse
import random
import re
from pathlib import Path
from collections import defaultdict
from tempfile import NamedTemporaryFile

import pandas as pd
import sentencepiece as spm
from tqdm import tqdm


# ── Split constants ────────────────────────────────────────────────────────────
TEST_HOURS   = 60.0
VALID_HOURS  = 10.0
DEFAULT_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_transcript(text):
    if not text or str(text).strip() == "":
        return ""
    text = re.sub(r'[:,.!?;\-"()\[\]{}<>@%]', '', str(text))
    text = re.sub(r'--+|\.\.\.+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_crop_type(data_dir):
    return 'face' if '_face' in os.path.basename(str(data_dir).rstrip('/')) else 'lips'


def load_csv_data(labels_dir: Path):
    csv_file = labels_dir / 'candor.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"candor.csv not found in {labels_dir}")
    df = pd.read_csv(csv_file)
    df['session_id'] = df['unique_id'].apply(lambda x: '_'.join(str(x).split('_')[:-2]))
    return df.to_dict('records')


def create_units_file(spm_txt_path, units_path):
    print("📖 Creating *_units.txt for TextTransform...")
    with open(spm_txt_path) as fin, open(units_path, 'w') as fout:
        for idx, line in enumerate(fin):
            token = line.split()[0]
            fout.write(f"{token} {idx}\n")
    print(f"   ✅ Created: {units_path}")


def _normalize_for_spm(text, use_shared_spm):
    txt = text or ""
    return txt.upper() if use_shared_spm else txt.lower()


def _load_shared_spm():
    repo_root = Path(__file__).resolve().parents[2]
    sp_model_path = repo_root / 'spm' / 'unigram' / 'unigram5000.model'
    units_path    = repo_root / 'spm' / 'unigram' / 'unigram5000_units.txt'
    if not sp_model_path.exists():
        raise FileNotFoundError(f"Shared SPM model not found: {sp_model_path}")
    if not units_path.exists():
        raise FileNotFoundError(f"Shared SPM units not found: {units_path}")
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    return sp, sp_model_path, units_path


def _train_candor_spm_from_train_wrd(metadata_dir: Path, vocab_size: int):
    train_wrd = metadata_dir / 'train.wrd'
    if not train_wrd.exists():
        raise FileNotFoundError(f"train.wrd not found: {train_wrd}")

    spm_dir = metadata_dir / f'spm{vocab_size}' / 'unigram'
    spm_dir.mkdir(parents=True, exist_ok=True)
    prefix = spm_dir / f'unigram{vocab_size}'

    print(f"\n🧠 Training Candor SPM (vocab={vocab_size}) from {train_wrd} ...")
    with NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf8') as tmp:
        with open(train_wrd, 'r', encoding='utf8') as fin:
            for line in fin:
                text = line.strip()
                if text:
                    tmp.write(text.lower() + '\n')
        tmp_path = tmp.name

    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type='unigram',
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            user_defined_symbols='<blank>',
        )
    finally:
        os.unlink(tmp_path)

    sp_model_path = Path(str(prefix) + '.model')
    sp_vocab_path = Path(str(prefix) + '.vocab')
    units_path    = spm_dir / f'unigram{vocab_size}_units.txt'
    create_units_file(sp_vocab_path, units_path)

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    print(f"   ✅ SPM model : {sp_model_path}")
    print(f"   ✅ units file: {units_path}")
    return sp, sp_model_path, units_path


# ── Split ID file I/O ──────────────────────────────────────────────────────────
def split_id_paths(splits_dir: Path):
    return {
        'train': splits_dir / 'candor-train.id',
        'valid': splits_dir / 'candor-valid.id',
        'test':  splits_dir / 'candor-test.id',
    }


def save_split_ids(split_sessions: dict, splits_dir: Path):
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, sessions in split_sessions.items():
        out = split_id_paths(splits_dir)[name]
        with open(out, 'w', encoding='utf8') as f:
            for sid in sorted(sessions):
                f.write(sid + '\n')
        print(f"   ✅ candor-{name}.id ({len(sessions)} sessions)")


def load_split_ids(splits_dir: Path) -> dict:
    paths = split_id_paths(splits_dir)
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
    split_sessions = {}
    for name, p in paths.items():
        with open(p, 'r', encoding='utf8') as f:
            split_sessions[name] = [line.strip() for line in f if line.strip()]
    print(f"📂 Loaded pre-created split IDs from {splits_dir}")
    return split_sessions


# ── Core split logic ───────────────────────────────────────────────────────────
def create_hour_budget_split_ids(data: list, seed: int = DEFAULT_SEED) -> dict:
    """
    test  : sessions where ALL speakers are single-session users (~TEST_HOURS)
            → deduplicated by session ID to avoid double-counting shared sessions
    valid : random sample of non-test sessions (~VALID_HOURS)
    train : everything else
    """
    print(
        f"🎯 Hour-budget split "
        f"(unseen speakers → {TEST_HOURS}h | "
        f"valid random → {VALID_HOURS}h | seed={seed})"
    )

    random.seed(seed)

    user_sessions = defaultdict(set)
    session_hours = defaultdict(float)

    for rec in data:
        uid = rec['original_user_id']
        sid = rec['session_id']
        user_sessions[uid].add(sid)
        session_hours[sid] += float(rec.get('duration', 0)) / 3600.0

    # collect sessions where at least one speaker only ever appears in 1 session
    single_session_users = {u for u, sids in user_sessions.items() if len(sids) == 1}

    # deduplicate: collect unique session IDs from those users
    # (two speakers in the same session may both be single-session users)
    candidate_sids = set()
    for u in single_session_users:
        candidate_sids.add(next(iter(user_sessions[u])))

    # sort deterministically: longest session first, then session_id as tiebreaker
    candidate_sids = sorted(candidate_sids, key=lambda s: (-session_hours[s], s))

    test_sessions = set()
    test_hours    = 0.0
    for sid in candidate_sids:
        if test_hours >= TEST_HOURS:
            break
        test_sessions.add(sid)
        test_hours += session_hours[sid]

    n_test_users = sum(
        1 for u in single_session_users
        if next(iter(user_sessions[u])) in test_sessions
    )
    print(f"   test  : {len(test_sessions)} sessions | {n_test_users} users — {test_hours:.1f}h")

    # valid: random shuffle of remaining non-test sessions
    remaining = sorted(sid for sid in session_hours if sid not in test_sessions)
    random.shuffle(remaining)

    valid_sessions = set()
    valid_hours    = 0.0
    for sid in remaining:
        if valid_hours >= VALID_HOURS:
            break
        valid_sessions.add(sid)
        valid_hours += session_hours[sid]

    print(f"   valid : {len(valid_sessions)} sessions — {valid_hours:.1f}h")

    train_sessions = set(session_hours.keys()) - test_sessions - valid_sessions
    print(f"   train : {len(train_sessions)} sessions")

    return {
        'train': sorted(train_sessions),
        'valid': sorted(valid_sessions),
        'test':  sorted(test_sessions),
    }


# ── Materialise splits from IDs ────────────────────────────────────────────────
def materialize_splits(data: list, split_sessions: dict) -> dict:
    session_records = defaultdict(list)
    for rec in data:
        session_records[rec['session_id']].append(rec)

    sets   = {k: set(v) for k, v in split_sessions.items()}
    splits = {k: [] for k in sets}

    for sid in sorted(session_records):
        for split_name, sid_set in sets.items():
            if sid in sid_set:
                splits[split_name].extend(session_records[sid])

    _print_split_summary(splits)
    return splits


def _print_split_summary(splits: dict):
    total = sum(len(v) for v in splits.values())
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                FINAL SPLIT SUMMARY                  │")
    print("  ├──────────────┬────────────┬────────────┬────────────┤")
    print("  │ Split        │  Utterances│      Hours │      % data│")
    print("  ├──────────────┼────────────┼────────────┼────────────┤")
    for name in ['train', 'valid', 'test']:
        records = splits.get(name, [])
        hours   = sum(float(r.get('duration', 0)) for r in records) / 3600.0
        pct     = len(records) / total * 100 if total else 0.0
        print(f"  │ {name:<12} │ {len(records):>10,} │ {hours:>9.1f}h │ {pct:>9.1f}% │")
    print("  ├──────────────┼────────────┼────────────┼────────────┤")
    print(f"  │ {'TOTAL':<12} │ {total:>10,} │            │            │")
    print("  └──────────────┴────────────┴────────────┴────────────┘\n")


# ── Manifest generation ────────────────────────────────────────────────────────
def generate_training_manifests(data_dir: Path, splits: dict, metadata_dir: Path):
    print("📝 Generating .tsv / .wrd manifests...")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    skipped  = []
    skip_log = metadata_dir / 'skip.log'

    for split_name in ['train', 'valid', 'test']:
        split_data = splits.get(split_name, [])
        if not split_data:
            print(f"⚠️  No data for {split_name} — skipping")
            continue

        tsv_path   = metadata_dir / f'{split_name}.tsv'
        wrd_path   = metadata_dir / f'{split_name}.wrd'
        valid_recs = []

        for record in tqdm(split_data, desc=f'  {split_name}'):
            video_path = data_dir / record['video_path']
            audio_path = video_path.with_suffix('.wav')

            if not video_path.exists():
                skipped.append(f"Missing video: {video_path}"); continue
            if not audio_path.exists():
                skipped.append(f"Missing audio: {audio_path}"); continue

            cap         = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps         = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if frame_count <= 0 or fps <= 0:
                skipped.append(f"Invalid frame/fps: {video_path}"); continue

            duration     = frame_count / fps
            audio_frames = int(duration * 16000)
            valid_recs.append((
                record['unique_id'],
                str(video_path),
                str(audio_path),
                frame_count,
                audio_frames,
                clean_transcript(record['transcript'])
            ))

        with open(tsv_path, 'w', encoding='utf8') as f:
            f.write('/\n')
            for r in valid_recs:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n")

        with open(wrd_path, 'w', encoding='utf8') as f:
            for r in valid_recs:
                f.write(r[5] + '\n')

        print(f"   ✅ {split_name}: {len(valid_recs):,}/{len(split_data):,} valid")

    if skipped:
        with open(skip_log, 'w', encoding='utf8') as f:
            for s in skipped: f.write(s + '\n')
        print(f"⚠️  {len(skipped)} files skipped — see {skip_log}")


# ── Inference CSV + token files ────────────────────────────────────────────────
def write_inference_files(metadata_dir: Path, crop_suffix: str, sp,
                          use_shared_spm: bool, dataset_name: str = 'candor'):
    print("📝 Writing inference CSVs...")
    split_to_label = {'train': 'train', 'valid': 'val', 'test': 'test'}

    for split in ['train', 'valid', 'test']:
        tsv_in = metadata_dir / f'{split}.tsv'
        wrd_in = metadata_dir / f'{split}.wrd'
        if not tsv_in.exists() or not wrd_in.exists():
            continue

        with open(tsv_in, 'r', encoding='utf8') as f:
            _ = f.readline()
            tsv_lines = [ln.rstrip('\n') for ln in f if ln.strip()]

        video_paths, nframes = [], []
        for ln in tsv_lines:
            parts = ln.split('\t')
            if len(parts) < 4:
                raise ValueError(f"Malformed TSV line: {ln[:200]}")
            video_paths.append(parts[1])
            nframes.append(int(parts[3]))

        with open(wrd_in, 'r', encoding='utf8') as f:
            wrd_lines = [ln.rstrip('\n') for ln in f]

        if len(video_paths) != len(wrd_lines):
            raise ValueError(f"Line count mismatch for {split}")

        tokens_out = metadata_dir / f'{split}.tokens.txt'
        csv_label  = split_to_label[split]
        csv_out    = metadata_dir / f'{dataset_name}_{csv_label}_transcript_lengths_seg16s{crop_suffix}.csv'

        with open(tokens_out, 'w', encoding='utf8') as ft, \
             open(csv_out,    'w', encoding='utf8') as fc:
            for vid, nf, text in zip(video_paths, nframes, wrd_lines):
                ids       = sp.EncodeAsIds(_normalize_for_spm(text, use_shared_spm))
                token_str = ' '.join(str(i) for i in ids)
                ft.write(token_str + '\n')
                fc.write(f"{dataset_name},{os.path.abspath(vid)},{nf},{token_str}\n")

        print(f"   ✅ {tokens_out.name}")
        print(f"   ✅ {csv_out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Candor Step 2: reproducible hour-budget manifests (unseen test only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--candor-data-dir',    required=True)
    parser.add_argument('--metadata-dir',        required=True)
    parser.add_argument('--splits-dir',          default=None,
                        help='Folder to write/read candor-*.id files')
    parser.add_argument('--use-existing-splits', action='store_true',
                        help='Load pre-created .id files instead of generating new ones')
    parser.add_argument('--seed',                type=int, default=DEFAULT_SEED)
    parser.add_argument('--vocab-size',          type=int, default=None,
                        help='Train a Candor SPM; omit to use shared repo SPM')
    args = parser.parse_args()

    data_dir     = Path(args.candor_data_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    splits_dir   = Path(args.splits_dir).resolve() if args.splits_dir \
                   else metadata_dir / 'splits'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    labels_dir = None
    for cand in [data_dir / 'labels', data_dir.parent / 'labels']:
        if cand.exists():
            labels_dir = cand
            break
    if labels_dir is None:
        raise FileNotFoundError('Labels dir not found')

    crop_type   = detect_crop_type(data_dir)
    crop_suffix = f'_{crop_type}' if crop_type != 'lips' else ''

    print('=' * 62)
    print('  🎯 Candor Step 2 — Reproducible Hour-Budget Manifests')
    print('=' * 62)
    print(f'  Data dir     : {data_dir}')
    print(f'  Labels dir   : {labels_dir}')
    print(f'  Metadata dir : {metadata_dir}')
    print(f'  Splits dir   : {splits_dir}')
    print(f'  Crop type    : {crop_type}')
    print(f'  test target  : {TEST_HOURS}h  (unseen speakers)')
    print(f'  valid target : {VALID_HOURS}h')
    print(f'  seed         : {args.seed}')
    print('=' * 62)
    print()

    all_data = load_csv_data(labels_dir)
    print(f"📂 Loaded {len(all_data):,} utterances from {labels_dir / 'candor.csv'}\n")

    if args.use_existing_splits:
        split_sessions = load_split_ids(splits_dir)
    else:
        split_sessions = create_hour_budget_split_ids(all_data, seed=args.seed)
        print('💾 Saving canonical split IDs...')
        save_split_ids(split_sessions, splits_dir)
        print()

    splits = materialize_splits(all_data, split_sessions)
    generate_training_manifests(data_dir, splits, metadata_dir)

    use_shared_spm = args.vocab_size is None
    if use_shared_spm:
        sp, sp_model_path, _ = _load_shared_spm()
        print(f"🔤 Using shared SPM: {sp_model_path}")
    else:
        sp, sp_model_path, _ = _train_candor_spm_from_train_wrd(metadata_dir, args.vocab_size)
        print(f"🔤 Using trained Candor SPM: {sp_model_path}")

    write_inference_files(metadata_dir, crop_suffix, sp, use_shared_spm)

    print()
    print('✅ Done')
    print(f'   Split IDs      → {splits_dir}')
    print(f'   TSV/WRD/tokens → {metadata_dir}')


if __name__ == '__main__':
    raise SystemExit(main())