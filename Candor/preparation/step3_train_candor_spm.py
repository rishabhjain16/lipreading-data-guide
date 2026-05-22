#!/usr/bin/env python3
"""
Train a Candor SentencePiece model from metadata `train.wrd`.

Outputs (default under <metadata_dir>/spm{V}/unigram/):
  - unigram{V}.model
  - unigram{V}.vocab
  - unigram{V}_units.txt       (for TextTransform)
  - dict.wrd.txt               (simple freq=1 dict for Fairseq)

Usage:
  python train_candor_spm.py --metadata-dir /path/to/meta --vocab-size 5000

If outputs already exist the script will refuse to overwrite unless `--overwrite` is passed.
"""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import sentencepiece as spm


def create_units_file(spm_txt_path: Path, units_path: Path):
    """Convert sentencepiece .vocab into *_units.txt format for TextTransform.

    SentencePiece .vocab lines look like: token \tscore
    units.txt should be: token id
    """
    print(f"Creating units file: {units_path}")
    with open(spm_txt_path, 'r', encoding='utf8') as fin, open(units_path, 'w', encoding='utf8') as fout:
        for idx, line in enumerate(fin):
            token = line.split()[0]
            fout.write(f"{token} {idx}\n")
    print(f"  -> {units_path}")


def write_dict_wrd(sp_proc: spm.SentencePieceProcessor, out_path: Path):
    """Write a very small dict.wrd.txt where each token has frequency 1 (skipping special tokens)."""
    skipped = {"<unk>", "<blank>", "<s>", "</s>", "<eos>", "<pad>"}
    with open(out_path, 'w', encoding='utf8') as fo:
        for i in range(sp_proc.GetPieceSize()):
            token = sp_proc.IdToPiece(i)
            if token in skipped:
                continue
            fo.write(f"{token} 1\n")
    print(f"  -> dict.wrd.txt: {out_path} ({sp_proc.GetPieceSize()} tokens)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-dir', required=True, help='Directory containing train.wrd (from step2_split_unseen)')
    parser.add_argument('--vocab-size', type=int, required=True, help='Desired SPM vocab size')
    parser.add_argument('--model-type', choices=['unigram', 'bpe', 'word'], default='unigram')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    if not metadata_dir.exists():
        raise FileNotFoundError(f"metadata-dir not found: {metadata_dir}")

    train_wrd = metadata_dir / 'train.wrd'
    if not train_wrd.exists():
        raise FileNotFoundError(f"train.wrd not found in metadata-dir: {train_wrd}")

    spm_dir = metadata_dir / f'spm{args.vocab_size}' / args.model_type
    spm_dir.mkdir(parents=True, exist_ok=True)
    prefix = spm_dir / f'{args.model_type}{args.vocab_size}'

    model_path = Path(str(prefix) + '.model')
    vocab_path = Path(str(prefix) + '.vocab')
    units_path = spm_dir / f'{args.model_type}{args.vocab_size}_units.txt'
    dict_path  = metadata_dir / 'dict.wrd.txt'

    if (model_path.exists() or vocab_path.exists()) and not args.overwrite:
        print('Model or vocab already exists. Use --overwrite to replace.')
        print(f'  model: {model_path}\n  vocab: {vocab_path}')
        return 2

    # Prepare a temporary cleaned input file (lowercased).
    with NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf8') as tmp:
        with open(train_wrd, 'r', encoding='utf8') as fin:
            for line in fin:
                text = line.strip()
                if text:
                    tmp.write(text.lower() + '\n')
        tmp_path = tmp.name

    print(f"Training SPM (type={args.model_type}, vocab={args.vocab_size})...")
    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=str(prefix),
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols='<blank>',
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Load and write auxiliary files
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    create_units_file(vocab_path, units_path)
    write_dict_wrd(sp, dict_path)

    print('\nSPM training complete:')
    print(f'  model : {model_path}')
    print(f'  vocab : {vocab_path}')
    print(f'  units : {units_path}')
    print(f'  dict  : {dict_path}')


if __name__ == '__main__':
    raise SystemExit(main())
