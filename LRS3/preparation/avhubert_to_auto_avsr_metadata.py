#!/usr/bin/env python3
"""
Convert AV-HuBERT prepared LRS3 manifests to Auto-AVSR-ready metadata.

Expected input metadata folder contains:
  - train.tsv / train.wrd
  - valid.tsv / valid.wrd
  - test.tsv  / test.wrd

Output metadata folder will contain:
  - train.tsv / valid.tsv / test.tsv
  - train.wrd / valid.wrd / test.wrd
  - dict.wrd.txt                          (from shared repo SPM units)
  - train.tokens.txt / valid.tokens.txt / test.tokens.txt
  - lrs3_train_transcript_lengths_seg16s.csv
  - lrs3_val_transcript_lengths_seg16s.csv
  - lrs3_test_transcript_lengths_seg16s.csv

CSV format (no header):
  dataset,abs_video_path,input_length(nframes_video),token_ids
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REQUIRED_SPLIT_FILES = [
    "train.tsv", "train.wrd",
    "valid.tsv", "valid.wrd",
    "test.tsv", "test.wrd",
]


def resolve_input_metadata_dir(
    input_metadata_dir: str | None,
    avhubert_root: str | None,
    variant: str,
) -> Path:
    """Resolve source metadata directory that contains split TSV/WRD files."""
    if input_metadata_dir:
        src = Path(input_metadata_dir).resolve()
    elif avhubert_root:
        root = Path(avhubert_root).resolve()
        if (root / "train.tsv").exists() and (root / "valid.tsv").exists() and (root / "test.tsv").exists():
            src = root
        else:
            src = root / variant
    else:
        raise ValueError("Provide either --input-metadata-dir or --avhubert-root")

    if not src.exists():
        raise FileNotFoundError(f"Input metadata directory not found: {src}")

    missing = [f for f in REQUIRED_SPLIT_FILES if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required split files in {src}: {', '.join(missing)}"
        )
    return src


def load_shared_spm():
    """Load repo-wide shared SPM model and units."""
    import sentencepiece as spm  # imported lazily to keep --help lightweight

    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "spm" / "unigram" / "unigram5000.model"
    units_path = repo_root / "spm" / "unigram" / "unigram5000_units.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Shared SPM model not found: {model_path}")
    if not units_path.exists():
        raise FileNotFoundError(f"Shared SPM units not found: {units_path}")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    return sp, model_path, units_path


def copy_base_manifests(src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_SPLIT_FILES:
        shutil.copyfile(src_dir / name, out_dir / name)


def write_tokens_and_csvs(out_dir: Path, dataset_name: str = "lrs3") -> None:
    sp, sp_model_path, units_path = load_shared_spm()
    print(f"Using shared SPM model: {sp_model_path}")

    # Keep dict format aligned with existing shared-SPM pipelines in this repo.
    shutil.copyfile(units_path, out_dir / "dict.wrd.txt")
    print(f"✅ Wrote dict.wrd.txt from shared units: {out_dir / 'dict.wrd.txt'}")

    split_to_csv_name = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    for split in ["train", "valid", "test"]:
        tsv_in = out_dir / f"{split}.tsv"
        wrd_in = out_dir / f"{split}.wrd"
        tok_out = out_dir / f"{split}.tokens.txt"
        csv_out = out_dir / f"{dataset_name}_{split_to_csv_name[split]}_transcript_lengths_seg16s.csv"

        with open(tsv_in, "r", encoding="utf8") as ftsv:
            _root = ftsv.readline()  # '/\n'
            tsv_lines = [ln.rstrip("\n") for ln in ftsv if ln.strip()]

        video_paths = []
        nframes_video = []
        for ln in tsv_lines:
            parts = ln.split("\t")
            if len(parts) < 4:
                raise ValueError(f"Malformed TSV line in {tsv_in}: {ln[:200]}")
            video_paths.append(parts[1])
            nframes_video.append(parts[3])

        with open(wrd_in, "r", encoding="utf8") as fwrd:
            wrd_lines = [ln.rstrip("\n") for ln in fwrd]

        if len(video_paths) != len(wrd_lines):
            raise ValueError(
                f"Line count mismatch for {split}: {tsv_in} has {len(video_paths)} examples, "
                f"but {wrd_in} has {len(wrd_lines)} lines"
            )

        with open(tok_out, "w", encoding="utf8") as ftok, open(csv_out, "w", encoding="utf8") as fcsv:
            for vid, nf, text in zip(video_paths, nframes_video, wrd_lines):
                ids = sp.EncodeAsIds((text or "").upper())
                token_str = " ".join(str(i) for i in ids)
                ftok.write(token_str + "\n")
                fcsv.write(f"{dataset_name},{os.path.abspath(vid)},{nf},{token_str}\n")

        print(f"✅ Wrote {tok_out.name}")
        print(f"✅ Wrote {csv_out.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert AV-HuBERT processed LRS3 metadata into Auto-AVSR format (shared SPM)."
    )
    parser.add_argument(
        "--input-metadata-dir",
        type=str,
        default=None,
        help="Directory that directly contains train/valid/test TSV+WRD files.",
    )
    parser.add_argument(
        "--avhubert-root",
        type=str,
        default=None,
        help="AV-HuBERT LRS3 root containing 30h_data / 433h_data (or manifests directly).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="433h_data",
        choices=["30h_data", "433h_data"],
        help="Which AV-HuBERT manifest subset to use when --avhubert-root is provided.",
    )
    parser.add_argument(
        "--output-metadata-dir",
        type=str,
        required=True,
        help="Output directory for converted metadata.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="lrs3",
        help="Dataset name to write in CSV first column.",
    )
    args = parser.parse_args()

    src_dir = resolve_input_metadata_dir(args.input_metadata_dir, args.avhubert_root, args.variant)
    out_dir = Path(args.output_metadata_dir).resolve()

    print("🚀 AV-HuBERT -> Auto-AVSR conversion")
    print(f"📥 Source metadata: {src_dir}")
    print(f"📤 Output metadata: {out_dir}")

    copy_base_manifests(src_dir, out_dir)
    print("✅ Copied train/valid/test .tsv and .wrd")

    write_tokens_and_csvs(out_dir, dataset_name=args.dataset_name)

    print("\n✅ Conversion complete")
    print(f"   • train/valid/test.tsv + .wrd in {out_dir}")
    print(f"   • dict.wrd.txt in {out_dir} (shared SPM units)")
    print(f"   • train/valid/test.tokens.txt in {out_dir}")
    print(f"   • {args.dataset_name}_{{train,val,test}}_transcript_lengths_seg16s.csv in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
