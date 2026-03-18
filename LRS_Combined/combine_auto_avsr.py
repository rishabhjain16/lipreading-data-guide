#!/usr/bin/env python3
"""Combine Auto-AVSR style CSVs across multiple datasets.

This repo standardizes an Auto-AVSR compatible CSV format (no header):

  dataset,abs_video_path,input_length,token_ids

Where:
- dataset: dataset name string (e.g. grid, lrs2, vox2, ...)
- abs_video_path: absolute path to the mp4
- input_length: usually nframes_video
- token_ids: space-separated integer ids (LRS2-style, 1-based; 0 reserved for <blank>)

This script pools multiple datasets together by concatenating CSV files
split-wise (train/valid/test).

It supports common filename patterns used in this repo:
- train_auto_avsr.csv / valid_auto_avsr.csv / test_auto_avsr.csv
- *_train_transcript_lengths_seg*.csv (train)
- *_val_transcript_lengths_seg*.csv or *_valid_transcript_lengths_seg*.csv (valid)
- *_test_transcript_lengths_seg*.csv (test)

It writes:
- train_auto_avsr.csv
- valid_auto_avsr.csv
- test_auto_avsr.csv
in the output directory.

Example:
  python combine_auto_avsr_csvs.py \
    --d1 /path/to/grid/meta \
    --d2 /path/to/lrs_combined/meta \
    --output /path/to/combined_meta

Notes:
- This is a *CSV-only* combiner. It doesn't merge TSV/WRD/tokens.
- If a dataset doesn't have a particular split CSV, it's skipped for that split.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Optional


SPLITS = ("train", "valid", "test")


def _candidate_csvs_for_split(d: Path, split: str) -> list[Path]:
    # Highest priority: explicit auto_avsr outputs
    direct = d / f"{split}_auto_avsr.csv"
    if direct.exists():
        return [direct]

    # Common per-dataset patterns
    patterns: list[str] = []
    if split == "train":
        patterns = ["*_train_transcript_lengths_seg*.csv"]
    elif split == "valid":
        patterns = ["*_valid_transcript_lengths_seg*.csv", "*_val_transcript_lengths_seg*.csv"]
    elif split == "test":
        patterns = ["*_test_transcript_lengths_seg*.csv"]

    found: list[Path] = []
    for pat in patterns:
        found.extend(sorted(d.glob(pat)))

    # Some datasets (GRID, etc.) may only have a test csv named like grid_test_transcript_lengths_seg16s.csv
    # The glob above already catches that.
    return found


def _iter_rows(csv_path: Path) -> Iterable[tuple[str, str, str, str]]:
    """Yield validated 4-column rows as strings."""
    with open(csv_path, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"Expected 4 columns in {csv_path}, got {len(row)}: {row[:10]}")
            ds, path, in_len, tok = row
            if not ds:
                raise ValueError(f"Empty dataset field in {csv_path}")
            if not path:
                raise ValueError(f"Empty path field in {csv_path}")
            if not in_len.isdigit():
                # allow '0' too
                raise ValueError(f"Non-integer input_length in {csv_path}: {in_len}")
            if tok.strip() == "":
                raise ValueError(f"Empty token_ids in {csv_path} for path {path}")
            yield ds, path, in_len, tok


def _write_combined(split: str, input_dirs: list[Path], out_dir: Path) -> int:
    out_path = out_dir / f"{split}_auto_avsr.csv"
    total = 0

    with open(out_path, "w", encoding="utf8", newline="") as fo:
        writer = csv.writer(fo)

        for d in input_dirs:
            candidates = _candidate_csvs_for_split(d, split)
            if not candidates:
                continue

            # If there are multiple matches (e.g. multiple seg lengths), we take the first and warn.
            chosen = candidates[0]
            if len(candidates) > 1:
                print(f"⚠️  Multiple {split} CSV matches in {d}, using: {chosen.name}")

            n = 0
            for ds, path, in_len, tok in _iter_rows(chosen):
                writer.writerow([ds, path, in_len, tok])
                n += 1

            print(f"✅ {split}: added {n} rows from {chosen}")
            total += n

    if total == 0:
        # Remove empty outputs to avoid confusion
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    return total


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine Auto-AVSR 4-column CSVs across multiple datasets into pooled train/valid/test CSVs."
    )
    parser.add_argument("--output", required=True, help="Output directory to write pooled *_auto_avsr.csv files")

    # Allow up to 12 dataset dirs without forcing users to pass a list syntax.
    for i in range(1, 13):
        parser.add_argument(f"--d{i}", default=None, help=f"Dataset metadata dir {i} (folder containing split CSVs)")

    args = parser.parse_args(argv)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dirs: list[Path] = []
    for i in range(1, 13):
        v = getattr(args, f"d{i}")
        if v:
            p = Path(v).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Input dir not found: {p}")
            input_dirs.append(p)

    if not input_dirs:
        raise ValueError("Provide at least one dataset dir via --d1")

    print("Pooling CSVs from:")
    for d in input_dirs:
        print(f"  - {d}")

    totals = {}
    for split in SPLITS:
        totals[split] = _write_combined(split, input_dirs, out_dir)

    print("\nDone.")
    for split in SPLITS:
        if totals[split] > 0:
            print(f"  ✅ {split}: {totals[split]} rows -> {out_dir}/{split}_auto_avsr.csv")
        else:
            print(f"  ⏭️  {split}: no rows (no output written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
