#!/usr/bin/env python3
"""VoxCeleb2 Step 3: Metadata preparation (Auto-AVSR compatible)

This script creates the same set of metadata artifacts we generate for other
datasets in this repo (e.g. GRID Step 3):

Outputs (written under --metadata-dir):
  - {subset}.tsv            (Auto-AVSR manifest; header is a single "/" line)
  - {subset}.wrd            (one transcript line per example)
  - {subset}.tokens.txt     (one space-separated token-id line per example)
  - dict.wrd.txt            (dictionary from shared SPM units)
  - {dataset}_{subset}_transcript_lengths_seg{SEG}s.csv
        4 columns (no header): dataset,abs_video_path,input_length,token_ids

Assumptions / expected inputs:
  - You already ran preprocessing to produce segmented media under:
        <root-dir>/<dataset>/<dataset>_video_seg{SEG}s/
    containing paired *.mp4 and *.wav files.
  - You already ran ASR inference which created transcript text files under:
        <root-dir>/<dataset>/<dataset>_text_seg{SEG}s/
    mirroring the same relative structure as the video folder.

Matching rule:
  - For each video:  .../<rel>.mp4
    transcript is:  ..._text_seg{SEG}s/<rel>.txt

Tokenization:
  - Uses Voxceleb2/preparation/transforms.py::TextTransform (shared SPM units)
    which produces 1-based token ids (0 is reserved for <blank>).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from scipy.io import wavfile
from tqdm import tqdm

from transforms import TextTransform


def _count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _iter_mp4_files(root: Path):
    # Use rglob for simplicity; Vox2 directory trees can be deep.
    for p in root.rglob("*.mp4"):
        # Skip combined audio-video variants if any
        if p.name.endswith(".m.mp4"):
            continue
        yield p


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf8").strip()
    except FileNotFoundError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxCeleb2 Step 3: create TSV/WRD/tokens/dict and Auto-AVSR CSV"
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        type=str,
        help="Root directory used by preprocessing/asr_infer (contains <dataset>/...)",
    )
    parser.add_argument(
        "--dataset",
        default="vox2",
        type=str,
        help="Dataset name folder under root-dir (default: vox2)",
    )
    parser.add_argument(
        "--subset",
        default="train",
        type=str,
        help="Subset name to write (default: train). Vox2 is usually train-only.",
    )
    parser.add_argument(
        "--seg-duration",
        default=24,
        type=int,
        help="Segment duration in seconds (default: 24)",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        type=str,
        help=(
            "Optional override for segmented video directory. If omitted, uses: "
            "<root-dir>/<dataset>/<dataset>_video_seg{seg}s"
        ),
    )
    parser.add_argument(
        "--text-dir",
        default=None,
        type=str,
        help=(
            "Optional override for transcript directory. If omitted, uses: "
            "<root-dir>/<dataset>/<dataset>_text_seg{seg}s"
        ),
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        type=str,
        help="Where to write metadata outputs",
    )
    parser.add_argument(
        "--max-examples",
        default=None,
        type=int,
        help="Optional cap for quick dry-runs",
    )
    parser.add_argument(
        "--require-transcript",
        action="store_true",
        help="Skip samples without a transcript .txt (default: keep but empty transcript is skipped)",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    dataset = args.dataset
    subset = args.subset
    seg = args.seg_duration

    video_root = (
        Path(args.video_dir)
        if args.video_dir
        else root_dir / dataset / f"{dataset}_video_seg{seg}s"
    )
    text_root = (
        Path(args.text_dir)
        if args.text_dir
        else root_dir / dataset / f"{dataset}_text_seg{seg}s"
    )
    metadata_dir = Path(args.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not video_root.exists():
        raise FileNotFoundError(f"Video dir not found: {video_root}")

    print("Initializing TextTransform...")
    text_transform = TextTransform()
    print(f"✅ token_list size = {len(text_transform.token_list)}")

    # Write dict.wrd.txt (same convention used elsewhere in repo)
    dict_out = metadata_dir / "dict.wrd.txt"
    with open(dict_out, "w", encoding="utf8") as f:
        for idx, token in enumerate(text_transform.token_list):
            if token not in ["<blank>", "<eos>", "<unk>"]:
                f.write(f"{token} {idx}\n")
    print(f"✅ {dict_out.name}")

    tsv_out = metadata_dir / f"{subset}.tsv"
    wrd_out = metadata_dir / f"{subset}.wrd"
    tok_out = metadata_dir / f"{subset}.tokens.txt"
    csv_out = metadata_dir / f"{dataset}_{subset}_transcript_lengths_seg{seg}s.csv"

    mp4_files = list(_iter_mp4_files(video_root))
    mp4_files.sort()
    if args.max_examples is not None:
        mp4_files = mp4_files[: args.max_examples]

    kept = 0
    skipped_no_txt = 0
    skipped_empty_txt = 0
    skipped_no_wav = 0
    skipped_bad_video = 0

    with (
        open(tsv_out, "w", encoding="utf8") as ftsv,
        open(wrd_out, "w", encoding="utf8") as fwrd,
        open(tok_out, "w", encoding="utf8") as ftok,
        open(csv_out, "w", encoding="utf8") as fcsv,
    ):
        # Auto-AVSR manifest header is root path used to interpret relative paths.
        ftsv.write(str(video_root.resolve()) + "\n")

        for mp4_path in tqdm(mp4_files, desc="Building manifests"):
            rel = mp4_path.relative_to(video_root)
            # paired wav should exist (preprocess_vox2.py writes both)
            wav_path = mp4_path.with_suffix(".wav")
            if not wav_path.exists():
                skipped_no_wav += 1
                continue

            txt_path = text_root / rel.with_suffix(".txt")
            if not txt_path.exists():
                if args.require_transcript:
                    skipped_no_txt += 1
                    continue
                transcript = ""
            else:
                transcript = _read_text(txt_path)

            # In practice we want transcripts for AVSR; skip empties.
            # (If you want to keep empties for some reason, we can add a flag.)
            if not transcript:
                skipped_empty_txt += 1
                continue

            # count frames (video) for input_length
            try:
                nframes = _count_video_frames(mp4_path)
            except Exception:
                skipped_bad_video += 1
                continue
            if nframes <= 0:
                skipped_bad_video += 1
                continue

            utt_id = rel.as_posix()[:-4]  # without .mp4
            abs_video = str(mp4_path.resolve())
            abs_audio = str(wav_path.resolve())
            rel_video = rel.as_posix()
            rel_audio = rel.with_suffix('.wav').as_posix()

            token_ids = text_transform.tokenize(transcript)
            token_str = " ".join(str(t.item()) for t in token_ids)
            if not token_str:
                skipped_empty_txt += 1
                continue

            # TSV columns:
            #   id, video_path, audio_path, nframes_video, nframes_audio
            # With header being the root dir, paths are typically relative.
            try:
                nframes_audio = len(wavfile.read(str(wav_path))[1])
            except Exception:
                nframes_audio = 0
            ftsv.write(f"{utt_id}\t{rel_video}\t{rel_audio}\t{nframes}\t{nframes_audio}\n")
            fwrd.write(transcript + "\n")
            ftok.write(token_str + "\n")
            # Auto-AVSR CSV: dataset,abs_video_path,input_length,token_ids
            fcsv.write(f"{dataset},{abs_video},{nframes},{token_str}\n")

            kept += 1

    print("\nDone.")
    print(f"✅ wrote: {tsv_out}")
    print(f"✅ wrote: {wrd_out}")
    print(f"✅ wrote: {tok_out}")
    print(f"✅ wrote: {csv_out}")
    print(f"Kept: {kept}")
    print(f"Skipped (no wav): {skipped_no_wav}")
    print(f"Skipped (no txt): {skipped_no_txt}")
    print(f"Skipped (empty txt): {skipped_empty_txt}")
    print(f"Skipped (bad video): {skipped_bad_video}")


if __name__ == "__main__":
    main()
