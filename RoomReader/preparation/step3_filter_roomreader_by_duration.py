#!/usr/bin/env python3
"""
Filter processed RoomReader dataset by clip duration and copy kept samples.

What it does:
1) Scans processed RoomReader video folders under --src-root (roomreader_video*).
2) Computes duration from each .wav file.
3) Keeps samples with duration strictly greater than --min-duration-sec and
    (optionally) less than or equal to --max-duration-sec.
4) Copies matching files to --dst-root while preserving folder layout:
   - video/audio: roomreader_video*/<mode>/<session>/<id>.mp4|.wav
   - text:        roomreader_text*/<mode>/<session>/<id>.txt
   - AV mode:     roomreader_av*/<mode>/<session>/<id>_av.mp4 and <id>.txt
5) Writes filtered CSVs in dst labels/ for roomreader_*.csv files if present.

Example:
    python preparation/filter_roomreader_by_duration.py \
        --src-root /media/rishabhjain/SSD/Data/Roomreader-AV \
        --dst-root /media/rishabhjain/SSD/Data/Roomreader-AV-filtered \
        --min-duration-sec 1.0

    # Keep only (1.0, 30.0] seconds
    python preparation/filter_roomreader_by_duration.py \
        --src-root /media/rishabhjain/SSD/Data/Roomreader-AV \
        --dst-root /media/rishabhjain/SSD/Data/Roomreader-AV-filtered-1to30 \
        --min-duration-sec 1.0 \
        --max-duration-sec 30.0
"""

from __future__ import annotations

import argparse
import csv
import shutil
import wave
from pathlib import Path
from typing import Dict, List, Set, Tuple


def wav_duration_seconds(wav_path: Path) -> float:
    """Get WAV duration in seconds using stdlib only."""
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        framerate = wf.getframerate()
        if framerate <= 0:
            return 0.0
        return float(nframes) / float(framerate)


def safe_copy(src: Path, dst: Path) -> bool:
    """Copy file if source exists; create parent dirs. Returns True if copied."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def parse_video_root(video_root: Path) -> Tuple[str, str]:
    """Split roomreader_video_lips[_det]/mode path into (video_root_name, mode)."""
    # Expected: <src_root>/<video_root_name>/<mode>
    mode = video_root.name
    video_root_name = video_root.parent.name
    return video_root_name, mode


def corresponding_root_name(video_root_name: str, target_prefix: str) -> str:
    """Map roomreader_video_* -> roomreader_text_* or roomreader_av_*"""
    if video_root_name.startswith("roomreader_video"):
        return video_root_name.replace("roomreader_video", target_prefix, 1)
    return f"{target_prefix}_{video_root_name}"


def read_transcript_if_exists(txt_path: Path) -> str:
    """Read transcript text if present; return empty string otherwise."""
    if not txt_path.exists():
        return ""
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def print_duration_histogram(durations: List[float], max_sec: int = 30) -> None:
    """Print histogram buckets: 0-1, 1-2, ..., 29-30, and >30 seconds."""
    bins = [0 for _ in range(max_sec)]  # index i => [i, i+1)
    overflow = 0

    for d in durations:
        if d < 0:
            continue
        if d >= max_sec:
            overflow += 1
        else:
            bins[int(d)] += 1

    print("\n=== Duration stats (all scanned clips) ===")
    for i, count in enumerate(bins):
        print(f"{i:02d}-{i+1:02d}s: {count}")
    print(f">={max_sec}s: {overflow}")


def collect_kept_ids_and_copy(
    src_root: Path,
    dst_root: Path,
    min_duration: float,
    max_duration: float | None = None,
) -> Dict[str, Set[str]]:
    """
    Copy kept media/text/av files and return kept IDs by mode.

    Returns:
        dict: {"individual": {id1, id2}, "conversational": {...}}
    """
    kept_by_mode: Dict[str, Set[str]] = {}

    # Find all mode folders under roomreader_video* roots
    video_mode_dirs = []
    for video_parent in sorted(src_root.glob("roomreader_video*")):
        if not video_parent.is_dir():
            continue
        for mode_dir in sorted(video_parent.iterdir()):
            if mode_dir.is_dir() and mode_dir.name in {"individual", "conversational"}:
                video_mode_dirs.append(mode_dir)

    if not video_mode_dirs:
        raise FileNotFoundError(
            f"No processed RoomReader video folders found under: {src_root}\n"
            "Expected folders like roomreader_video_lips/individual/..."
        )

    copied = {
        "video_mp4": 0,
        "audio_wav": 0,
        "text_txt": 0,
        "av_mp4": 0,
        "av_txt": 0,
    }
    total_seen = 0
    total_kept = 0
    all_durations: List[float] = []
    dropped_lt_threshold_lines: List[str] = []
    dropped_gt_threshold_lines: List[str] = []

    wav_records = []

    for mode_dir in video_mode_dirs:
        video_root_name, mode = parse_video_root(mode_dir)
        text_root_name = corresponding_root_name(video_root_name, "roomreader_text")
        av_root_name = corresponding_root_name(video_root_name, "roomreader_av")

        kept_by_mode.setdefault(mode, set())

        for wav_path in mode_dir.rglob("*.wav"):
            clip_id = wav_path.stem
            duration = wav_duration_seconds(wav_path)

            # Relative session path under mode dir, e.g. S01/abc.wav
            rel_under_mode = wav_path.relative_to(mode_dir)
            session_rel = rel_under_mode.parent

            wav_records.append({
                "video_root_name": video_root_name,
                "text_root_name": text_root_name,
                "av_root_name": av_root_name,
                "mode": mode,
                "wav_path": wav_path,
                "clip_id": clip_id,
                "duration": duration,
                "session_rel": session_rel,
            })

    # Print stats window before copying/filtering
    all_durations = [float(r["duration"]) for r in wav_records]
    print_duration_histogram(all_durations, max_sec=30)

    # Filter + copy pass
    for rec in wav_records:
        total_seen += 1
        duration = float(rec["duration"])

        if duration < min_duration:
            rel = rec["wav_path"].relative_to(src_root)
            clip_id = str(rec["clip_id"])
            mode = str(rec["mode"])
            text_root_name = str(rec["text_root_name"])
            session_rel = Path(rec["session_rel"])
            txt_path = src_root / text_root_name / mode / session_rel / f"{clip_id}.txt"
            transcript = read_transcript_if_exists(txt_path).replace("\n", " ").replace("\t", " ")
            dropped_lt_threshold_lines.append(f"{rel}\t{duration:.6f}\t{transcript}")

        if max_duration is not None and duration > max_duration:
            rel = rec["wav_path"].relative_to(src_root)
            clip_id = str(rec["clip_id"])
            mode = str(rec["mode"])
            text_root_name = str(rec["text_root_name"])
            session_rel = Path(rec["session_rel"])
            txt_path = src_root / text_root_name / mode / session_rel / f"{clip_id}.txt"
            transcript = read_transcript_if_exists(txt_path).replace("\n", " ").replace("\t", " ")
            dropped_gt_threshold_lines.append(f"{rel}\t{duration:.6f}\t{transcript}")

        if not (duration > min_duration):
            continue

        if max_duration is not None and duration > max_duration:
            continue

        clip_id = str(rec["clip_id"])
        mode = str(rec["mode"])
        video_root_name = str(rec["video_root_name"])
        text_root_name = str(rec["text_root_name"])
        av_root_name = str(rec["av_root_name"])
        session_rel = Path(rec["session_rel"])
        wav_path = Path(rec["wav_path"])

        total_kept += 1
        kept_by_mode[mode].add(clip_id)

        src_video_mp4 = wav_path.with_suffix(".mp4")
        src_audio_wav = wav_path
        src_text_txt = src_root / text_root_name / mode / session_rel / f"{clip_id}.txt"
        src_av_mp4 = src_root / av_root_name / mode / session_rel / f"{clip_id}_av.mp4"
        src_av_txt = src_root / av_root_name / mode / session_rel / f"{clip_id}.txt"

        dst_video_mp4 = dst_root / video_root_name / mode / session_rel / f"{clip_id}.mp4"
        dst_audio_wav = dst_root / video_root_name / mode / session_rel / f"{clip_id}.wav"
        dst_text_txt = dst_root / text_root_name / mode / session_rel / f"{clip_id}.txt"
        dst_av_mp4 = dst_root / av_root_name / mode / session_rel / f"{clip_id}_av.mp4"
        dst_av_txt = dst_root / av_root_name / mode / session_rel / f"{clip_id}.txt"

        if safe_copy(src_video_mp4, dst_video_mp4):
            copied["video_mp4"] += 1
        if safe_copy(src_audio_wav, dst_audio_wav):
            copied["audio_wav"] += 1
        if safe_copy(src_text_txt, dst_text_txt):
            copied["text_txt"] += 1
        if safe_copy(src_av_mp4, dst_av_mp4):
            copied["av_mp4"] += 1
        if safe_copy(src_av_txt, dst_av_txt):
            copied["av_txt"] += 1

    dropped_log_path = dst_root / f"dropped_lt_{str(min_duration).replace('.', 'p')}s.txt"
    dropped_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dropped_log_path, "w", encoding="utf-8") as f:
        f.write("# Files with duration below threshold\n")
        f.write(f"# Threshold: < {min_duration:.6f} seconds\n")
        f.write("# Format: relative_wav_path<TAB>duration_seconds<TAB>transcript\n")
        for line in dropped_lt_threshold_lines:
            f.write(line + "\n")

    dropped_gt_log_path = None
    if max_duration is not None:
        dropped_gt_log_path = dst_root / f"dropped_gt_{str(max_duration).replace('.', 'p')}s.txt"
        dropped_gt_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dropped_gt_log_path, "w", encoding="utf-8") as f:
            f.write("# Files with duration above threshold\n")
            f.write(f"# Threshold: > {max_duration:.6f} seconds\n")
            f.write("# Format: relative_wav_path<TAB>duration_seconds<TAB>transcript\n")
            for line in dropped_gt_threshold_lines:
                f.write(line + "\n")

    print("\n=== Duration filtering summary ===")
    print(f"Source root: {src_root}")
    print(f"Destination root: {dst_root}")
    print(f"Min duration (strict): > {min_duration:.3f}s")
    if max_duration is not None:
        print(f"Max duration (strict): <= {max_duration:.3f}s")
    print(f"Total clips scanned: {total_seen}")
    print(f"Total clips kept: {total_kept}")
    print(f"Total clips removed: {total_seen - total_kept}")
    print(f"Logged (<{min_duration:.3f}s) files: {len(dropped_lt_threshold_lines)}")
    print(f"Dropped log path: {dropped_log_path}")
    if max_duration is not None:
        print(f"Logged (>{max_duration:.3f}s) files: {len(dropped_gt_threshold_lines)}")
        print(f"Dropped long-file log path: {dropped_gt_log_path}")
    print("Copied files:")
    print(f"  mp4 (video): {copied['video_mp4']}")
    print(f"  wav (audio): {copied['audio_wav']}")
    print(f"  txt (text):  {copied['text_txt']}")
    print(f"  mp4 (AV):    {copied['av_mp4']}")
    print(f"  txt (AV):    {copied['av_txt']}")

    for mode, ids in kept_by_mode.items():
        print(f"  Kept IDs in {mode}: {len(ids)}")

    return kept_by_mode


def filter_label_csvs(src_root: Path, dst_root: Path, kept_by_mode: Dict[str, Set[str]]) -> None:
    """Filter roomreader_*.csv files by kept unique_id and write to destination labels/."""
    src_labels = src_root / "labels"
    if not src_labels.exists():
        print("ℹ️ No labels/ folder found in source. Skipping CSV filtering.")
        return

    dst_labels = dst_root / "labels"
    dst_labels.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(src_labels.glob("roomreader_*.csv"))
    if not csv_files:
        print("ℹ️ No roomreader_*.csv files found in labels/. Skipping CSV filtering.")
        return

    print("\n=== Filtering label CSV files ===")
    for csv_path in csv_files:
        mode = "conversational" if "conversational" in csv_path.name.lower() else "individual"
        keep_ids = kept_by_mode.get(mode, set())

        dst_csv = dst_labels / csv_path.name
        kept_rows = 0
        total_rows = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as src_f:
            reader = csv.DictReader(src_f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print(f"⚠️ Skipping empty CSV: {csv_path.name}")
                continue

            with open(dst_csv, "w", encoding="utf-8", newline="") as dst_f:
                writer = csv.DictWriter(dst_f, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    total_rows += 1
                    uid = row.get("unique_id", "")
                    if uid in keep_ids:
                        writer.writerow(row)
                        kept_rows += 1

        print(f"{csv_path.name}: kept {kept_rows}/{total_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter processed RoomReader data by duration and copy kept files to a new root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-root", required=True, help="Path to processed RoomReader root (contains roomreader_video*, roomreader_text*, labels, etc.)")
    parser.add_argument("--dst-root", required=True, help="Path to write filtered RoomReader dataset")
    parser.add_argument("--min-duration-sec", type=float, default=1.0, help="Keep clips with duration strictly greater than this value")
    parser.add_argument("--max-duration-sec", type=float, default=None, help="Optional upper bound: keep clips with duration less than or equal to this value")

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    if args.max_duration_sec is not None and args.max_duration_sec <= args.min_duration_sec:
        raise ValueError("--max-duration-sec must be greater than --min-duration-sec")

    dst_root.mkdir(parents=True, exist_ok=True)

    kept_by_mode = collect_kept_ids_and_copy(
        src_root,
        dst_root,
        args.min_duration_sec,
        args.max_duration_sec,
    )
    filter_label_csvs(src_root, dst_root, kept_by_mode)

    print("\n✅ Done. Filtered RoomReader dataset is ready.")


if __name__ == "__main__":
    main()
