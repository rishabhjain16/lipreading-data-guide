#!/usr/bin/env python3
"""RoomReader Step 3: Create RR_easy and RR_hard subsets from Step 2 metadata.

This script takes a *single* Step-2 metadata folder (e.g. `meta/combined/`) that
contains `test.tsv` and `test.wrd`, and splits utterances into:

- RR_hard: duration <= threshold seconds (default 2.0)
- RR_easy: duration >  threshold seconds

For each subset it writes:
- AV-HuBERT-style manifests: test.tsv + test.wrd
- Token helpers: test.tokens.txt + label.csv
- Auto-AVSR 4-col CSV: <subset>_test_transcript_lengths_seg16s.csv
- Stats: stats.json + stats.txt (includes total duration in hours and word count stats)

Notes
- Duration is estimated from `nframes_video / --fps` using the frame count in TSV.
- Tokenization uses repo `TextTransform()` (shared SPM setup in this repo).

Example:
  python RoomReader/preparation/step3_filter_roomreader_by_duration.py \
    --metadata-dir /data/.../meta/combined \
    --output-dir   /data/.../meta_split
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import wave
from pathlib import Path
from typing import Dict, List, Tuple

from transforms import TextTransform


def read_test_tsv_wrd(metadata_dir: Path) -> List[Dict]:
    tsv_path = metadata_dir / "test.tsv"
    wrd_path = metadata_dir / "test.wrd"
    if not tsv_path.exists() or not wrd_path.exists():
        raise FileNotFoundError(f"Expected test.tsv and test.wrd in: {metadata_dir}")

    with open(tsv_path, "r", encoding="utf8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        return []

    # AV-HuBERT style header is just '/'
    if lines[0].strip() == "/":
        lines = lines[1:]

    with open(wrd_path, "r", encoding="utf8") as f:
        wrds = [ln.rstrip("\n") for ln in f]

    if len(lines) != len(wrds):
        raise ValueError(
            f"Line mismatch: {tsv_path} has {len(lines)} examples, {wrd_path} has {len(wrds)} lines"
        )

    records: List[Dict] = []
    for ln, text in zip(lines, wrds):
        parts = ln.split("\t")
        if len(parts) < 4:
            continue

        unique_id = parts[0]
        video_path = parts[1]
        audio_path = parts[2]
        try:
            nframes_video = int(parts[3])
        except Exception:
            nframes_video = -1

        nframes_audio = None
        if len(parts) >= 5:
            try:
                nframes_audio = int(parts[4])
            except Exception:
                nframes_audio = None

        records.append(
            {
                "unique_id": unique_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "nframes_video": nframes_video,
                "nframes_audio": nframes_audio,
                "transcript": text.strip(),
            }
        )

    return records


def compute_duration_sec(record: Dict, fps: float) -> Tuple[float, str]:
    """Compute duration by reading the audio file on disk.

    This is intentionally simple and follows your requirement:
    - Use audio file duration as ground truth.
    - Do not rely on manifest frame counts.
    - No caching.

    The audio path comes from the TSV's 3rd column.

    Returns (duration_seconds, source) where source is:
    - "audio_wav": duration read from wav header
    """

    audio_path = record.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError(f"Record {record.get('unique_id')} missing audio_path")

    try:
        with wave.open(audio_path, "rb") as wf:
            nframes = wf.getnframes()
            fr = wf.getframerate()
            if fr <= 0:
                raise ValueError(f"Invalid wav framerate={fr} for {audio_path}")
            return (nframes / float(fr), "audio_wav")
    except Exception as e:
        uid = record.get("unique_id")
        raise RuntimeError(f"Failed to read WAV duration for id={uid} path={audio_path}: {e}")


def split_easy_hard(records: List[Dict], threshold_sec: float, fps: float) -> Tuple[List[Dict], List[Dict]]:
    easy: List[Dict] = []
    hard: List[Dict] = []

    for r in records:
        dur, dur_src = compute_duration_sec(r, fps=fps)
        r2 = dict(r)
        r2["duration_sec"] = float(dur)
        r2["duration_source"] = dur_src
        if dur <= threshold_sec:
            hard.append(r2)
        else:
            easy.append(r2)

    return easy, hard


def word_count_stats(records: List[Dict]) -> Dict:
    counts = [len((r.get("transcript") or "").split()) for r in records]
    if not counts:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0.0}

    return {
        "min": int(min(counts)),
        "max": int(max(counts)),
        "avg": float(sum(counts) / len(counts)),
        "median": float(statistics.median(counts)),
    }


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_words(text: str) -> List[str]:
    # Simple, robust "word" tokenization that ignores punctuation.
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def word_level_analysis(records: List[Dict], top_k: int = 30) -> Dict:
    """Compute word-level stats from transcripts (derived from .wrd lines)."""

    word_freq: Dict[str, int] = {}
    uniq_words_per_utt: List[int] = []
    total_words = 0

    for r in records:
        words = _tokenize_words(r.get("transcript") or "")
        total_words += len(words)
        uniq_words_per_utt.append(len(set(words)))
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

    vocab_size = len(word_freq)
    top_words = sorted(word_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    hapax_count = sum(1 for c in word_freq.values() if c == 1)

    long_words = [(w, c) for (w, c) in word_freq.items() if len(w) >= 7]
    top_long_words = sorted(long_words, key=lambda kv: (-kv[1], kv[0]))[:top_k]

    uniq_stats = {
        "min": int(min(uniq_words_per_utt)) if uniq_words_per_utt else 0,
        "max": int(max(uniq_words_per_utt)) if uniq_words_per_utt else 0,
        "avg": float(sum(uniq_words_per_utt) / len(uniq_words_per_utt)) if uniq_words_per_utt else 0.0,
        "median": float(statistics.median(uniq_words_per_utt)) if uniq_words_per_utt else 0.0,
    }

    return {
        "total_words": int(total_words),
        "vocab_size": int(vocab_size),
        "hapax_count": int(hapax_count),
        "hapax_ratio": float(hapax_count / vocab_size) if vocab_size else 0.0,
        "unique_words_per_utt": uniq_stats,
        "top_words": [{"word": w, "count": c} for (w, c) in top_words],
        "top_long_words": [{"word": w, "count": c} for (w, c) in top_long_words],
    }


def duration_hours(records: List[Dict]) -> float:
    total_sec = sum(float(r.get("duration_sec", 0.0)) for r in records)
    return total_sec / 3600.0


def duration_audit(records: List[Dict], fps: float) -> Dict:
    """Compare wav-read duration with manifest-derived duration (if possible)."""

    wav_secs: List[float] = []
    manifest_secs: List[float] = []
    ratios: List[float] = []

    for r in records:
        dur, _src = compute_duration_sec(r, fps=fps)
        wav_secs.append(dur)

        nfv = r.get("nframes_video")
        if isinstance(nfv, int) and nfv >= 0 and fps > 0:
            ms = nfv / fps
            manifest_secs.append(ms)
        else:
            ms = None

        if ms is not None and dur > 0:
            ratios.append(ms / dur)

    def _summ(x: List[float]) -> Dict:
        if not x:
            return {"count": 0}
        return {
            "count": len(x),
            "min": float(min(x)),
            "max": float(max(x)),
            "avg": float(sum(x) / len(x)),
            "median": float(statistics.median(x)),
        }

    return {
        "wav_duration_sec": _summ(wav_secs),
        "manifest_duration_sec": _summ(manifest_secs),
        "manifest_over_wav_ratio": _summ(ratios),
    }


def write_subset(
    out_dir: Path,
    records: List[Dict],
    subset_name: str,
    fps: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = out_dir / "test.tsv"
    wrd_path = out_dir / "test.wrd"
    tokens_path = out_dir / "test.tokens.txt"
    label_csv_path = out_dir / "label.csv"
    avsr_csv_path = out_dir / f"{subset_name}_test_transcript_lengths_seg16s.csv"

    tt = TextTransform()

    # Write manifests
    with open(tsv_path, "w", encoding="utf8") as ftsv, open(wrd_path, "w", encoding="utf8") as fwrd:
        ftsv.write("/\n")
        for r in records:
            nfv = int(r.get("nframes_video", -1))
            nfa = r.get("nframes_audio")
            if nfa is None or (isinstance(nfa, int) and nfa < 0):
                dur = float(r.get("duration_sec", (nfv / fps) if (nfv >= 0 and fps > 0) else 0.0))
                nfa = int(dur * 16000)

            audio_path = r["audio_path"]

            ftsv.write(
                "\t".join([
                    str(r["unique_id"]),
                    str(r["video_path"]),
                    str(audio_path),
                    str(nfv),
                    str(nfa),
                ]) + "\n"
            )
            fwrd.write((r.get("transcript") or "") + "\n")

    # Write tokens + CSVs
    with open(tokens_path, "w", encoding="utf8") as ftok, open(label_csv_path, "w", encoding="utf8") as flab, open(avsr_csv_path, "w", encoding="utf8") as fav:
        for r in records:
            text = r.get("transcript") or ""
            token_ids = tt.tokenize(text)
            token_str = " ".join(str(t.item()) for t in token_ids)
            ftok.write(token_str + "\n")
            flab.write(f"{subset_name},{r['video_path']},{token_str}\n")
            fav.write(f"{subset_name},{r['video_path']},{r.get('nframes_video', -1)},{token_str}\n")

    src_counts: Dict[str, int] = {}
    for r in records:
        src = r.get("duration_source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    stats = {
        "subset": subset_name,
        "utterances": len(records),
        "duration_hours": duration_hours(records),
        "word_count": word_count_stats(records),
        "word_analysis": word_level_analysis(records),
        "duration_audit": duration_audit(records, fps=fps),
        "duration_source_counts": src_counts,
        "fps": fps,
    }

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf8")
    (out_dir / "stats.txt").write_text(
        "\n".join([
            f"subset: {subset_name}",
            f"utterances: {stats['utterances']}",
            f"duration_hours: {stats['duration_hours']:.4f}",
            f"word_count_min: {stats['word_count']['min']}",
            f"word_count_max: {stats['word_count']['max']}",
            f"word_count_avg: {stats['word_count']['avg']:.4f}",
            f"word_count_median: {stats['word_count']['median']:.4f}",
            f"duration_sources: {json.dumps(stats['duration_source_counts'], sort_keys=True)}",
            f"total_words: {stats['word_analysis']['total_words']}",
            f"vocab_size: {stats['word_analysis']['vocab_size']}",
            f"hapax_count: {stats['word_analysis']['hapax_count']}",
            f"hapax_ratio: {stats['word_analysis']['hapax_ratio']:.4f}",
            "top_words:",
            *[f"  {d['word']}\t{d['count']}" for d in stats["word_analysis"]["top_words"]],
        ]) + "\n",
        encoding="utf8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split RoomReader Step2 metadata into RR_easy/RR_hard and write AV-HuBERT + Auto-AVSR artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-dir", required=True, help="Step2 metadata folder (contains test.tsv + test.wrd)")
    parser.add_argument("--output-dir", required=True, help="Output folder for RR_easy/ and RR_hard/")
    parser.add_argument("--duration-threshold", type=float, default=2.0, help="<= threshold => hard; > threshold => easy")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS used to convert nframes_video -> seconds")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_test_tsv_wrd(metadata_dir)
    if not records:
        print(f"No records found in {metadata_dir}. Nothing to do.")
        return 0

    easy, hard = split_easy_hard(records, args.duration_threshold, args.fps)

    print(f"Loaded {len(records)} utterances from {metadata_dir}")
    print(f"Threshold: {args.duration_threshold:.3f}s (fps={args.fps})")
    print(f"RR_easy: {len(easy)}")
    print(f"RR_hard: {len(hard)}")

    write_subset(
        output_dir / "RR_easy",
        easy,
        subset_name="RR_easy",
        fps=args.fps,
    )
    write_subset(
        output_dir / "RR_hard",
        hard,
        subset_name="RR_hard",
        fps=args.fps,
    )

    easy_stats = json.loads((output_dir / "RR_easy" / "stats.json").read_text(encoding="utf8"))
    hard_stats = json.loads((output_dir / "RR_hard" / "stats.json").read_text(encoding="utf8"))
    print("\n=== Summary ===")
    print(f"RR_easy hours: {easy_stats['duration_hours']:.3f} | word_avg: {easy_stats['word_count']['avg']:.2f}")
    print(f"RR_hard hours: {hard_stats['duration_hours']:.3f} | word_avg: {hard_stats['word_count']['avg']:.2f}")
    print(f"Wrote to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

