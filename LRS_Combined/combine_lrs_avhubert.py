import os
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
import sentencepiece as spm


def combine_datasets_multi(dataset_dirs, output_path, vocab_size=2000):
    dataset_dirs = [str(Path(p).resolve()) for p in dataset_dirs if p]
    if len(dataset_dirs) < 2:
        raise ValueError("Provide at least two dataset directories")

    for p in dataset_dirs:
        if not os.path.exists(p):
            raise ValueError(f"Directory {p} does not exist")

    os.makedirs(output_path, exist_ok=True)

    required_files = [
        "train.tsv", "train.wrd",
        "test.tsv",  "test.wrd",
        "valid.tsv", "valid.wrd",
    ]
    for ds in dataset_dirs:
        for f in required_files:
            fp = os.path.join(ds, f)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Required file {f} not found in {ds}")

    print(f"Combining {len(dataset_dirs)} datasets into: {output_path}")

    # ── 1. Merge TSV + WRD (lowercase) ──────────────────────────────────────
    for split in ["train", "test", "valid"]:
        # TSV (keep only one header line)
        with open(os.path.join(output_path, f"{split}.tsv"), "w") as out:
            wrote_header = False
            for ds in dataset_dirs:
                with open(os.path.join(ds, f"{split}.tsv"), "r") as inp:
                    header = inp.readline()
                    if not wrote_header:
                        out.write(header)
                        wrote_header = True
                    out.write(inp.read())

        # WRD – lowercase
        with open(os.path.join(output_path, f"{split}.wrd"), "w", encoding="utf8") as out:
            for ds in dataset_dirs:
                with open(os.path.join(ds, f"{split}.wrd"), "r", encoding="utf8") as inp:
                    for line in inp:
                        out.write((line.rstrip("\n") or "").lower() + "\n")

    # ── 2. Train fresh SPM from combined train.wrd ───────────────────────────
    print(f"\nTraining new SPM (vocab_size={vocab_size}) from combined train.wrd ...")
    spm_dir = Path(output_path) / f"spm{vocab_size}" / "unigram"
    spm_dir.mkdir(parents=True, exist_ok=True)
    prefix = spm_dir / f"unigram{vocab_size}"

    train_wrd = os.path.join(output_path, "train.wrd")
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf8") as tmp:
        with open(train_wrd, "r", encoding="utf8") as f:
            for line in f:
                text = line.strip()
                if text:
                    tmp.write(text.lower() + "\n")
        tmp_path = tmp.name

    spm.SentencePieceTrainer.Train(
        input=tmp_path,
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols="<blank>",
    )
    os.unlink(tmp_path)
    print(f"  ✅ SPM model : {prefix}.model")
    print(f"  ✅ SPM vocab : {prefix}.vocab")

    # ── 3. Load trained SPM ──────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")

    # ── 4. Write dict.wrd.txt (fairseq frequency=1 format) ──────────────────
    output_dict_path = Path(output_path) / "dict.wrd.txt"
    skipped = {"<unk>", "<blank>", "<s>", "</s>", "<eos>", "<pad>"}
    with open(output_dict_path, "w", encoding="utf8") as fo:
        for i in range(sp.GetPieceSize()):
            token = sp.IdToPiece(i)
            if token not in skipped:
                fo.write(f"{token} 1\n")
    print(f"  ✅ dict.wrd.txt : {output_dict_path}  ({sp.GetPieceSize()} SPM tokens)")

    # ── 5. Write tokens.txt + auto_avsr CSV per split ────────────────────────
    def _write_tokens(split):
        wrd_in   = Path(output_path) / f"{split}.wrd"
        tsv_in   = Path(output_path) / f"{split}.tsv"
        tok_out  = Path(output_path) / f"{split}.tokens.txt"
        csv_out  = Path(output_path) / f"{split}_auto_avsr.csv"

        # Read video paths from TSV (skip root header line)
        with open(tsv_in, "r", encoding="utf8") as ftsv:
            ftsv.readline()
            tsv_lines = [ln.rstrip("\n") for ln in ftsv]

        video_paths = []
        for ln in tsv_lines:
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Malformed TSV line in {tsv_in}: {ln[:200]}")
            video_paths.append(parts[1])

        with open(wrd_in, "r", encoding="utf8") as fwrd:
            wrd_lines = [ln.rstrip("\n") for ln in fwrd]

        if len(video_paths) != len(wrd_lines):
            raise ValueError(
                f"Line count mismatch for {split}: "
                f"tsv={len(video_paths)}, wrd={len(wrd_lines)}"
            )

        with open(tok_out, "w", encoding="utf8") as ftok, \
             open(csv_out, "w", encoding="utf8") as fc:
            for vid, text in zip(video_paths, wrd_lines):
                ids = sp.EncodeAsIds((text or "").lower())
                id_str = " ".join(str(i) for i in ids)
                ftok.write(id_str + "\n")
                fc.write(f"lrs_combined,{os.path.abspath(vid)},{id_str}\n")

        print(f"  ✅ {split:5s} : {tok_out.name}  |  {csv_out.name}")

    print()
    for split in ["train", "valid", "test"]:
        _write_tokens(split)

    # ── 6. Sanity-check line counts ──────────────────────────────────────────
    print()
    all_ok = True
    for split in ["train", "test", "valid"]:
        tsv_n = sum(1 for _ in open(os.path.join(output_path, f"{split}.tsv"))) - 1
        wrd_n = sum(1 for _ in open(os.path.join(output_path, f"{split}.wrd")))
        tok_n = sum(1 for _ in open(os.path.join(output_path, f"{split}.tokens.txt")))
        ok = tsv_n == wrd_n == tok_n
        mark = "✅" if ok else "⚠️ "
        print(f"  {mark} {split:5s}: tsv={tsv_n}  wrd={wrd_n}  tokens={tok_n}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All line counts match.")
    else:
        print("WARNING: line count mismatches detected above — fix before training.")

    print(f"\nDone. Combined dataset → {output_path}")
    print(f"Use this SPM in your training config:")
    print(f"  tokenizer_bpe_model: {prefix}.model")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple dataset metadata folders into one "
                    "(trains fresh lowercase SPM + writes dict/tokens/CSV)."
    )
    parser.add_argument("--d1", required=True,  help="Dataset 1 metadata dir")
    parser.add_argument("--d2", required=True,  help="Dataset 2 metadata dir")
    parser.add_argument("--d3", default=None,   help="Dataset 3 metadata dir (optional)")
    parser.add_argument("--d4", default=None,   help="Dataset 4 metadata dir (optional)")
    parser.add_argument("--d5", default=None,   help="Dataset 5 metadata dir (optional)")
    parser.add_argument("--d6", default=None,   help="Dataset 6 metadata dir (optional)")
    parser.add_argument("--output",     required=True,      help="Output path for combined dataset")
    parser.add_argument("--vocab_size", type=int, default=2000,
                        help="SPM vocabulary size (default: 2000)")
    args = parser.parse_args()

    dirs = [args.d1, args.d2, args.d3, args.d4, args.d5, args.d6]
    combine_datasets_multi([d for d in dirs if d], args.output, args.vocab_size)

# Usage example:
# python combine_datasets.py \
#   --d1 /path/to/lrs2_metadata \
#   --d2 /path/to/lrs3_metadata \
#   --d3 /path/to/candor_metadata \
#   --output /path/to/combined_output \
#   --vocab_size 2000