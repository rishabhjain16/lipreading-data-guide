#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

import sentencepiece as spm

def merge_dictionaries(lrs2_dict_path, lrs3_dict_path, output_dict_path):
    """
    Merge two dictionary files while preserving the format and removing duplicates.
    """
    print("Merging dictionaries...")
    
    # Read dictionaries
    lrs2_dict = {}
    lrs3_dict = {}
    
    with open(lrs3_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                index_or_freq = parts[1]
                lrs3_dict[word] = index_or_freq
    
    with open(lrs2_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                index_or_freq = parts[1]
                lrs2_dict[word] = index_or_freq
    
    # Determine if dictionaries use indices or frequencies
    uses_indices = all(value.isdigit() for value in list(lrs3_dict.values()))
    
    # Merge dictionaries
    merged_dict = {**lrs2_dict, **lrs3_dict}  # LRS3 takes precedence for duplicates
    
    # If dictionaries use indices, reassign indices to be continuous
    if uses_indices:
        print("Dictionaries use indices, reassigning to be continuous...")
        merged_items = sorted(merged_dict.items(), key=lambda x: int(x[1]))
        merged_dict = {word: str(i) for i, (word, _) in enumerate(merged_items)}
    
    # Write merged dictionary
    with open(output_dict_path, 'w') as f:
        for word, value in sorted(merged_dict.items(), key=lambda x: int(x[1]) if uses_indices else x[0]):
            f.write(f"{word} {value}\n")
    
    print(f"Merged dictionary created with {len(merged_dict)} entries")
    return merged_dict

def combine_datasets(lrs2_path, lrs3_path, output_path):
    """
    Combine LRS2 and LRS3 datasets into a single dataset for training.
    Assumes both datasets already have .tsv, .wrd, and .cluster_counts files.
    """
    raise NotImplementedError("Legacy entrypoint. Use combine_datasets_multi(...)")


def combine_datasets_multi(dataset_dirs, output_path):
    """Combine multiple prepared metadata folders into one.

    Inputs:
      dataset_dirs: list of metadata directories, each containing:
        train/valid/test.{tsv,wrd,cluster_counts} and dict.wrd.txt
      output_path: output folder for combined manifests and shared-SPM outputs
    """
    dataset_dirs = [str(Path(p).resolve()) for p in dataset_dirs if p]
    if len(dataset_dirs) < 2:
        raise ValueError("Provide at least two dataset directories (d1..d6)")

    for p in dataset_dirs:
        if not os.path.exists(p):
            raise ValueError(f"Directory {p} does not exist")

    os.makedirs(output_path, exist_ok=True)

    required_files = [
        "train.tsv", "train.wrd", "train.cluster_counts",
        "test.tsv", "test.wrd", "test.cluster_counts",
        "valid.tsv", "valid.wrd", "valid.cluster_counts",
        "dict.wrd.txt",
    ]

    for ds in dataset_dirs:
        for file in required_files:
            fp = os.path.join(ds, file)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Required file {file} not found in {ds}")

    print(f"Combining {len(dataset_dirs)} datasets...")
    
    # Merge files for each split across all datasets (keep only one TSV header)
    for split in ["train", "test", "valid"]:
        # TSV
        with open(os.path.join(output_path, f"{split}.tsv"), 'w') as outfile:
            wrote_header = False
            for ds in dataset_dirs:
                with open(os.path.join(ds, f"{split}.tsv"), 'r') as infile:
                    header = infile.readline()
                    if not wrote_header:
                        outfile.write(header)
                        wrote_header = True
                    outfile.write(infile.read())

        # WRD
        with open(os.path.join(output_path, f"{split}.wrd"), 'w') as outfile:
            for ds in dataset_dirs:
                with open(os.path.join(ds, f"{split}.wrd"), 'r') as infile:
                    outfile.write(infile.read())

        # cluster_counts
        with open(os.path.join(output_path, f"{split}.cluster_counts"), 'w') as outfile:
            for ds in dataset_dirs:
                with open(os.path.join(ds, f"{split}.cluster_counts"), 'r') as infile:
                    outfile.write(infile.read())
    
    # --- Shared SPM outputs (dict.wrd.txt, tokens.txt, per-split CSV) ---
    # We intentionally do NOT merge per-dataset dicts here, because downstream token ids
    # should be consistent across datasets via the repo-wide shared SPM model.
    repo_root = Path(__file__).resolve().parents[1]
    sp_model_path = repo_root / "spm" / "unigram" / "unigram5000.model"
    units_path = repo_root / "spm" / "unigram" / "unigram5000_units.txt"

    if not sp_model_path.exists():
        raise FileNotFoundError(f"Shared SPM model not found: {sp_model_path}")
    if not units_path.exists():
        raise FileNotFoundError(f"Shared SPM units not found: {units_path}")

    print(f"Using shared SPM model: {sp_model_path}")
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))

    # Write dict.wrd.txt from shared SPM units
    output_dict_path = Path(output_path) / "dict.wrd.txt"
    with open(units_path, "r", encoding="utf8") as fi, open(output_dict_path, "w", encoding="utf8") as fo:
        # Match other scripts: skip <unk>/<blank>/<eos> special tokens
        for line in fi:
            tok = line.strip().split()
            if not tok:
                continue
            token = tok[0]
            if token in {"<unk>", "<blank>", "<eos>"}:
                continue
            fo.write(line)
    print(f"Wrote shared dictionary: {output_dict_path}")

    def _write_tokens_and_label_csv(split: str):
        """Create tokens.txt and label.csv (no header) for a split.

        label.csv format:
            lrs_combined,<abs_video_path>,<space-separated-token-ids>
        """
        wrd_in = Path(output_path) / f"{split}.wrd"
        tsv_in = Path(output_path) / f"{split}.tsv"
        tokens_out = Path(output_path) / f"{split}.tokens.txt"
        csv_out = Path(output_path) / f"{split}_auto_avsr.csv"

        if not wrd_in.exists() or not tsv_in.exists():
            raise FileNotFoundError(f"Missing combined split files for {split}: {wrd_in} / {tsv_in}")

        # Read video paths from tsv (skip header root line)
        with open(tsv_in, "r", encoding="utf8") as ftsv:
            root = ftsv.readline()  # '/\n'
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
                f"Line count mismatch for {split}: {tsv_in} has {len(video_paths)} examples, "
                f"but {wrd_in} has {len(wrd_lines)} lines"
            )

        dataset_name = "lrs_combined"
        with open(tokens_out, "w", encoding="utf8") as ftok, open(csv_out, "w", encoding="utf8") as fc:
            for vid, text in zip(video_paths, wrd_lines):
                # Shared vocab is uppercase
                ids = sp.EncodeAsIds((text or "").upper())
                token_str = " ".join(str(i) for i in ids)
                ftok.write(token_str + "\n")
                fc.write(f"{dataset_name},{os.path.abspath(vid)},{token_str}\n")

        print(f"Wrote: {tokens_out}")
        print(f"Wrote: {csv_out}")

    for split in ["train", "valid", "test"]:
        _write_tokens_and_label_csv(split)
    
    # Verify line counts match between corresponding files
    for split in ["train", "test", "valid"]:
        tsv_count = len(open(os.path.join(output_path, f"{split}.tsv")).readlines()) - 1  # Subtract header line
        wrd_count = len(open(os.path.join(output_path, f"{split}.wrd")).readlines())
        cluster_count = len(open(os.path.join(output_path, f"{split}.cluster_counts")).readlines())
        
        if not (tsv_count == wrd_count == cluster_count):
            print(f"Warning: Line count mismatch in {split} files:")
            print(f"  tsv: {tsv_count}, wrd: {wrd_count}, cluster_counts: {cluster_count}")
            print("  This may cause issues during training.")
        else:
            print(f"{split} set: {tsv_count} examples merged successfully")
    
    print(f"Combined dataset successfully created at {output_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple dataset metadata folders into one (shared SPM + Auto-AVSR CSVs)")
    parser.add_argument("--d1", required=True, help="Dataset 1 metadata dir (train/valid/test TSV+WRD+cluster_counts)")
    parser.add_argument("--d2", required=True, help="Dataset 2 metadata dir")
    parser.add_argument("--d3", default=None, help="Dataset 3 metadata dir")
    parser.add_argument("--d4", default=None, help="Dataset 4 metadata dir")
    parser.add_argument("--d5", default=None, help="Dataset 5 metadata dir")
    parser.add_argument("--d6", default=None, help="Dataset 6 metadata dir")
    parser.add_argument("--output", required=True, help="Path for the combined dataset")
    
    args = parser.parse_args()
    
    dirs = [args.d1, args.d2, args.d3, args.d4, args.d5, args.d6]
    combine_datasets_multi([d for d in dirs if d], args.output)

#Usage: python combine_datasets.py   --lrs2 /home/rishabh/Desktop/Datasets/lrs2_rf/lrs2/lrs2_video_seg16s/data_lrs2/   --lrs3 /home/rishabh/Desktop/Datasets/lrs3/433h_data   --output /home/rishabh/Desktop/Datasets/lrs_combined