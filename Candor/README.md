# Candor Dataset Preprocessing

Complete preprocessing pipeline for Candor conversational dataset into AV-HuBERT/Auto-AVSR format.

## Quick Start

```bash
# Step 1: Process videos
python preparation/step1_prepare_candor.py \
    --candor-path /media/rishabhjain/HDD/Candor_test \
    --speechmatics-path /media/rishabhjain/HDD/candor_speechmatics \
    --output-path ./candor_output \
    --crop-type lips

# Step 2: Generate training manifests
python preparation/step2_generate_file_lists.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --use-official-splits

This step generates the standard AV-HuBERT manifests (`train/valid/test.tsv` + `.wrd`).
It also generates **shared-SPM tokenization outputs** next to the manifests:

- `train.tokens.txt`, `valid.tokens.txt`, `test.tokens.txt`
- `candor_train_transcript_lengths_seg16s*.csv`, `candor_val_transcript_lengths_seg16s*.csv`, `candor_test_transcript_lengths_seg16s*.csv` (no header)
    - **4 columns** (Auto-AVSR style, matches LRS2): `dataset,rel_video_path,input_length,token_ids`
    - `input_length` is the **number of video frames**
    - `token_ids` are generated using the repo-wide shared SentencePiece model:
        - `spm/unigram/unigram5000.model`
        - input text is uppercased before encoding (shared vocab is uppercase)

Optional (legacy) Auto-AVSR CSVs with duration + TextTransform (4 columns) can be generated with:

```bash
python preparation/step2_generate_file_lists.py \
    --candor-data-dir /path/to/candor_processed \
    --metadata-dir /path/to/metadata \
    --write-legacy-avsr-csv
```

# Done! Training data in ./candor_output/metadata/
```

---

## Output Structure

```
candor_output/
├── candor_video/              # Videos + Audio
│   └── {session_id}/
│       ├── {session_id}_spk0_0000.mp4
│       ├── {session_id}_spk0_0000.wav
│       └── ...
├── candor_text/               # Transcripts
├── labels/                    # CSV files
│   ├── candor.csv            # All data (from Step 1)
│   ├── candor_train.csv      # Train split (Auto-AVSR format with SPM tokens)
│   ├── candor_valid.csv      # Valid split (Auto-AVSR format with SPM tokens)
│   └── candor_test.csv       # Test split (Auto-AVSR format with SPM tokens)
└── metadata/                  # AV-HuBERT format (from Step 2)
    ├── train.tsv, valid.tsv, test.tsv
    └── train.wrd, valid.wrd, test.wrd
```

**Note**: CSV files use SentencePiece tokenization (spm1000 by default) for Auto-AVSR compatibility.

---

## Key Parameters

**Step 1 (Video Processing)**:
- `--crop-type`: `lips` (96x96) or `face` (224x224)
- `--min-duration-ms`: Minimum clip duration (default: 800ms)
- `--min-word-count`: Minimum words per phrase (default: 2)
- `--filter-fillers`: Filter filler-only phrases (uhm, uh, etc.)
- `--save-combined-av`: Save combined AV files for sanity checking

**Step 2 (Manifest Generation)**:
- `--use-official-splits`: Use fixed train/val/test splits (recommended)
- `--split-by`: `session` or `speaker` (if not using official splits)
- `--spm-model`: Path to SentencePiece model (default: uses spm1000)

---

## Official Splits (Reproducibility)

The `splits/` folder contains fixed train/val/test splits so everyone uses the same data splits.

**Using official splits** (recommended):
```bash
python preparation/step2_generate_file_lists.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --use-official-splits
```

**Creating new splits** (only if processing full dataset for first time):
```bash
# After Step 1, before Step 2
python preparation/create_official_splits.py \
    --candor-data-dir ./candor_output/candor_video \
    --output-dir ./splits \
    --seed 42
```

See `splits/README.md` for details.

---

## Common Use Cases

**Standard processing**:
```bash
python preparation/step1_prepare_candor.py \
    --candor-path /path/to/Candor \
    --speechmatics-path /path/to/candor_speechmatics \
    --output-path ./candor_output \
    --crop-type lips
```

**With sanity check**:
```bash
# Add --save-combined-av to create combined AV files
python preparation/step1_prepare_candor.py \
    ... \
    --save-combined-av

# Check lip sync
vlc ./candor_output/candor_video_av/{session_id}/*_av.mp4
```

**Lenient filtering** (keep more data):
```bash
python preparation/step1_prepare_candor.py \
    ... \
    --min-duration-ms 500 \
    --min-word-count 1
```

---

## Citation

```bibtex
@article{reece2023candor,
  title={The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation},
  author={Reece, Andrew and Cooney, Gus and Bull, Peter and Chung, Christine},
  journal={Science Advances},
  volume={9},
  pages={eadf3197},
  year={2023}
}
```

---

## Additional Documentation

- `docs/PREPROCESSING_METHODOLOGY.md` - Detailed methodology for research papers
- `docs/DATASET_STRUCTURE.md` - Dataset structure analysis
- `splits/README.md` - Official splits documentation
- `WORKFLOW.md` - Visual workflow diagram
