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

# Step 2: Create splits and manifests
python preparation/step2_split_unseen.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --splits-dir ./splits \
    --use-existing-splits

# Step 3: Train SPM tokenizer (if needed)
python preparation/step3_train_candor_spm.py \
    --metadata-dir ./candor_output/metadata \
    --vocab-size 5000
```

- Step 2 generates the standard AV-HuBERT manifests (`train/valid/test.tsv` + `.wrd`) using official or custom splits.
- Step 3 trains a SentencePiece model from `train.wrd` and writes:
  - `spm5000/unigram/unigram5000.model` (SPM model)
  - `spm5000/unigram/unigram5000.vocab` (SPM vocab)
  - `spm5000/unigram/unigram5000_units.txt` (for TextTransform)
  - `dict.wrd.txt` (Fairseq-style dictionary)

**Note:** The legacy `step2_generate_file_lists.py` is no longer required for the standard workflow. Use it only for legacy/compatibility purposes.

### Output Structure

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
    ├── train.wrd, valid.wrd, test.wrd
    ├── spm5000/
    │   └── unigram/
    │       ├── unigram5000.model
    │       ├── unigram5000.vocab
    │       └── unigram5000_units.txt
    └── dict.wrd.txt

**Note**: SPM files are created in Step 3. Tokenization for downstream tasks should use the model and units file from `spm5000/unigram/`.

---

## Key Parameters

**Step 1 (Video Processing)**:
- `--crop-type`: `lips` (96x96) or `face` (224x224)
- `--min-duration-ms`: Minimum clip duration (default: 800ms)
- `--min-word-count`: Minimum words per phrase (default: 2)
- `--filter-fillers`: Filter filler-only phrases (uhm, uh, etc.)
- `--save-combined-av`: Save combined AV files for sanity checking

**Step 2 (Splits & Manifests)**:
- `--use-existing-splits`: Use fixed train/val/test splits (recommended)
- `--splits-dir`: Directory containing `candor-train.id`, `candor-valid.id`, `candor-test.id`

**Step 3 (SPM Training)**:
- `--vocab-size`: Vocabulary size for SentencePiece model (e.g., 5000)

---

## Official Splits (Reproducibility)

The `splits/` folder contains fixed train/val/test splits so everyone uses the same data splits.

**Using official splits** (recommended):
```bash
python preparation/step2_split_unseen.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --splits-dir ./splits \
    --use-existing-splits
```

**Creating new splits** (only if processing full dataset for first time):
```bash
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

python preparation/step2_split_unseen.py \
    --candor-data-dir ./candor_output/candor_video \
    --metadata-dir ./candor_output/metadata \
    --splits-dir ./splits \
    --use-existing-splits

python preparation/step3_train_candor_spm.py \
    --metadata-dir ./candor_output/metadata \
    --vocab-size 5000
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
