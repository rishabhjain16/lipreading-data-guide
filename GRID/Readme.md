# GRID Corpus Dataset Preprocessing

Complete preprocessing pipeline for GRID Corpus - a large multitalker audiovisual sentence corpus with 33 speakers and ~33,000 utterances.

## Dataset Overview

- **Speakers**: 33 (s1-s34, s21 missing)
- **Videos per speaker**: 1000 utterances
- **Total videos**: ~33,000
- **Sentence structure**: 6 words (command + color + preposition + letter + digit + adverb)
- **Vocabulary**: 51 words (fixed grammar)
- **Video format**: 360x288, 25 fps, MPG files
- **Duration**: ~3 seconds per video

**Note**: Some files may be skipped during preprocessing due to corrupted video files or empty videos. This is expected behavior and the pipeline will continue processing remaining files.

## Quick Start

```bash
# Step 1: Process videos with RetinaFace
python preparation/step1_prepare_grid.py \
    --data-dir /media/rishabhjain/SSD/GRID \
    --root-dir /path/to/output \
    --crop-type lips

# Step 2: Generate file lists (optional)
python preparation/step2_generate_file_lists.py \
    --grid-data-dir /path/to/output/grid_video

# Step 3: Create metadata (optional)
python preparation/step3_metadata_prep.py \
    --grid-data-dir /path/to/output/grid_video \
    --metadata-dir /path/to/output/metadata \
    --split-ratios 0.7,0.15,0.15

# Step 3 (inference-only): skip splitting and write a single manifest
python preparation/step3_metadata_prep.py \
    --grid-data-dir /path/to/output/grid_video \
    --metadata-dir /path/to/output/metadata \
    --no-split
```

Notes:

- Step 2 only writes `file*.list` and `label*.list` (it does **not** do any splitting).
- `--split-ratios` is only used in Step 3 because that’s where `train.tsv` / `valid.tsv` / `test.tsv` are created.

### Tokenization (shared SPM)

`step3_metadata_prep.py` tokenizes labels using the repo-wide SentencePiece model:

- Model: `spm/unigram/unigram5000.model`
- Units: `spm/unigram/unigram5000_units.txt`

Important detail: this SPM model’s vocabulary is **uppercase**, so GRID labels are normalized to **uppercase before encoding**.
If you see outputs like repeated `501 1 501 1 ...` in `tokens.txt`, it usually means the input text case didn’t match the SPM model.

## Dataset Structure

```
/media/rishabhjain/SSD/GRID/
├── s1/
│   └── s1/
│       ├── bbaf2n.mpg
│       ├── bbaf3s.mpg
│       └── ... (1000 videos)
├── s2/
│   └── s2/
│       └── ... (1000 videos)
├── s3/, s4/, ..., s34/
├── alignments/
│   └── alignments/
│       ├── s1/
│       │   ├── bbaf2n.align
│       │   ├── bbaf3s.align
│       │   └── ...
│       ├── s2/, s3/, ..., s34/
└── audio_25k/
    └── (optional 25kHz audio files)
```

## Output Structure

```
output/
├── grid_video/                              # Videos + Audio (lips, 96x96)
│   ├── s1/
│   │   ├── bbaf2n.mp4
│   │   ├── bbaf2n.wav
│   │   └── ...
│   ├── s2/, s3/, ..., s34/
│   ├── file.list
│   ├── label.list
│   └── nframes.audio, nframes.video
├── grid_video_face_224x224/                 # Videos + Audio (face, 224x224)
├── grid_text/                               # Text Files (lips)
│   ├── s1/
│   │   ├── bbaf2n.txt
│   │   └── ...
│   ├── s2/, s3/, ..., s34/
├── grid_text_face_224x224/                  # Text Files (face)
├── labels/                                  # CSV Metadata
│   ├── grid_all.csv                         # All speakers (~33,000 videos)
│   ├── grid_s1.csv                          # Speaker s1 (1000 videos)
│   ├── grid_s2.csv                          # Speaker s2 (1000 videos)
│   └── ... (33 speaker CSVs)
└── metadata/                                # Training Manifests (optional)
    ├── train.tsv, valid.tsv, test.tsv
    ├── train.wrd, valid.wrd, test.wrd
    ├── dict.wrd.txt
    ├── label.csv                             # Simple CSV (no header): dataset,abs_video_path,token_ids
    ├── tokens.txt                            # One token-id sequence per utterance (SentencePiece ids)
    └── grid_test_transcript_lengths_seg16s.csv # Auto-AVSR CSV (no header): dataset,abs_video_path,input_length,token_ids
    └── spm100/
```

## Options

### Step 1
- `--crop-type`: `lips` (96x96) or `face` (224x224)
- `--groups`, `--job-index`: For parallel processing

### Step 2
- `--speaker`: Process specific speaker (s1, s2, ...) or omit for all

### Step 3
- `--speaker`: Process specific speaker or omit for all
- `--split-ratios`: Train/val/test ratios (default: 0.7,0.15,0.15)
- `--no-split`: Write a single `all.tsv`/`all.wrd` instead of `train/valid/test` (useful for inference-only)

Step 3 no longer generates a custom vocabulary for GRID; it uses the shared root SPM model.

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install ibug-face_detection ibug-face_alignment
pip install opencv-python pandas tqdm
```

## Citation

```bibtex
@inproceedings{cooke2006audio,
  title={An audio-visual corpus for speech perception and automatic speech recognition},
  author={Cooke, Martin and Barker, Jon and Cunningham, Stuart and Shao, Xu},
  booktitle={2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings},
  volume={5},
  pages={V--V},
  year={2006},
  organization={IEEE}
}
```
