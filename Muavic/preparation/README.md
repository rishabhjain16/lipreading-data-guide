# MuAViC Preprocessing Scripts

This directory contains preprocessing scripts for the MuAViC (Multilingual Audio-Visual Corpus) dataset.

## Two-Stage Preprocessing

MuAViC preprocessing is divided into two stages:

1. **Download Stage**: Use official MuAViC scripts to download raw data
2. **RetinaFace Stage**: Apply RetinaFace preprocessing to raw videos

This approach combines MuAViC's robust download/segmentation with Auto-AVSR's superior face detection.

## Scripts

### `step0_download_muavic.py`
Downloads raw MuAViC data using official scripts (WITHOUT face cropping).

**Features:**
- Downloads videos from YouTube (mTEDx or LRS3)
- Segments videos based on timestamps
- Extracts audio files
- Creates metadata TSV files
- Preserves raw full-frame videos for RetinaFace processing

**Usage:**
```bash
python step0_download_muavic.py \
    --root-path /path/to/muavic_data \
    --src-lang en
```

### `step1_prepare_muavic_retinaface.py`
Main preprocessing script that uses RetinaFace for face detection and landmark localization.

**Features:**
- RetinaFace-based face detection (robust for wild/YouTube videos)
- Lips (96x96) or Face (224x224) crop modes
- Audio extraction (16kHz mono WAV)
- Multilingual support (9 languages)
- Parallel processing support

**Usage:**
```bash
python step1_prepare_muavic_retinaface.py \
    --data-dir /path/to/muavic/data \
    --root-dir /path/to/output \
    --language en \
    --split train \
    --crop-type lips
```

### `step2_generate_file_lists.py`
Generates train/valid/test file lists from processed MuAViC data.

**Features:**
- Creates file lists for each split
- Language-specific organization
- Compatible with MuAViC's predefined splits

**Usage:**
```bash
python step2_generate_file_lists.py \
    --muavic-data-dir /path/to/output/muavic/muavic_video \
    --language en
```

### `step3_metadata_prep.py`
Creates metadata files for training (frame counts, TSV manifests, word files).

**Features:**
- Counts audio and video frames
- Creates TSV manifests (LRS-compatible format)
- Generates word files with transcriptions
- Split-aware processing

**Usage:**
```bash
python step3_metadata_prep.py \
    --muavic-data-dir /path/to/output/muavic/muavic_video \
    --metadata-dir /path/to/output/muavic/metadata \
    --language en \
    --vocab-size 1000
```

## Shared Resources

This directory uses symbolic links to share resources with TCD-TIMIT preprocessing:

- `detectors/` → Links to TCD-TIMIT RetinaFace and MediaPipe detectors
- `utils.py` → Links to TCD-TIMIT utility functions
- `data/` → Links to TCD-TIMIT data loading modules

This ensures consistency across datasets and reduces code duplication.

## Requirements

Same as TCD-TIMIT preprocessing:
- PyTorch with CUDA support
- ibug-face_detection
- ibug-face_alignment
- OpenCV, NumPy, Pandas, tqdm, ffmpeg

See `../../TCD-TIMIT/README.md` for detailed installation instructions.
