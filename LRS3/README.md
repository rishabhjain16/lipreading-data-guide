# LRS3 Dataset Preparation Guide

## Overview
This project is based on [Auto-AVSR data preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation), however we have simplified the scripts to work with Auto-AVSR and AV-Hubert Kind repo. 

```bash
AVHUBERT Data prep: 
└── metadata/                       # Step 3 output: Final training files
    ├── train.tsv                   # Training manifest
    ├── valid.tsv                   # Validation manifest  
    ├── test.tsv                    # Test manifest
    ├── train.wrd                   # Training transcripts (words)
    ├── valid.wrd                   # Validation transcripts (words)
    ├── test.wrd                    # Test transcripts (words)
    ├── dict.wrd.txt                # Word vocabulary dictionary
    ├── train.phn                   # Training transcripts (phonemes) [OPTIONAL]
    ├── valid.phn                   # Validation transcripts (phonemes) [OPTIONAL]
    ├── test.phn                    # Test transcripts (phonemes) [OPTIONAL]
    ├── dict.phn.txt                # Phoneme vocabulary dictionary [OPTIONAL]
    └── smp1000/                    # SentencePiece vocabulary files
        ├── spm_unigram1000.model
        ├── smp_unigram1000.vocab
        └── spm_unigram1000.txtfollow the original Auto-AVSR approach or use the streamlined scripts we provide here.
```
## Setup
Before processing LRS3 data, download the validation split file:

```bash
# Download the official LRS3 validation split
wget https://raw.githubusercontent.com/facebookresearch/av_hubert/main/avhubert/preparation/data/lrs3-valid.id -P /path/to/your/lrs3/dataset/
```

This ensures proper train/validation separation and prevents data leakage.

## Complete 3-Step Workflow

### Step 1: Video Preprocessing
Process videos with face detection and cropping:

```bash
python preparation/step1_prepare_lrs3_all.py \
  --data-dir /path/to/original/lrs3 \
  --root-dir /path/to/output \
  --subset train \
  --dataset lrs3 \
  --detector mediapipe \
  --crop-type lips
```

### Step 2: Generate File Lists
Generate file.list and label.list from processed CSV files:

```bash
python preparation/step2_generate_file_lists.py \
  --lrs3-data-dir /path/to/processed/lrs3/lrs3_video_seg16s
```

### Step 3: Create Metadata 
Count frames and create manifest files:

```bash
python preparation/step3_metadata_prep.py \
  --lrs3-data-dir /path/to/processed/lrs3/lrs3_video_seg16s \
  --metadata-dir /path/to/processed/lrs3/metadata \
  --vocab-size 1000
```

### Optional: Phoneme Conversion
Convert word transcripts to phoneme transcripts for phoneme-based training:

```bash
python ../Phones/create_phoneme_metadata.py \
  --metadata-dir /path/to/processed/lrs3/metadata
```

**Benefits of Phoneme-Level Training:**
- More fine-grained speech representation
- Better handling of pronunciation variations
- Improved accuracy for lip reading models

**Output:** Creates `.phn` transcript files and `dict.phn.txt` dictionary alongside existing word files.

For detailed phoneme conversion options, see [Phones/README.md](../Phones/README.md).

---

## Detailed Step Instructions

### Step 1: Video Preprocessing

```bash
python preparation/step1_prepare_lrs3_all.py \
  --data-dir /path/to/original/lrs3 \
  --root-dir /path/to/output \
  --subset train \
  --dataset lrs3 \
  --detector mediapipe \
  --crop-type lips
```

**Crop Options:**
- `--crop-type lips`: 96x96 mouth region (default)
- `--crop-type face`: 128x128 face region using landmarks
- `--crop-type full`: Original video size

**Detector Options:**
- `--detector mediapipe`: Use MediaPipe face detector (default)
- `--detector retinaface`: Use RetinaFace detector (requires GPU)

**Important for Validation:**
- For `--subset val`: Requires `lrs3-valid.id` file in your data directory
- This file contains the validation video IDs (one per line)
- **Download from**: [Facebook AV-HuBERT](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation/data/lrs3-valid.id)
- **Data Leakage Prevention**: Training automatically excludes validation files when `lrs3-valid.id` is present
- Without this file, validation processing will fail and training may include validation data

**Output directories:**
- Lips: `lrs3_video_seg16s/`
- Face: `lrs3_video_seg16s_face/`
- Full: `lrs3_video_seg16s_full/`

Process all subsets:
```bash
for subset in train val test; do
  python preparation/step1_prepare_lrs3_all.py \
    --data-dir /path/to/lrs3 \
    --root-dir /path/to/output \
    --subset $subset \
    --dataset lrs3 \
    --detector mediapipe \
    --crop-type lips
done
```

**Output Files:**
- Processed videos in: `lrs3_video_seg16s/` directory
- Processed transcripts in: `lrs3_text_seg16s/` directory  
- CSV metadata files: `lrs3_train_transcript_lengths_seg16s.csv`, `lrs3_val_transcript_lengths_seg16s.csv`, `lrs3_test_transcript_lengths_seg16s.csv`

**Note:** The target directory name depends on your chosen segment size:
- `lrs3_video_seg16s` (for 16-second segments)  
- `lrs3_video_seg24s` (for 24-second segments)

### Step 2: Generate File Lists
Generate file.list and label.list from processed CSV files:

```bash
python step2_generate_file_lists.py \
  --lrs3-data-dir /path/to/processed/lrs3/lrs3_video_segXXs
```

**Example:**
```bash
python step2_generate_file_lists.py \
  --lrs3-data-dir /home/rishabh/Desktop/Datasets/lrs3_test_new/lrs3/lrs3_video_seg16s
```

**Parameters:**
- `--lrs3-data-dir`: Path to your **processed** segmented video folder (contains video files and CSV metadata)
- `--labels-dir` (optional): Path to directory with CSV files (auto-detected if not specified)

**What this does:**
- ✅ Reads CSV files generated by Step 1 (train, val, test transcript lengths)
- ✅ Generates `file.list` with video file paths
- ✅ Generates `label.list` with actual transcript text content
- ✅ Saves both files directly to the data directory for Step 3

### Step 3: Metadata Prep  
Count frames and create manifest files in one command:

```bash
python step3_metadata_prep.py \
  --lrs3-data-dir /path/to/processed/lrs3/lrs3_video_segXXs \
  --metadata-dir /path/to/processed/lrs3/metadata \
  --vocab-size 1000
```

**Example:**
```bash
python step3_metadata_prep.py \
  --lrs3-data-dir /home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/lrs3_video_seg16s \
  --metadata-dir /home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/metadata \
  --vocab-size 1000
```

**Parameters:**
- `--lrs3-data-dir`: Path to your segmented video folder (contains file.list, label.list, trainval/, test/, pretrain/)
- `--metadata-dir`: Path where you want to create the metadata folder with manifest files
- `--vocab-size`: Vocabulary size for sentencepiece (default: 1000)

**What this does:**
- ✅ Counts frames in audio and video files
- ✅ Creates `nframes.audio` and `nframes.video` files
- ✅ Generates sentencepiece vocabulary  
- ✅ Creates manifest TSV files (`train.tsv`, `valid.tsv`, `test.tsv`)
- ✅ Creates word files (`train.wrd`, `valid.wrd`, `test.wrd`)
- ✅ Creates dictionary file (`dict.wrd.txt`)

**Note for face/full modes:** Steps 2 and 3 now **automatically detect** the crop type from directory names and handle the corresponding CSV files:

```bash
# Automatic crop type detection - works for any crop type
python step2_generate_file_lists.py --lrs3-data-dir /path/to/lrs3_video_seg16s_face
python step3_metadata_prep.py --lrs3-data-dir /path/to/lrs3_video_seg16s_face --metadata-dir /path/to/metadata

# Or for full mode  
python step2_generate_file_lists.py --lrs3-data-dir /path/to/lrs3_video_seg16s_full
python step3_metadata_prep.py --lrs3-data-dir /path/to/lrs3_video_seg16s_full --metadata-dir /path/to/metadata
```

The scripts automatically:
- 🎥 **Detect crop type** from directory suffix (`_face`, `_full`, or default `lips`)
- 📄 **Use correct CSV files** (e.g., `lrs3_train_transcript_lengths_seg16s_face.csv`)
- 📁 **Handle all processing** without manual intervention

---

## Directory Structure

After running the complete pipeline, your directory structure will look like this:

```
/path/to/processed/lrs3/
├── lrs3_video_seg16s/              # Step 1 output: Processed videos (lips mode)
│   │                               # OR: lrs3_video_seg16s_face/ (face mode)  
│   │                               # OR: lrs3_video_seg16s_full/ (full mode)
│   ├── trainval/                   # Training videos (speaker_id/video_id.mp4)
│   ├── test/                       # Test videos  
│   ├── pretrain/                   # Pretrain videos (if processed)
│   ├── file.list                   # Step 2 output: Video file paths
│   ├── label.list                  # Step 2 output: Transcript content
│   ├── nframes.audio               # Step 3 output: Audio frame counts
│   └── nframes.video               # Step 3 output: Video frame counts
├── lrs3_text_seg16s/               # Step 1 output: Processed transcripts
│   ├── trainval/                   # Training transcripts (speaker_id/video_id.txt)
│   ├── test/                       # Test transcripts
│   └── pretrain/                   # Pretrain transcripts (if processed)
├── labels/                         # Step 1 output: CSV metadata files
│   ├── lrs3_train_transcript_lengths_seg16s.csv         # (lips mode)
│   ├── lrs3_train_transcript_lengths_seg16s_face.csv    # (face mode) 
│   ├── lrs3_train_transcript_lengths_seg16s_full.csv    # (full mode)
│   ├── lrs3_val_transcript_lengths_seg16s.csv           # (+ val/test variants)
│   └── lrs3_test_transcript_lengths_seg16s.csv
└── metadata/                       # Step 3 output: Final training files
    ├── train.tsv                   # Training manifest
    ├── valid.tsv                   # Validation manifest  
    ├── test.tsv                    # Test manifest
    ├── train.wrd                   # Training transcripts
    ├── valid.wrd                   # Validation transcripts
    ├── test.wrd                    # Test transcripts
    ├── dict.wrd.txt                # Vocabulary dictionary
    └── spm1000/                    # SentencePiece vocabulary files
        ├── spm_unigram1000.model
        ├── spm_unigram1000.vocab
        └── spm_unigram1000.txt
```

## Pipeline Flow

1. **Step 1** processes raw LRS3 videos → creates `lrs3_video_seg16s/`, `lrs3_text_seg16s/`, and CSV files
2. **Step 2** reads CSV files → creates `file.list` and `label.list` in the video directory  
3. **Step 3** reads file/label lists → creates final training manifests in `metadata/`

## Key Features

✅ **Proper Train/Val Split**: Uses official `lrs3-valid.id` to prevent data leakage  
✅ **Path Resolution**: Automatically handles complex LRS3 directory structures  
✅ **CSV-Based Workflow**: Step 1 creates CSV files that Step 2 processes  
✅ **Flexible Vocabulary**: Automatically adjusts vocab size for dataset size  
✅ **Multiple Crop Types**: Supports lips (96x96), face (128x128), and full video  
✅ **Automatic Crop Detection**: Steps 2 & 3 auto-detect crop type from directory names  
✅ **Phoneme Support**: Optional phoneme-level transcripts for enhanced training

---

## That's it! 🎉
