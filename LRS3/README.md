# LRS3 Dataset Preparation Guide

## Overview
This project is based on [Auto-AVSR data preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation), however we have simplified the scripts to work with ease. You can follow the original Auto-AVSR approach or use the streamlined scripts we provide here.

## New: Single Script Preprocessing

For a simplified approach, use the all-in-one preprocessing script:

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

**Note:** The target directory name depends on your chosen segment size:
- `lrs3_video_seg16s` (for 16-second segments)  
- `lrs3_video_seg24s` (for 24-second segments)

**Example Directory Structure After Auto-AVSR:**
```
/home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/lrs3_video_seg16s/
├── trainval/
├── test/
├── pretrain/
└── (audio/video files will be here)
```

## Metadata Prep

### Step 2: Process Metadata
Generate file lists and copy dataset splits in one command:

```bash
python step2_process_metadata.py \
  --lrs3-root /path/to/original/lrs3 \
  --target-dir /path/to/processed/lrs3/lrs3_video_segXXs
```

**Example:**
```bash
python step2_process_metadata.py \
  --lrs3-root /home/rishabh/Desktop/Datasets/lrs3 \
  --target-dir /home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/lrs3_video_seg16s
```

**Parameters:**
- `--lrs3-root`: Path to your **original** LRS3 dataset (contains lrs3/ folders and text files)
- `--target-dir`: Path to your **processed** segmented video folder from auto_avsr

**Note:** The script will automatically find labels and text files in either:
- The original LRS3 directory (`--lrs3-root`)
- The processed LRS3 directory (parent of `--target-dir`)

**What this does:**
- ✅ Generates `file.list` and `label.list` 
- ✅ Saves them directly to the target directory 
- ✅ Copies dataset split files (`train.txt`, `val.txt`, `test.txt`, `pretrain.txt`, `trainval.txt`)

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

**Note for face/full modes:** If using `--crop-type face` or `--crop-type full`, use the suffixed directories with the step-by-step scripts:
```bash
# For face mode
python step2_process_metadata.py --target-dir /path/to/lrs3_video_seg16s_face
python step3_metadata_prep.py --lrs3-data-dir /path/to/lrs3_video_seg16s_face

# For full mode  
python step2_process_metadata.py --target-dir /path/to/lrs3_video_seg16s_full
python step3_metadata_prep.py --lrs3-data-dir /path/to/lrs3_video_seg16s_full
```

---
## That's it! 🎉

Your LRS3 dataset is now ready for training. The final processed data will be in:
```
/path/to/processed/lrs3/metadata/
├── train.tsv, valid.tsv, test.tsv
├── train.wrd, valid.wrd, test.wrd
├── dict.wrd.txt
└── spm1000/ (vocabulary files)
```

**Example final structure:**
```
/home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/
├── lrs3_video_seg16s/           # Segmented videos from auto_avsr
│   ├── trainval/
│   ├── test/
│   ├── pretrain/
│   ├── file.list, label.list
│   ├── nframes.audio, nframes.video
│   └── train.txt, val.txt, test.txt, pretrain.txt, trainval.txt
└── metadata/                    # Generated manifest files
    ├── train.tsv, valid.tsv, test.tsv
    ├── train.wrd, valid.wrd, test.wrd
    ├── dict.wrd.txt
    └── spm1000/
```

## Directory Structure Expected:
```
Option 1 - Labels in original LRS3:
/home/rishabh/Desktop/Datasets/lrs3/
├── labels/                          # CSV files with transcripts
├── lrs3/lrs3_text_seg16s/          # Text transcripts
└── train.txt, val.txt, test.txt, pretrain.txt, trainval.txt

Option 2 - Labels in processed LRS3 (auto_avsr output):
/home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/
├── labels/                          # CSV files (from auto_avsr)
├── lrs3_text_seg16s/               # Text transcripts (from auto_avsr)
├── train.txt, val.txt, test.txt, pretrain.txt, trainval.txt
└── lrs3_video_seg16s/              # Segmented videos
    ├── trainval/
    ├── test/
    └── pretrain/

Final Output:
/home/rishabh/Desktop/Datasets/lrs3_ffs/lrs3/
├── lrs3_video_seg16s/              # Contains generated files
│   ├── file.list, label.list
│   ├── nframes.audio, nframes.video
│   └── train.txt, val.txt, test.txt, pretrain.txt, trainval.txt
└── metadata/                        # Generated manifest files
    ├── train.tsv, valid.tsv, test.tsv
    └── dict.wrd.txt, etc.
```

## Key Differences from LRS2:
- **Dataset Structure**: LRS3 uses `trainval/`, `test/`, and `pretrain/` instead of `main/` and `pretrain/`
- **Validation Split**: LRS3 validation is extracted from `trainval/` using `val.txt`
- **File Naming**: Adapted for LRS3's speaker_id/video_id.mp4 structure
- **Split Files**: Includes `trainval.txt` in addition to standard splits
