# LRS Combined Dataset

Scripts to combine prepared LRS2 and LRS3 datasets for training.

## Prerequisites

Make sure you have already prepared both LRS2 and LRS3 datasets using their respective preparation scripts (completed all 3 steps for each).

Expected input structure:
```
lrs2_rf/
├── labels/           # CSV files
└── lrs2/
    ├── lrs2_video_seg16s/
    └── lrs2_text_seg16s/

lrs3_rf/
├── labels/           # CSV files  
└── lrs3/
    ├── lrs3_video_seg16s/
    └── lrs3_text_seg16s/
```

## Script 1: combine_datasets.py

Combines video/audio/text files and creates updated CSV files with correct paths. (For Auto_AVSR)

### Usage

```bash
python combine_datasets.py \
  --lrs2-root /path/to/prepared/lrs2 \
  --lrs3-root /path/to/prepared/lrs3 \
  --output-dir /path/to/combined/output
```

### Example

```bash
python combine_datasets.py \
  --lrs2-root /home/rishabh/Desktop/Datasets/lrs2_rf \
  --lrs3-root /home/rishabh/Desktop/Datasets/lrs3_rf \
  --output-dir /home/rishabh/Desktop/Datasets/combine_lrs
```

### Output Structure

```
combine_lrs/
├── labels/
│   ├── lrs_combined_train_transcript_lengths_seg16s.csv
│   ├── lrs_combined_val_transcript_lengths_seg16s.csv
│   └── lrs_combined_test_transcript_lengths_seg16s.csv
├── lrs_combined/
│   ├── lrs_combined_video_seg16s/
│   │   ├── lrs2/ (all LRS2 video/audio files)
│   │   └── lrs3/ (all LRS3 video/audio files)
│   └── lrs_combined_text_seg16s/
│       ├── lrs2/ (all LRS2 text files)
│       └── lrs3/ (all LRS3 text files)
└── dataset_info.txt
```

**Key Features:**
- Copies all video, audio, and text files
- Updates CSV file paths to point to combined structure
- Paths in CSV files start with `lrs_combined/lrs_combined_video_seg16s/lrs2/` or `lrs_combined/lrs_combined_video_seg16s/lrs3/`

## Script 2: combine_lrs_avhubert.py

Combines TSV, WRD, and cluster_counts files (for AV-HuBERT/VSP-LLM).

### Usage

```bash
python combine_lrs_avhubert.py \
  --lrs2 /path/to/lrs2/metadata \
  --lrs3 /path/to/lrs3/metadata \
  --output /path/to/combined/metadata
```

### Example

```bash
python combine_lrs_avhubert.py \
  --lrs2 /home/rishabh/Desktop/Datasets/lrs2_rf/lrs2/lrs2_video_seg16s/data_lrs2/ \
  --lrs3 /home/rishabh/Desktop/Datasets/lrs3/433h_data \
  --output /home/rishabh/Desktop/Datasets/lrs_combined_metadata
```

### Output

```
lrs_combined_metadata/
├── train.tsv, valid.tsv, test.tsv
├── train.wrd, valid.wrd, test.wrd
├── train.cluster_counts, valid.cluster_counts, test.cluster_counts
└── dict.wrd.txt
```

