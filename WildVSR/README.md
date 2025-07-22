# WildVSR Dataset Preparation

Simple preparation of WildVSR dataset for inference.

## Overview

WildVSR is a test dataset for visual speech recognition containing challenging real-world videos.

## Requirements

No additional dependencies required - uses only Python standard library.

## Dataset Structure

Your WildVSR dataset should be organized as:
```
wildvsr_dataset/
├── videos/           # Video files (.mp4)
└── labels.json       # Transcriptions
```

## Usage

### 1. Basic Data Preparation

```bash
python test_prep.py /path/to/wildvsr/dataset
```

This script:
- Reads video files and transcriptions from `labels.json`
- Creates `test_data/` folder with:
  - `test.tsv` - manifest file for the dataset
  - `test.wrd` - normalized transcriptions

## Output

After running `test_prep.py`, you'll get:

```
wildvsr_dataset/
├── videos/
├── labels.json
└── test_data/          # Created by script
    ├── test.tsv        # Dataset manifest
    └── test.wrd        # Cleaned transcriptions
```

### test.tsv format:
```
/absolute/path/to/videos
video_id1    /path/to/video1.mp4    /path/to/audio1.wav    0    0
video_id2    /path/to/video2.mp4    /path/to/audio2.wav    0    0
...
```

### test.wrd format:
```
normalized transcript 1
normalized transcript 2
...
```

## Next Steps

After basic preparation:

1. **For Clustering**: Follow the clustering procedure from the main documentation
2. **For HuBERT Features**: Extract features using the clustering scripts
3. **For Inference**: Use the generated `test.tsv` and `test.wrd` files

## Notes

- The script handles malformed JSON files with fallback parsing
- Text normalization removes punctuation and converts to lowercase
- Frame counts are set to 0 as placeholders (to be updated later if needed)
