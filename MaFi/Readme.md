# MaFi Dataset Preprocessing

Complete preprocessing pipeline for MaFi (Massive Faces) dataset - a word guessing dataset with silent videos for lip reading evaluation.

## Dataset Overview

- **Speakers**: 5 (A1, A2, B1, B2, B3)
- **Total videos**: 2,519
- **Video format**: 1920x1080, 25 fps, ~1.8 seconds
- **Audio**: None (silent/muted videos)
- **Transcripts**: Extracted from filenames

## Quick Start

```bash
# Step 1: Process videos with RetinaFace
python preparation/step1_prepare_mafi.py \
    --data-dir "/media/rishabhjain/SSD/MaFi/Videos Folder" \
    --root-dir /path/to/output \
    --crop-type lips

# Step 2: Generate file lists (optional)
python preparation/step2_generate_file_lists.py \
    --mafi-data-dir /path/to/output/mafi_video

# Step 3: Create metadata (optional)
python preparation/step3_metadata_prep.py \
    --mafi-data-dir /path/to/output/mafi_video \
    --metadata-dir /path/to/output/metadata \
    --vocab-size 1000
```

## Dataset Structure

```
/media/rishabhjain/SSD/MaFi/Videos Folder/
├── A1/
│   └── Stimuli_Anna's study/
│       ├── cauliflower_clear_small_muted.mp4
│       ├── paintbrush_clear_small_muted.mp4
│       └── ... (816 videos)
├── A2/
│   └── stimuli_A2/
│       ├── working.mp4
│       ├── office.mp4
│       └── ... (121 videos)
├── B1/
│   └── stimuli_B1/
│       ├── big.mp4
│       ├── official.mp4
│       └── ... (680 videos)
├── B2/
│   └── stimuli_B2/
│       └── ... (315 videos)
└── B3/
    └── stimuli_B3/
        └── ... (587 videos)
```

## Output Structure

```
output/
├── mafi_video/                              # Videos (lips, 96x96)
│   ├── A1/
│   │   ├── cauliflower.mp4
│   │   ├── paintbrush.mp4
│   │   └── ...
│   ├── A2/, B1/, B2/, B3/
│   ├── file.list
│   ├── label.list
│   └── nframes.video
├── mafi_video_face_224x224/                 # Videos (face, 224x224)
├── mafi_text/                               # Text Files (lips)
│   ├── A1/
│   │   ├── cauliflower.txt
│   │   ├── paintbrush.txt
│   │   └── ...
│   ├── A2/, B1/, B2/, B3/
├── mafi_text_face_224x224/                  # Text Files (face)
└── labels/                                  # CSV Metadata
    ├── mafi_all.csv                         # All speakers (2,519 videos)
    ├── mafi_A1.csv                          # Speaker A1 (816 videos)
    ├── mafi_A2.csv                          # Speaker A2 (121 videos)
    ├── mafi_B1.csv                          # Speaker B1 (680 videos)
    ├── mafi_B2.csv                          # Speaker B2 (315 videos)
    └── mafi_B3.csv                          # Speaker B3 (587 videos)
```

## Options

### Step 1
- `--crop-type`: `lips` (96x96) or `face` (224x224)
- `--groups`, `--job-index`: For parallel processing

### Step 2
- `--speaker`: Process specific speaker (A1, A2, B1, B2, B3) or omit for all

### Step 3
- `--speaker`: Process specific speaker or omit for all
- `--vocab-size`: Vocabulary size (default: 1000)

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install ibug-face_detection ibug-face_alignment
pip install opencv-python pandas tqdm
```

## Notes

- MaFi videos are **silent** (no audio processing)
- Transcripts are extracted from filenames
- Speaker A1 has complex filenames that are automatically cleaned
- Designed for testing/evaluation (single-word utterances)
