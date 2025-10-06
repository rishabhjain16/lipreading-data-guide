# Lombard GRID Dataset Preprocessing

Complete preprocessing pipeline for Lombard GRID - an audiovisual speech corpus recorded in noisy conditions (Lombard effect).

## Dataset Overview

- **Speakers**: 54 (s2-s55)
- **Total videos**: 5,390
- **Conditions**: 2 main conditions
  - `_l_`: Lombard speech (2,699 videos)
  - `_p_`: Plain noise (2,690 videos)
- **Sentence structure**: Same as GRID (6 words)
- **Video format**: 720x480, ~24 fps, MOV files
- **Duration**: ~2.5 seconds per video
- **Views**: Front and side camera angles

## Quick Start

```bash
# Step 1: Process videos with RetinaFace (front view)
python preparation/step1_prepare_lombardgrid.py \
    --data-dir /media/rishabhjain/SSD/lombardgrid \
    --root-dir /media/rishabhjain/SSD/LombardGrid_Clean \
    --crop-type lips \
    --view front

# Step 1: Process side view (optional)
python preparation/step1_prepare_lombardgrid.py \
    --data-dir /media/rishabhjain/SSD/lombardgrid \
    --root-dir /media/rishabhjain/SSD/LombardGrid_Clean \
    --crop-type lips \
    --view side

# Step 2: Generate file lists (optional)
python preparation/step2_generate_file_lists.py \
    --lombardgrid-data-dir /media/rishabhjain/SSD/LombardGrid_Clean/lombardgrid_video

# Step 3: Create metadata (optional)
python preparation/step3_metadata_prep.py \
    --lombardgrid-data-dir /media/rishabhjain/SSD/LombardGrid_Clean/lombardgrid_video \
    --metadata-dir /media/rishabhjain/SSD/LombardGrid_Clean/metadata \
    --split-ratios 0.7,0.15,0.15 \
    --vocab-size 100
```

## Dataset Structure

```
/media/rishabhjain/SSD/lombardgrid/
├── front/
│   ├── s10_l_bbat9p.mov                     # Speaker 10, lombard, utterance
│   ├── s10_l_bbay5n.mov
│   ├── s2_p_bbaf2n.mov                      # Speaker 2, plain noise
│   └── ... (5,390 videos)
├── side/
│   └── ... (5,390 videos, side view)
├── audio/
│   ├── s10_l_bbat9p.wav
│   └── ...
├── alignment/
│   ├── s10_l_bbat9p.json                    # Phone-level alignments
│   └── ...
└── json/
    └── ... (metadata)
```

## Output Structure

```
/media/rishabhjain/SSD/LombardGrid_Clean/
├── lombardgrid_video/                       # Videos + Audio (lips, 96x96)
│   ├── front/                               # Front view
│   │   ├── s2/
│   │   │   ├── s2_l_bbaf2n.mp4             # Lombard condition
│   │   │   ├── s2_l_bbaf2n.wav
│   │   │   ├── s2_p_bbaf3s.mp4             # Plain noise condition
│   │   │   ├── s2_p_bbaf3s.wav
│   │   │   └── ... (~100 videos per speaker)
│   │   ├── s3/, s4/, ..., s55/
│   ├── side/                                # Side view (if processed)
│   │   ├── s2/
│   │   │   ├── s2_l_bbaf2n.mp4
│   │   │   ├── s2_l_bbaf2n.wav
│   │   │   └── ...
│   │   ├── s3/, s4/, ..., s55/
│   ├── file.list
│   ├── label.list
│   └── nframes.audio, nframes.video
├── lombardgrid_video_face_224x224/          # Videos + Audio (face, 224x224)
│   ├── front/
│   │   └── s2/, s3/, ..., s55/
│   ├── side/
│   │   └── s2/, s3/, ..., s55/
├── lombardgrid_text/                        # Text Files (lips)
│   ├── front/
│   │   ├── s2/
│   │   │   ├── s2_l_bbaf2n.txt             # "bin blue at f two now"
│   │   │   ├── s2_p_bbaf3s.txt             # "bin blue at f three soon"
│   │   │   └── ...
│   │   ├── s3/, s4/, ..., s55/
│   ├── side/
│   │   └── s2/, s3/, ..., s55/
├── lombardgrid_text_face_224x224/           # Text Files (face)
│   ├── front/
│   │   └── s2/, s3/, ..., s55/
│   ├── side/
│   │   └── s2/, s3/, ..., s55/
├── labels/                                  # CSV Metadata
│   ├── lombardgrid_front.csv                # Front view all speakers (~2,700 videos)
│   └── lombardgrid_side.csv                 # Side view all speakers (~2,700 videos)
└── metadata/                                # Training Manifests (optional)
    ├── train.tsv, valid.tsv, test.tsv
    ├── train.wrd, valid.wrd, test.wrd
    ├── dict.wrd.txt
    └── spm100/
```

## Options

### Step 1
- `--crop-type`: `lips` (96x96) or `face` (224x224)
- `--view`: `front` or `side` (camera angle)
- `--groups`, `--job-index`: For parallel processing

### Step 2
- `--speaker`: Process specific speaker (s2, s3, ..., s55) or omit for all

### Step 3
- `--speaker`: Process specific speaker or omit for all
- `--split-ratios`: Train/val/test ratios (default: 0.7,0.15,0.15)
- `--vocab-size`: Vocabulary size (default: 100)

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install ibug-face_detection ibug-face_alignment
pip install opencv-python pandas tqdm
```

## Notes

- Lombard GRID extends GRID corpus with noisy conditions
- Same 6-word grammar as GRID (51 words)
- Phone-level alignments available in JSON format
- Two main recording conditions: lombard (`_l_`) and plain noise (`_p_`)
- Both front and side camera views available
- Condition info preserved in filenames, not as separate folders
