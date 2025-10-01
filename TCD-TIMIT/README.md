# TCD-TIMIT Preprocessing

Complete preprocessing pipeline for TCD-TIMIT dataset with HD video processing, audio extraction, and multiple output formats.

**Note**: The RetinaFace detector is similar to Auto-AVSR data preparation codebase, modified to work with TCD-TIMIT dataset. RetinaFace tends to work better with 1080p videos of TCD-TIMIT as compared to OpenCV Haar cascades, providing superior face detection and landmark accuracy for high-resolution video processing. 


## Quick Start

```bash
# Step 1: Process videos with RetinaFace (lips, 96x96)
python preparation/step1_prepare_tcd_retinaface.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type lips

# Step 1: Process videos with RetinaFace (face, 224x224)
python preparation/step1_prepare_tcd_retinaface.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type face

# Step 2: Generate splits (reproducible with seed)
python preparation/step2_generate_file_lists.py \
    --tcd-data-dir /path/to/output/tcd_timit/tcd_timit_video \
    --seed 42

# Step 3: Create metadata
python preparation/step3_metadata_prep.py \
    --tcd-data-dir /path/to/output/tcd_timit/tcd_timit_video \
    --metadata-dir /path/to/output/tcd_timit/metadata
```

## Key Features

### Step 1: Video Preprocessing with RetinaFace
- **HD Processing**: Uses full 1920x1080 resolution for accurate face detection
- **RetinaFace Detection**: High-accuracy face detection and landmark localization, consistent with VoxCeleb2 preprocessing
- **Two Crop Modes**: 
  - `lips`: Mouth region only (96x96) - Best for pure lip-reading models
  - `face`: Full face crop (224x224) - Balanced face/lip context for multimodal models
- **Audio Extraction**: Co-located 16kHz mono WAV files using FFmpeg
- **Text Files**: 
  - `.txt` files: TIMIT sentence mapping (e.g., "She had your dark suit...")
- **Unique Naming**: `{speaker}_{session}_{camera}_{transcript}` format prevents conflicts
- **Temporal Smoothing**: Advanced jitter reduction with RetinaFace landmark tracking
- **Color Output**: RGB videos (not grayscale) for better visual quality

### Step 2: Data Splits
- **Speaker-Based**: No speaker leakage between train/val/test
- **Reproducible**: Fixed seed (default: 42) for consistent splits
- **Configurable**: Custom ratios (default: 70/15/15)

### Step 3: Metadata Generation
- **Frame Counting**: Audio/video synchronization (`nframes.audio`, `nframes.video`)
- **TSV Manifests**: LRS-compatible format with audio/video paths (`train.tsv`, `valid.tsv`, `test.tsv`)
- **Word Files**: Text transcriptions (`train.wrd`, `valid.wrd`, `test.wrd`)
- **Vocabulary**: SentencePiece tokenization (`dict.wrd.txt`)
- **Training Ready**: Compatible with LRS2/LRS3 training pipelines

## File Structure

```
output/
├── tcd_timit/
│   ├── tcd_timit_video/                         # Videos + Audio (lips, 96x96)
│   │   └── volunteers/01M/Clips/30degcam/
│   │       ├── 01M_Clips_30degcam_sa1.mp4       # Video
│   │       ├── 01M_Clips_30degcam_sa1.wav       # Audio  
│   │       └── ...
│   ├── tcd_timit_video_face_224x224/            # Videos + Audio (face, 224x224)
│   ├── tcd_timit_text/                          # Text Files (lips)
│   │   └── volunteers/01M/Clips/30degcam/
│   │       ├── 01M_Clips_30degcam_sa1.txt       # Sentences
│   │       └── ...
│   ├── tcd_timit_text_face_224x224/             # Text Files (face)
│   ├── labels/                                  # CSV Metadata
│   │   ├── tcd_timit_volunteers_retinaface.csv
│   │   └── tcd_timit_volunteers_face_224x224_retinaface.csv
│   └── metadata/                                # Training Manifests
│       ├── train.txt, val.txt, test.txt
│       └── vocab files
```

## Utility Scripts

```bash
# Explore dataset structure
python preparation/explore_tcd_timit.py --data-dir /path/to/TCD-TIMIT/

# Check video quality/stability
python preparation/check_video_quality.py \
    --processed-dir /path/to/output/videos/

# Parse MLF files manually
python preparation/parse_mlf.py --mlf-file /path/to/file.mlf
```

## Options

### Step 1 Options
- `--crop-type`: Choose processing mode based on your use case:
  - `lips`: Mouth region only (96x96) - For pure lip-reading, fastest processing
  - `face`: Balanced face crop (224x224) - Good face/lip context, recommended
  - `full`: Extended head area (224x224+) - Maximum context, slower processing
- `--output-size`: Target resolution (96, 224, 512, etc.) - Higher = better quality
- `--subset`: `volunteers` (59 speakers) or `lipspeakers` (56 speakers)
- `--max-videos`: Limit processing for testing (e.g., 100 videos)

### Step 2 Options  
- `--split-ratios`: Train/val/test ratios (default: "0.7,0.15,0.15")
- `--seed`: Random seed for reproducible splits (default: 42)

### Step 3 Options
- `--vocab-size`: SentencePiece vocabulary size (default: 1000)

## RetinaFace Processing

**Note**: The RetinaFace detector is similar to Auto-AVSR data preparation codebase, modified to work with TCD-TIMIT dataset. RetinaFace tends to work better with 1080p videos of TCD-TIMIT as compared to OpenCV Haar cascades, providing superior face detection and landmark accuracy for high-resolution video processing.

### Why RetinaFace?
- **Higher Accuracy**: Superior face detection and landmark localization compared to other methods
- **Robust**: Handles pose variations and challenging lighting conditions well
- **Consistent**: Same preprocessing pipeline as VoxCeleb2 for dataset compatibility
- **Research Grade**: Used in state-of-the-art lip-reading research
- **HD Optimized**: Specifically designed to leverage TCD-TIMIT's 1080p video quality

### Requirements
- **GPU**: CUDA-capable GPU recommended for optimal performance
- **Dependencies**: `pip install ibug-face_detection ibug-face_alignment torch torchvision torchaudio`

## Examples

```bash
# Lips processing (96x96, recommended for lip-reading)
python preparation/step1_prepare_tcd_retinaface.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type lips

# Face processing (224x224, for multimodal models)
python preparation/step1_prepare_tcd_retinaface.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type face

# Process both subsets
for subset in volunteers lipspeakers; do
    echo "Processing $subset subset..."
    python preparation/step1_prepare_tcd_retinaface.py \
        --data-dir /path/to/TCD-TIMIT/ \
        --root-dir /path/to/output \
        --subset $subset \
        --crop-type lips
done

# Custom splits for research reproducibility
python preparation/step2_generate_file_lists.py \
    --tcd-data-dir /path/to/processed/videos \
    --split-ratios "0.8,0.1,0.1" \
    --seed 123  # Always use the same seed for reproducible results

# Test with small subset first (process ~117 files instead of 11,662)
python preparation/step1_prepare_tcd_retinaface.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type lips \
    --groups 100 \
    --job-index 0

# Parallel processing (split into 4 jobs)
for i in {0..3}; do
    python preparation/step1_prepare_tcd_retinaface.py \
        --data-dir /path/to/TCD-TIMIT/ \
        --root-dir /path/to/output \
        --subset volunteers \
        --crop-type lips \
        --groups 4 \
        --job-index $i &
done
wait
```

## Dependencies

### Installation:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
pip install ibug-face_detection ibug-face_alignment
```

Main dependencies: OpenCV, NumPy, Pandas, tqdm, ffmpeg, PyTorch

**Note**: RetinaFace requires a CUDA-capable GPU for optimal performance.
