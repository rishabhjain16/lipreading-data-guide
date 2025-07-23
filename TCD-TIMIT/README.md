# TCD-TIMIT Preprocessing

Complete preprocessing pipeline for TCD-TIMIT   
```bash

└── metadata/                                # Training Manifests
       ├── train.txt, val.txt, test.txt           # File lists
       ├── train.tsv, valid.tsv, test.tsv         # LRS-format manifests
       ├── train.wrd, valid.wrd, test.wrd         # Transcriptions
       ├── dict.wrd.txt                           # Vocabulary
       ├── nframes.audio, nframes.video           # Frame counts
       └── vocab filestaset with HD video processing, audio extraction, and multiple output formats.
```

## Quick Start

```bash
# Step 1: Process videos (face mode, 224x224)
python preparation/step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type face \
    --output-size 224

# Step 2: Generate splits (reproducible with seed)
python preparation/step2_generate_file_lists.py \
    --tcd-data-dir /path/to/output/tcd_timit/tcd_timit_video_seg16s_face_224x224 \
    --seed 42

# Step 3: Create metadata
python preparation/step3_metadata_prep.py \
    --tcd-data-dir /path/to/output/tcd_timit/tcd_timit_video_seg16s_face_224x224 \
    --metadata-dir /path/to/output/tcd_timit/metadata
```

## Key Features

### Step 1: Video Preprocessing
- **HD Processing**: Uses full 1920x1080 resolution for accurate face detection
- **Face Detection**: OpenCV Haar Cascade (optimized parameters for stability)
- **Multiple Crop Modes**: 
  - `lips`: Mouth region only (96x96) - Best for pure lip-reading models
  - `face`: Tight face crop with 5% padding (224x224) - Balanced face/lip context
  - `full`: Extended face area with 40% padding (224x224+) - Full head context
- **Audio Extraction**: Co-located 16kHz mono WAV files using FFmpeg
- **Dual Text Formats**: 
  - `.phn` files: Phonemes from MLF (e.g., "sh iy hh ae d...")
  - `.txt` files: TIMIT sentence mapping (e.g., "She had your dark suit...")
- **Unique Naming**: `{speaker}_{transcript}` format (e.g., `01M_sa1`) prevents conflicts
- **Temporal Smoothing**: Advanced jitter reduction (0.8 smoothing factor)
- **Quality Output**: Clean progress display with stability metrics

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
│   ├── tcd_timit_video_seg16s_face_224x224/     # Videos + Audio
│   │   └── volunteers/01M/
│   │       ├── 01M_sa1.mp4                      # Video
│   │       ├── 01M_sa1.wav                      # Audio  
│   │       └── ...
│   ├── tcd_timit_text_seg16s_face/              # Text Files
│   │   └── volunteers/01M/
│   │       ├── 01M_sa1.phn                      # Phonemes
│   │       ├── 01M_sa1.txt                      # Sentences
│   │       └── ...
│   ├── labels/                                  # CSV Metadata
│   │   └── tcd_timit_volunteers_face_224x224.csv
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

## Face Detection Comparison

### OpenCV Haar Cascade (Current Implementation)
✅ **Pros**: 
- Fast and lightweight
- No additional dependencies 
- Stable across different lighting conditions
- Good temporal consistency with smoothing
- Works well with TCD-TIMIT's controlled environment

❌ **Cons**:
- Less accurate than deep learning methods
- May struggle with extreme poses/lighting

### MediaPipe (Alternative)
✅ **Pros**: 
- More accurate face detection
- Better landmark detection
- Good real-time performance

❌ **Cons**:
- Requires additional pip install
- More complex integration
- May have temporal inconsistencies

### RetinaFace (Alternative) 
✅ **Pros**:
- State-of-the-art accuracy
- Excellent for challenging conditions

❌ **Cons**:
- Requires deep learning models
- Slower processing
- Heavy dependencies (PyTorch)

**Recommendation**: For TCD-TIMIT's controlled studio environment, OpenCV Haar Cascade provides the best balance of speed, stability, and accuracy. The temporal smoothing we've implemented addresses most consistency issues.

## Examples

```bash
# Lips processing (96x96, for pure lip-reading models)
# Best for: Lightweight models, mobile deployment, research focusing on lip motion
python preparation/step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type lips \
    --output-size 96

# Balanced face processing (224x224, recommended)
# Best for: General lip-reading, good face/lip context, most versatile
python preparation/step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type face \
    --output-size 224

# High-resolution processing (512x512)
# Best for: Research, fine-grained analysis, when compute isn't limited
python preparation/step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type face \
    --output-size 512

# Full head context (maximum information)
# Best for: Multimodal models, gesture analysis, head pose studies
python preparation/step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT/ \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type full \
    --output-size 224

# Custom splits for research reproducibility
python preparation/step2_generate_file_lists.py \
    --tcd-data-dir /path/to/processed/videos \
    --split-ratios "0.8,0.1,0.1" \
    --seed 123  # Always use the same seed for reproducible results

# Process both subsets for complete dataset
for subset in volunteers lipspeakers; do
    echo "Processing $subset subset..."
    python preparation/step1_prepare_tcd_timit.py \
        --data-dir /path/to/TCD-TIMIT/ \
        --root-dir /path/to/output \
        --subset $subset \
        --crop-type face \
        --output-size 224
done
```

## Dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

Main dependencies: OpenCV, NumPy, Pandas, tqdm, ffmpeg
