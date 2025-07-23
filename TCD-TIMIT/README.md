# TCD-TIMIT Preprocessing

Simple preprocessing pipeline for TCD-TIMIT dataset.

## Files

- `step1_prepare_tcd_timit.py` - Main preprocessing script
- `explore_tcd_timit.py` - Dataset structure explorer  
- `check_video_quality.py` - Video quality checker
- `step2_generate_file_lists.py` - Generate train/val/test splits
- `step3_metadata_prep.py` - Create final metadata
- `parse_mlf.py` - MLF file parser utility

## Usage

### 1. Main Preprocessing (`step1_prepare_tcd_timit.py`)

**Full command example:**
```bash
python step1_prepare_tcd_timit.py \
    --data-dir /home/rishabh/Desktop/Datasets/TCD-TIMIT/ \
    --root-dir /home/rishabh/Desktop/Datasets/tcd_test \
    --subset volunteers \
    --crop-type face \
    --output-size 224 \
    --max-videos 50
```

**Default settings:**
- **Face detector**: OpenCV Haar Cascade (built-in, no GPU needed)
- **Crop type**: `lips` 
- **Output size**: `96` (for lips), `224` (for face/full)

**All options:**
- `--data-dir`: Path to TCD-TIMIT dataset (required)
- `--root-dir`: Output directory (required)
- `--subset`: `volunteers` or `lipspeakers` (required)
- `--crop-type`: `lips`, `face`, or `full` (default: lips)
- `--output-size`: Target resolution like 96, 224, 512 (default: 96)
- `--max-videos`: Limit for testing (optional)

### 2. Dataset Explorer (`explore_tcd_timit.py`)

```bash
python explore_tcd_timit.py \
    --data-dir /home/rishabh/Desktop/Datasets/TCD-TIMIT/
```

Shows dataset structure, video count, resolutions, and file sizes.

### 3. Video Quality Checker (`check_video_quality.py`)

```bash
python check_video_quality.py \
    --processed-dir /home/rishabh/Desktop/Datasets/tcd_test/tcd_timit/tcd_timit_video_seg16s_96x96/
```

**Options:**
- `--processed-dir`: Directory with processed videos (required)
- `--original-dir`: Original videos for comparison (optional)
- `--visual-check`: Show videos for manual inspection (optional)

### 4. Generate Splits (`step2_generate_file_lists.py`)

```bash
python step2_generate_file_lists.py \
    --base-dir /home/rishabh/Desktop/Datasets/tcd_test \
    --dataset tcd_timit
```

Creates speaker-based train/val/test splits.

### 5. Create Metadata (`step3_metadata_prep.py`)

```bash
python step3_metadata_prep.py \
    --base-dir /home/rishabh/Desktop/Datasets/tcd_test \
    --dataset tcd_timit
```

Generates final training manifests.

## Common Examples

```bash
# Quick test (10 videos, lips)
python step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type lips \
    --output-size 96 \
    --max-videos 10

# Face processing (224x224)
python step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type face \
    --output-size 224

# Full video (no cropping, no jitter)
python step1_prepare_tcd_timit.py \
    --data-dir /path/to/TCD-TIMIT \
    --root-dir /path/to/output \
    --subset volunteers \
    --crop-type full \
    --output-size 224

# Process both subsets
python step1_prepare_tcd_timit.py --subset volunteers --crop-type lips --output-size 96
python step1_prepare_tcd_timit.py --subset lipspeakers --crop-type lips --output-size 96
```

## Output Structure
```
output/
├── tcd_timit/
│   ├── tcd_timit_video_seg16s_96x96/     # 96x96 lip videos
│   ├── tcd_timit_video_seg16s_face_224x224/  # 224x224 face videos
│   ├── tcd_timit_text_seg16s/            # Transcript files
│   └── labels/                           # CSV metadata
│       └── tcd_timit_volunteers_96x96.csv
```
