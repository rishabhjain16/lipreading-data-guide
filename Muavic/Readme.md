# MuAViC Dataset Preprocessing

Complete preprocessing pipeline for MuAViC (Multilingual Audio-Visual Corpus) dataset with RetinaFace detection, audio extraction, and multilingual support.

**Note**: MuAViC is a multilingual audio-visual speech translation dataset from YouTube videos. The preprocessing pipeline uses RetinaFace for robust face detection in wild/in-the-wild videos with varying quality, poses, and lighting conditions.

**Original Repository**: https://github.com/facebookresearch/muavic

## Dataset Overview

MuAViC contains audio-visual speech data in 9 languages:
- English (en)
- Spanish (es)
- French (fr)
- Portuguese (pt)
- Italian (it)
- Greek (el)
- Arabic (ar)
- German (de)
- Russian (ru)

## Preprocessing Workflow

The MuAViC preprocessing workflow:

1. **Download Stage (Step 0)**: Use official MuAViC script to download videos
   - Downloads videos from YouTube
   - Segments videos based on timestamps
   - Extracts audio
   - Creates metadata
   - **Note**: The official script also does face cropping, but we'll ignore those cropped videos

2. **RetinaFace Stage (Step 1)**: Apply RetinaFace to the raw downloaded videos
   - Use the raw videos from `mtedx/video/{lang}/` (before MuAViC's cropping)
   - Apply RetinaFace face detection (superior to MuAViC's default)
   - Create Auto-AVSR compatible preprocessed data

3. **Metadata Stage (Step 2-3)**: Generate file lists and training metadata

This approach allows us to:
- Leverage MuAViC's robust download and segmentation
- Use superior RetinaFace face detection instead of their default method
- Create Auto-AVSR compatible preprocessed data

## Quick Start

### Step 0: Download Raw MuAViC Data

We've modified the official MuAViC `get_data.py` to skip their face cropping step. It will:
- Download videos from YouTube
- Segment videos based on timestamps  
- Extract audio files
- **Skip face detection/cropping** (we'll use RetinaFace instead)

```bash
cd muavic
python get_data.py --root-path /path/to/muavic_data --src-lang ar
```

**What gets downloaded:**
- Raw videos: `mtedx/video/{lang}/` (full-frame, unsegmented)
- Audio files: `muavic/{lang}/audio/` (segmented)
- Metadata: `mtedx/{lang}-{lang}/data/` (transcripts, timestamps)

**Note**: We commented out the `preprocess_mtedx_video()` and manifest creation steps in `get_data.py`. These will be replaced by RetinaFace preprocessing in step 1.

### Expected Folder Structure After Download (Step 0)

After running the modified `get_data.py`, your directory structure will look like this:

```
muavic_data/
├── mtedx/                                   # mTEDx data (for non-English languages)
│   ├── ar-ar/                              # Arabic data
│   │   └── data/
│   │       ├── train/
│   │       │   ├── txt/
│   │       │   │   ├── segments            # Segment timestamps (seg_id, video_id, start, end)
│   │       │   │   └── train.ar            # Transcripts
│   │       │   └── wav/                    # Original audio files
│   │       ├── valid/
│   │       └── test/
│   ├── video/                              # Raw downloaded videos from YouTube
│   │   └── ar/
│   │       ├── train/
│   │       │   ├── video_001.mp4           # Full-frame raw videos
│   │       │   ├── video_002.mp4
│   │       │   └── ...
│   │       ├── valid/
│   │       └── test/
│   └── mtedx_ar.tgz                        # Downloaded archive
├── muavic/                                  # Processed data
│   └── ar/
│       ├── audio/                          # Segmented audio files (16kHz mono WAV)
│       │   ├── train/
│       │   │   ├── segment_001.wav
│       │   │   └── ...
│       │   ├── valid/
│       │   └── test/
│       ├── train.tsv                       # Manifest (created by get_data.py)
│       ├── valid.tsv
│       ├── test.tsv
│       ├── train.ar                        # Transcripts
│       ├── valid.ar
│       └── test.ar
├── lrs3/                                    # For English (if using LRS3)
├── metadata/                                # Temporary metadata
├── mt_trans/                                # Temporary translation data
└── ted2020/                                 # Temporary TED2020 data
```

**Key Directories for RetinaFace Processing:**
- **Raw videos**: `mtedx/video/{lang}/{split}/` - Full-frame videos downloaded from YouTube
- **Segment info**: `mtedx/{lang}-{lang}/data/{split}/txt/segments` - Timestamps for segmentation
- **Transcripts**: `mtedx/{lang}-{lang}/data/{split}/txt/{split}.{lang}` - Text transcriptions
- **Audio**: `muavic/{lang}/audio/{split}/` - Already segmented and processed audio

**What's missing (intentionally):**
- `muavic/{lang}/video/{split}/` - This is where MuAViC would save cropped videos, but we commented that out
- We'll create this directory with RetinaFace-processed videos in step 1

### Step 1: Preprocess with RetinaFace

After downloading raw videos with step0, apply RetinaFace preprocessing:

```bash
# Basic usage (lips, 96x96)
python preparation/step1_prepare_muavic_retinaface.py \
    --data-dir /path/to/muavic/data \
    --root-dir /path/to/output \
    --language ar \
    --split test \
    --crop-type lips

# Face processing (224x224, for multimodal models)
python preparation/step1_prepare_muavic_retinaface.py \
    --data-dir /path/to/muavic/data \
    --root-dir /path/to/output \
    --language ar \
    --split test \
    --crop-type face

# With face filtering (removes slides/audience shots)
python preparation/step1_prepare_muavic_retinaface.py \
    --data-dir /path/to/muavic/data \
    --root-dir /path/to/output \
    --language ar \
    --split test \
    --crop-type lips \
    --face-threshold 0.7
```

**Face Threshold Parameter:**

The `--face-threshold` parameter controls quality filtering based on face presence and size.

**How it works:**

1. **Threshold = 0.0 (default)**:
   - No quality filtering
   - Processes all segments where RetinaFace can detect faces
   - Includes blurry, small, or partially visible faces
   - Maximum dataset size
   - Recommended for most use cases

2. **Threshold > 0 (e.g., 0.7)**:
   - Quality filtering enabled
   - For each frame, checks:
     - Is a face detected? ✓
     - Is the face large enough? (≥40px for lips, ≥80px for face) ✓
   - Counts frames that pass both checks
   - Skips segment if < 70% of frames have good-quality faces

**What gets filtered when threshold > 0:**
- ✅ Slides/audience shots (no face detected)
- ✅ Blurry faces (face too small in frame)
- ✅ Far-away shots (face < minimum size)
- ✅ Segments with frequent camera cuts away from speaker

**Example with threshold = 0.7:**
```
Segment has 100 frames:
- 75 frames: Face detected, size 60x60px → ✅ Good quality
- 20 frames: Face detected, size 15x15px → ❌ Too small/blurry
- 5 frames: No face detected → ❌ No face

Valid frames: 75/100 = 75% ≥ 70% → ✅ Segment kept
```

**Recommended values:**
- `0.0`: Maximum data (default)
- `0.7`: Balanced quality filtering
- `0.9`: Strict, only high-quality segments

**Note**: The script uses mTEDx pre-made segments which are already optimized for ASR (better than raw VTT subtitles)

### Step 2: Generate File Lists and Metadata

```bash
# Generate file lists (step 2)
python preparation/step2_generate_file_lists.py \
    --muavic-data-dir /path/to/output/muavic/muavic_video \
    --language en

# Create metadata for training (step 3)
python preparation/step3_metadata_prep.py \
    --muavic-data-dir /path/to/output/muavic/muavic_video \
    --metadata-dir /path/to/output/muavic/metadata \
    --language en
```

## File Structure

After preprocessing, your directory structure will look like:

```
output/
├── muavic/
│   ├── muavic_video/                            # Videos + Audio (lips, 96x96)
│   │   └── en/
│   │       └── train/
│   │           ├── video_id_001.mp4             # Video
│   │           ├── video_id_001.wav             # Audio
│   │           └── ...
│   ├── muavic_video_face_224x224/               # Videos + Audio (face, 224x224)
│   ├── muavic_text/                             # Text Files (lips)
│   │   └── en/
│   │       └── train/
│   │           ├── video_id_001.txt             # Transcription
│   │           └── ...
│   ├── muavic_text_face_224x224/                # Text Files (face)
│   ├── labels/                                  # CSV Metadata
│   │   ├── muavic_en_train_retinaface.csv
│   │   └── muavic_en_train_face_224x224_retinaface.csv
│   └── metadata/                                # Training Manifests
│       ├── train.txt, valid.txt, test.txt
│       └── vocab files
```

## Key Features

### RetinaFace Processing for Wild Videos
- **Robust Detection**: Handles varying quality, poses, and lighting in YouTube videos
- **High Accuracy**: Superior face detection and landmark localization
- **Consistent**: Same preprocessing pipeline as VoxCeleb2 and TCD-TIMIT
- **Multilingual**: Supports all 9 MuAViC languages

### Preprocessing Features
- **Two Crop Modes**:
  - `lips`: Mouth region only (96x96) - Best for pure lip-reading models
  - `face`: Full face crop (224x224) - Balanced face/lip context for multimodal models
- **Audio Extraction**: Co-located 16kHz mono WAV files using FFmpeg
- **Multilingual Support**: Process any of the 9 supported languages
- **Split-aware**: Handles train/valid/test splits separately
- **Color Output**: RGB videos (not grayscale) for better visual quality

## Examples

### Process Multiple Languages

```bash
# Process English, Spanish, and French
for lang in en es fr; do
    echo "Processing $lang..."
    python preparation/step1_prepare_muavic_retinaface.py \
        --data-dir /path/to/muavic/data \
        --root-dir /path/to/output \
        --language $lang \
        --split train \
        --crop-type lips
done

# With face filtering to remove slides/audience
for lang in en es fr; do
    echo "Processing $lang..."
    python preparation/step1_prepare_muavic_retinaface.py \
        --data-dir /path/to/muavic/data \
        --root-dir /path/to/output \
        --language $lang \
        --split train \
        --crop-type lips \
        --face-threshold 0.7
done
```

### Process All Splits for One Language

```bash
# Process train, valid, and test splits for Arabic
for split in train valid test; do
    echo "Processing $split split..."
    python preparation/step1_prepare_muavic_retinaface.py \
        --data-dir /path/to/muavic/data \
        --root-dir /path/to/output \
        --language ar \
        --split $split \
        --crop-type lips
done
```

## Dependencies

**For detailed installation instructions and tools setup, refer to the `../tools/` folder.**

### Step 0 (Download) Dependencies:
```bash
# Fix numpy/OpenCV compatibility first
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install numpy==1.23.5
pip install opencv-python==4.8.1.78

# Install other dependencies
pip install pandas tqdm ffmpeg-python yt-dlp wget
```

**Optional but recommended:**
```bash
# Install SoX for audio processing (removes warning)
sudo apt-get install sox libsox-fmt-all
```

### Step 1-3 (RetinaFace) Dependencies:
```bash
pip install torch torchvision torchaudio
pip install ibug-face_detection ibug-face_alignment
```

**Important**: If you get `AttributeError: _ARRAY_API not found` or `numpy.core.multiarray failed to import`, you have a numpy/OpenCV version conflict. Fix it with:
```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install numpy==1.23.5
pip install opencv-python==4.8.1.78
```

Main dependencies: OpenCV, NumPy, Pandas, tqdm, ffmpeg, PyTorch

**Note**: RetinaFace (Step 1) requires a CUDA-capable GPU for optimal performance. See `../tools/` directory for complete setup instructions.

## Options

### Step 1 Options
- `--data-dir`: Path to downloaded MuAViC dataset
- `--root-dir`: Output directory for preprocessed data
- `--language`: Language code (en, es, fr, pt, it, el, ar, de, ru)
- `--split`: Dataset split (train, valid, test)
- `--crop-type`: Choose processing mode:
  - `lips`: Mouth region only (96x96) - For pure lip-reading
  - `face`: Full face crop (224x224) - For multimodal models
- `--face-threshold`: Face quality filtering (0.0-1.0, default: 0.0)
  - `0.0`: **No filtering** - processes all segments (default, maximum data)
  - `0.7`: **Quality filtering** - requires 70% of frames to have good faces
  - `0.9`: **Strict filtering** - requires 90% of frames to have good faces
  - When enabled (>0), quality check: Faces must be ≥40px (lips) or ≥80px (face)
- `--groups`: Number of parallel jobs for faster processing
- `--job-index`: Job index for parallel processing (0 to groups-1)



## Citation

If you use MuAViC dataset, please cite the original paper:

```bibtex
@inproceedings{anwar2023muavic,
  title={MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation},
  author={Anwar, Mohamed and Boito, Marcely Zanon and Bougares, Fethi and Nguyen, Ha and Barbosa, Lucas and Berard, Alexandre and Besacier, Laurent},
  booktitle={Proceedings of Interspeech},
  year={2023}
}
```

## References

- **Official MuAViC Repository**: https://github.com/facebookresearch/muavic
- **Paper**: [MuAViC: A Multilingual Audio-Visual Corpus](https://arxiv.org/abs/2303.00628)
- **Auto-AVSR**: RetinaFace preprocessing adapted from Auto-AVSR codebase