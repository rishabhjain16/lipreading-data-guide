# RoomReader Dataset Preparation Guide

### Add ons: 
Might consider adding a way to concetenate smaller chunks into larger ones

## Overview

The RoomReader dataset is a **multimodal corpus of online multiparty conversational interactions** containing 30 tutorial sessions with 118 unique participants. This preprocessing pipeline adapts the RoomReader dataset for lip reading and visual speech recognition tasks, following the same 3-step approach used for other datasets in this repository.

**Original Paper**: Reverdy, Justine, et al. (2022). "RoomReader: A Multimodal Corpus of Online Multiparty Conversational Interactions." In Proceedings of LREC 2022.

## Dataset Characteristics

- **Sessions**: 30 tutorial sessions (S01-S30)
- **Participants**: 118 unique participants
  - 65 females, 51 males, 2 others
  - 91 native, 27 non-native English speakers
  - All university students + recent graduates
- **Videos**: 322 total video files (~98.2 GB)
- **Audio**: Individual and group audio tracks (~11 GB)
- **Transcriptions**: 267 manually corrected transcriptions
- **Duration**: Variable session lengths (typically 10-15 minutes)

## Original Dataset Structure

```
RoomReader/
├── video/                                    # Video files (~98.2 GB)
│   ├── all_participants/                     # Grid view videos (30 sessions)
│   │   └── S01.mp4, S02.mp4, ...            # All participants in grid
│   └── individual_participants/
│       ├── individual_participants_audio_all/     # Individual videos + group audio
│       └── individual_participants_individual_audio/  # Individual videos + individual audio
│           ├── S01/
│           │   ├── S01_P011_Chrisian.mp4    # Participant videos
│           │   ├── S01_P017_Julian.mp4
│           │   ├── S01_T002_Charlie.mp4     # Tutor video
│           │   └── ...
│           ├── S02/, S03/, ..., S30/
├── audio/                                    # Audio files (~11 GB)
│   ├── all_participants_audio/              # Group audio per session
│   │   └── S01.wav, S02.wav, ...
│   └── individual_participants_audio/       # Individual audio tracks
│       ├── S01/
│       │   ├── S01_P011_Christian.wav
│       │   ├── S01_T002_Charlie.wav
│       │   └── ...
├── annotations/                              # Transcriptions & annotations
│   ├── transcriptions_corrected/            # Manual corrections
│   │   ├── textgrid_individual/             # Individual TextGrid files
│   │   │   ├── S01_P011_Christian_corrected.TextGrid
│   │   │   └── ...
│   │   ├── textgrids_individual_word_boundaries/  # Word-level boundaries
│   │   └── textgrids_session_level/         # Session-level annotations
│   ├── transcriptions_asr/                  # Original ASR output
│   ├── transcriptions_txt/                  # Plain text format
│   │   ├── S01.txt, S02.txt, ...           # Per-session transcripts
│   ├── elan_sessions/                       # ELAN annotation files
│   └── continuous engagement/               # Engagement annotations
├── features/                                 # OpenFace facial features (~40.6 GB)
├── metrics/                                  # Post-recording questionnaires
├── documentation/                            # Participant info, consent, personality tests
└── README_RoomReader.txt                     # Original documentation
```

## Naming Conventions

### Participant IDs
- **S01-S30**: Session identifiers (30 sessions)
- **P###**: Student participants (e.g., P011, P025)
- **T###**: Tutor participants (T001, T002)
- **Names**: Pseudonymized names for privacy

### File Naming
- **Videos**: `SessionID_ParticipantID_Name.mp4` (e.g., `S01_P011_Christian.mp4`)
- **Audio**: `SessionID_ParticipantID_Name.wav` (e.g., `S01_P011_Christian.wav`)
- **Transcripts**: `SessionID_ParticipantID_Name_corrected.TextGrid`

### Session Structure
Each session contains:
- **Introduction** (1-2 minutes)
- **Questions 1-3** with rankings (8-10 minutes total)
- **Closing** (30-60 seconds)

## Privacy & Consent Notice

⚠️ **Important**: Not all participants consented to image/video publication usage.

**Sessions with full consent for image/video publication**: S05, S06, S11, S15, S16, S19, S23, S29

For other sessions, refer to the Pre-Recording Questionnaire in the documentation folder for specific usage permissions.

## Preprocessing Pipeline

The RoomReader preprocessing follows a complete 2-step pipeline that processes the dataset into lip reading ready format:

### Step 1: Complete Video Preprocessing with Face Detection

Processes the RoomReader dataset to create lip/face cropped video segments with audio and text, following the same approach as LRS2/LRS3 datasets.

**Key Features:**
1. **RetinaFace Detection**: Uses 68-point facial landmarks for precise face detection
2. **Intelligent Cropping**: 
   - `lips`: Extracts mouth region → 96x96 output
   - `face`: Extracts full face → 224x224 output
3. **Audio Synchronization**: Extracts audio segments matching video timing
4. **Privacy Protection**: Creates anonymous speaker IDs (spk0, spk1, etc.)

#### Usage

```bash
# Basic usage - Individual mode with lip cropping (96x96)
python preparation/step1_prepare_roomreader.py \
    --data-path /media/rishabhjain/SSD/RoomReader/ \
    --output-path /media/rishabhjain/SSD/RR_processed \
    --video-mode individual \
    --crop-type lips \
    --detector retinaface

# Face cropping (224x224) for full face context
python preparation/step1_prepare_roomreader.py \
    --data-path /media/rishabhjain/SSD/RoomReader/ \
    --output-path /media/rishabhjain/SSD/RR_processed \
    --video-mode individual \
    --crop-type face \
    --detector retinaface

# Conversational mode (noisy audio with background speakers)
python preparation/step1_prepare_roomreader.py \
    --data-path /media/rishabhjain/SSD/RoomReader/ \
    --output-path /media/rishabhjain/SSD/RR_processed \
    --video-mode conversational \
    --crop-type lips \
    --detector retinaface
```

#### Arguments

- `--data-path`: Path to RoomReader dataset root directory
- `--output-path`: Output directory for processed files
- `--video-mode`: 
  - `individual`: Clean audio (speaker talks alone)
  - `conversational`: Noisy audio (speaker talks with others in background)
- `--crop-type`:
  - `lips`: Mouth region cropping (96x96) for lip reading
  - `face`: Full face cropping (224x224) for facial expression analysis
- `--detector`: Face detector to use (`retinaface` recommended)

#### Processing Statistics

The script generates a CSV file with detailed metadata including:
- Speaker IDs and session information
- Video/audio/text file paths
- Original participant names → anonymous mappings
- Word counts and duration statistics
- Processing parameters (detector, crop type, resolution)

#### Technical Details

**Face Detection Quality:**
- Uses RetinaFace for precise 68-point facial landmarks
- Significantly better than simple bounding box detection
- Enables accurate lip region extraction even in challenging online video conditions

**Temporal Consistency:**
- Smooths crop regions between frames to avoid jitter
- Maintains stable lip/face position throughout utterance
- Handles variable video quality from online recordings

**Text Preprocessing:**
- Removes RoomReader-specific disfluency markers ($ and # symbols)
- Strips punctuation and special characters (: , . ! ? ; - ' " ( ) [ ] { } etc.)
- Converts to lowercase for consistency
- Normalizes whitespace and removes empty utterances

**Data Modes:**
- `individual`: Participant speaks alone (clean audio for training)
- `conversational`: Participant speaks with others present (realistic noisy conditions)


**Data Modes:**
- `individual`: Participant speaks alone (clean audio for training)
- `conversational`: Participant speaks with others present (realistic noisy conditions)


### Audio Processing
- **Sample Rate**: 48kHz (will be downsampled to 16kHz)
- **Channels**: Mono (individual tracks) or multi-channel (group)
- **Format**: WAV files
- **Bitrate**: Variable quality due to online recording conditions

### Transcription Features
- **Level**: Utterance and word-level timestamps
- **Accuracy**: Manually corrected ASR transcriptions
- **Annotations**: Disfluencies, feedback, paralinguistic elements
- **Format**: TextGrid, ELAN, and plain text
- **Text Cleaning**: Automatic removal of disfluency markers ($, #) and punctuation for clean training data

## Getting Started

### Prerequisites
```bash
# Install basic dependencies
pip install torch torchvision torchaudio
pip install opencv-python pandas tqdm

# Install RetinaFace dependencies for face detection
pip install ibug-face_detection ibug-face_alignment

# Optional: For future TextGrid processing
pip install textgrid
```

### Quick Start

```bash
# Navigate to preparation directory
cd preparation/

# Process with lip cropping (recommended for lip reading)
python step1_prepare_roomreader.py \
  --data-path /path/to/RoomReader \
  --output-path /path/to/output \
  --video-mode individual \
  --crop-type lips \
  --detector retinaface

# Check output
ls /path/to/output/roomreader_video_lips/individual/S01/
# Expected: S01_spk0_001.mp4, S01_spk0_001.wav, S01_spk0_001.txt, ...
```

### Example Workflow

1. **Setup Environment**:
   ```bash
   conda create -n roomreader python=3.8
   conda activate roomreader
   pip install torch torchvision torchaudio ibug-face_detection ibug-face_alignment
   ```

2. **Test Processing** (small subset):
   ```bash
   # Process just individual mode first
   python step1_prepare_roomreader.py \
     --data-path /media/user/SSD/RoomReader/ \
     --output-path ./test_output \
     --video-mode individual \
     --crop-type lips
   ```

3. **Full Processing**:
   ```bash
   # Process both modes if needed
   python step1_prepare_roomreader.py \
     --data-path /media/user/SSD/RoomReader/ \
     --output-path ./roomreader_processed \
     --video-mode individual \
     --crop-type lips
   
   python step1_prepare_roomreader.py \
     --data-path /media/user/SSD/RoomReader/ \
     --output-path ./roomreader_processed \
     --video-mode conversational \
     --crop-type lips
   
   # Generate separate training manifests for each mode
   python step2.py \
     --roomreader-data-dir ./roomreader_processed/roomreader_video \
     --metadata-dir ./roomreader_processed/metadata \
     --split-ratios 0.7,0.15,0.15 \
     --create-mode-splits
   ```

4. **Verify Output**:
   ```bash
   # Check CSV metadata
   cat ./roomreader_processed/labels/roomreader_individual.csv
   
   # Count processed files
   find ./roomreader_processed -name "*.mp4" | wc -l
   ```

## Citation


## Complete Pipeline

### Step 1: Data Preparation (`step1_prepare_roomreader.py`)
Processes raw RoomReader videos using RetinaFace for face detection and creates individual participant segments with proper audio/video synchronization.

**Features:**
- RetinaFace-based face detection with 68-point landmarks
- Lip region cropping (96x96) and face cropping (224x224)
- Audio/video synchronization with correct FPS handling
- Individual participant video extraction with timestamps
- CSV metadata generation in `labels/` folder

**Usage:**
```bash
python step1_prepare_roomreader.py \
  --data-path /path/to/roomreader \
  --output-path /path/to/output \
  --video-mode individual \
  --crop-type lips \
  --detector retinaface
```

### Step 2: Training Manifest Generation (`step2.py`)
Creates manifests (.tsv and .wrd files) for use with standard lip reading training frameworks.

**Tokenization (shared SPM):**

In addition to the `.tsv`/`.wrd` manifests, Step 2 also writes tokenized text outputs using the
repo-wide SentencePiece model:

- Model: `spm/unigram/unigram5000.model`
- Units: `spm/unigram/unigram5000_units.txt`

The shared model vocabulary is **uppercase**, so transcripts are normalized to uppercase before encoding.

Step 2 writes these files next to each manifest folder it creates:

- `tokens.txt`: one space-separated SentencePiece id sequence per utterance
- `label.csv`: one line per utterance (no header) in the format:
  `roomreader,<abs_video_path>,<space-separated-token-ids>`
- `roomreader_test_transcript_lengths_seg16s.csv`: Auto-AVSR CSV (no header, 4 columns) in the format:
  `roomreader,<abs_video_path>,<input_length>,<space-separated-token-ids>`

**Default Behavior (Test-Only Manifests):**
- Creates three metadata folders: `conversational/`, `individual/`, and `combined/`
- Each folder contains `test.tsv` and `test.wrd` files
- Each folder also contains `tokens.txt` and `label.csv`
- All data treated as test data (no train/val splits)
- TSV format: `id, video_path, audio_path, num_video_frames, num_audio_frames`

**With Train/Val/Test Splits:**
- Use `--split-ratios` to create train/valid/test splits
- Use `--create-mode-splits` to separate conversational and individual modes
- Speaker-based or random data splitting

**Usage:**
```bash
# Default: Create test-only manifests for all three modes
python step2.py \
  --roomreader-data-dir /path/to/processed/data \
  --metadata-dir /path/to/manifests

# With splits: Create train/val/test splits
python step2.py \
  --roomreader-data-dir /path/to/processed/data \
  --metadata-dir /path/to/manifests \
  --split-ratios 0.7,0.15,0.15 \
  --create-mode-splits

# Random split (all data combined)
python step2.py \
  --roomreader-data-dir /path/to/processed/data \
  --metadata-dir /path/to/manifests \
  --split-ratios 0.7,0.15,0.15 \
  --random-split
```

**Arguments:**
- `--split-ratios`: Train/validation/test ratios (optional, e.g., 0.7,0.15,0.15)
- `--create-mode-splits`: Creates separate metadata folders for conversational and individual modes (requires --split-ratios)
- `--random-split`: Use random splitting instead of speaker-based splitting
- `--seed`: Random seed for reproducible splits (default: 42)

## Output Format

The pipeline generates LRS-compatible files with two modes of operation:

**Default (Test-Only):**
- Three metadata folders: `conversational/`, `individual/`, `combined/`
- Each contains `test.tsv` and `test.wrd` files
- All data treated as test data

**With Splits (Optional):**
- **Manifest files (.tsv)**: Tab-separated values with file paths, frame counts, and audio information
- **Word files (.wrd)**: Plain text transcriptions corresponding to each video segment
- **Token files**: `tokens.txt` and `label.csv` are written alongside each split's manifests
- **Directory structure**: Organized by session and speaker for easy navigation
- **Mode separation**: Individual vs conversational audio conditions for targeted training

### Training Scenarios

**Individual Mode**: Clean audio conditions
- Single speaker talking alone
- Minimal background noise
- Ideal for initial model training
- Better for learning visual-audio correspondences

**Conversational Mode**: Realistic noisy conditions  
- Multiple speakers present
- Background conversations
- More challenging evaluation scenario
- Better for robust model testing

## Complete Output Structure

After running the full pipeline, your output directory will have the following structure:

```
output_path/
├── roomreader_video/
│   ├── individual/                         # Clean audio (speaker alone)
│   │   ├── S01/
│   │   │   ├── S01_spk0_001.mp4          # 96x96 lip crops (or 224x224 face)
│   │   │   ├── S01_spk0_001.wav          # 16kHz clean audio segments
│   │   │   ├── S01_spk0_001.txt          # Clean text transcripts
│   │   │   └── ...
│   │   └── S02/, S03/, ..., S30/         # All 30 sessions
│   └── conversational/                     # Noisy audio (multiple speakers)
│       ├── S01/
│       │   ├── S01_spk0_001.mp4          # Same video crops
│       │   ├── S01_spk0_001.wav          # 16kHz noisy audio (background speakers)
│       │   ├── S01_spk0_001.txt          # Same transcripts
│       │   └── ...
│       └── S02/, S03/, ..., S30/
│
├── labels/
│   ├── roomreader_individual.csv          # Individual mode metadata
│   └── roomreader_conversational.csv      # Conversational mode metadata
│
└── manifests/                             # Training manifests (from step2)
    ├── conversational/                    # Conversational mode (default: test-only)
    │   ├── test.tsv, test.wrd
    ├── individual/                        # Individual mode (default: test-only)
    │   ├── test.tsv, test.wrd
    ├── combined/                          # Both modes (default: test-only)
    │   ├── test.tsv, test.wrd
    ├── metadata_individual/               # Individual mode (with --split-ratios --create-mode-splits)
    │   ├── train.tsv, valid.tsv, test.tsv
    │   └── train.wrd, valid.wrd, test.wrd
    ├── metadata_conversational/           # Conversational mode (with --split-ratios --create-mode-splits)
    │   ├── train.tsv, valid.tsv, test.tsv
    │   └── train.wrd, valid.wrd, test.wrd
    └── metadata/                          # Combined manifests (with --split-ratios, no --create-mode-splits)
        ├── train.tsv, valid.tsv, test.tsv
        └── train.wrd, valid.wrd, test.wrd
```

### File Naming Convention

- **Video files**: `SessionID_speakerID_segmentID.mp4` (e.g., `S01_spk0_001.mp4`)
- **Audio files**: `SessionID_speakerID_segmentID.wav` (e.g., `S01_spk0_001.wav`)
- **Text files**: `SessionID_speakerID_segmentID.txt` (e.g., `S01_spk0_001.txt`)
- **Speaker IDs**: Anonymous IDs (`spk0`, `spk1`, `spk2`, etc.) preserve privacy

### Manifest File Format

**train.tsv example:**
```
/
S01_spk0_001    /path/to/S01_spk0_001.mp4    /path/to/S01_spk0_001.wav    56    14933
S01_spk0_002    /path/to/S01_spk0_002.mp4    /path/to/S01_spk0_002.wav    73    19466
...
```

**train.wrd example:**
```
hello how are you
i think that's a good idea
yes i agree with that
...
```

If you use the RoomReader dataset, please cite the original paper:

```bibtex
@inproceedings{reverdy2022roomreader,
  title={RoomReader: A Multimodal Corpus of Online Multiparty Conversational Interactions},
  author={Reverdy, Justine and O'Connor, Sam Russell and Duquenne, Louise and Garaialde, Diego and Cowan, Benjamin R and Harte, Naomi},
  booktitle={Proceedings of the 13th International Conference on Language Resources and Evaluation (LREC 2022)},
  pages={},
  year={2022},
  address={Marseille, France}
}
```
