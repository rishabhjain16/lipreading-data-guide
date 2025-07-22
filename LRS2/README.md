# LRS2 Dataset Preparation Guide

## Overview
This project is based on [Auto-AVSR data preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation), however we have simplified the scripts to work with ease. You can follow the original Auto-AVSR approach or use the streamlined scripts we provide here.

### Setup

Install all dependency-packages.

```Shell
pip install -r requirements.txt
pip install torch torchvision torchaudio pytorch-lightning sentencepiece av 
```



### Step 1 - Auto_AVSR style output

```bash
python preparation/step1_prepare_lrs2_all.py \
  --data-dir /path/to/original/lrs2 \
  --root-dir /path/to/output \
  --subset train \
  --dataset lrs2 \
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
- Lips: `lrs2_video_seg16s/`
- Face: `lrs2_video_seg16s_face/`
- Full: `lrs2_video_seg16s_full/`

Process all subsets:
```bash
for subset in train val test; do
  python preparation/step1_prepare_lrs2_all.py \
    --data-dir /path/to/lrs2 \
    --root-dir /path/to/output \
    --subset $subset \
    --dataset lrs2 \
    --detector mediapipe \
    --crop-type lips
done
```


**Note:** The target directory name depends on your chosen segment size:
- `lrs2_video_seg16s` (for 16-second segments)  
- `lrs2_video_seg24s` (for 24-second segments)


### Step 2: Process Metadata
Generate file lists and copy dataset splits in one command:

```bash
python step2_process_metadata.py \
  --lrs2-root /path/to/original/lrs2 \
  --target-dir /path/to/processed/lrs2/lrs2_video_segXXs
```

**Example:**
```bash
python step2_process_metadata.py \
  --lrs2-root /home/rishabh/Desktop/Datasets/lrs2 \
  --target-dir /home/rishabh/Desktop/Datasets/lrs2_ffs/lrs2/lrs2_video_seg16s
```

**Parameters:**
- `--lrs2-root`: Path to your **original** LRS2 dataset (contains lrs2/ folders and text files)
- `--target-dir`: Path to your **processed** segmented video folder from auto_avsr

**Note:** The script will automatically find labels and text files in either:
- The original LRS2 directory (`--lrs2-root`)
- The processed LRS2 directory (parent of `--target-dir`)

**What this does:**
- ✅ Generates `file.list` and `label.list` 
- ✅ Saves them directly to the target directory 
- ✅ Copies dataset split files (`train.txt`, `val.txt`, `test.txt`, `pretrain.txt`)

### Step 3: Metadata Prep  
Count frames and create manifest files in one command:

```bash
python step3_metadata_prep.py \
  --lrs2-data-dir /path/to/processed/lrs2/lrs2_video_segXXs \
  --metadata-dir /path/to/processed/lrs2/metadata \
  --vocab-size 1000
```

**Example:**
```bash
python step3_metadata_prep.py \
  --lrs2-data-dir /home/rishabh/Desktop/Datasets/lrs2_ffs/lrs2/lrs2_video_seg16s \
  --metadata-dir /home/rishabh/Desktop/Datasets/lrs2_ffs/lrs2/metadata \
  --vocab-size 1000
```

**Parameters:**
- `--lrs2-data-dir`: Path to your segmented video folder (contains file.list, label.list, main/, pretrain/)
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
python step2_process_metadata.py --target-dir /path/to/lrs2_video_seg16s_face
python step3_metadata_prep.py --lrs2-data-dir /path/to/lrs2_video_seg16s_face

# For full mode  
python step2_process_metadata.py --target-dir /path/to/lrs2_video_seg16s_full
python step3_metadata_prep.py --lrs2-data-dir /path/to/lrs2_video_seg16s_full
```

---


Your LRS2 dataset is now ready for training. The final processed data will be in:
```
/path/to/processed/lrs2/metadata/
├── train.tsv, valid.tsv, test.tsv
├── train.wrd, valid.wrd, test.wrd
├── dict.wrd.txt
└── spm1000/ (vocabulary files)
```

**Example final structure:**
```
/home/rishabh/Desktop/Datasets/lrs2_ffs/lrs2/
├── lrs2_video_seg16s/           # Segmented videos from auto_avsr
│   ├── main/
│   ├── pretrain/
│   ├── file.list, label.list
│   ├── nframes.audio, nframes.video
│   └── train.txt, val.txt, test.txt, pretrain.txt
└── metadata/                    # Generated manifest files
    ├── train.tsv, valid.tsv, test.tsv
    ├── train.wrd, valid.wrd, test.wrd
    ├── dict.wrd.txt
    └── spm1000/
```

