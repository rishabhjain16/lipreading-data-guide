# MaFi to Auto-AVSR Format Converter (Step 1.1)

This script converts processed MaFi dataset to the auto_avsr data format with 4 required fields.

## Requirements

- MaFi dataset already processed by `step1_prepare_mafi.py`
- LRS3 SentencePiece model (`unigram5000.model`)
- Python packages: `pandas`, `sentencepiece`, `tqdm`

## Usage

```bash
python step1.1_mafi_to_autoavsr.py \
    --input-csv /path/to/mafi_processed.csv \
    --output-csv /path/to/mafi_autoavsr.csv \
    --spm-model-path /path/to/LRS3/spm/unigram/unigram5000.model
```

## Example

```bash
python step1.1_mafi_to_autoavsr.py \
    --input-csv /media/rishabhjain/SSD/MaFi_Clean/labels/mafi_lips.csv \
    --output-csv /media/rishabhjain/SSD/MaFi_Clean/mafi_autoavsr.csv \
    --spm-model-path /home/rishabhjain/Desktop/Experiments/lipreading-data-guide/LRS3/spm/unigram/unigram5000.model
```

## Output Format

The output CSV contains exactly 4 columns:

| Column | Description | Example |
|--------|-------------|---------|
| `ID` | Unique video identifier | `F001_hello` |
| `video_path` | Path to processed video | `/path/to/F001_hello.mp4` |
| `audio_path` | Path to corresponding audio | `/path/to/F001_hello.wav` |
| `tokenized` | SentencePiece tokenized text | `123 456 789` |

## Features

- ✅ Extracts transcripts from MaFi filenames (word after speaker ID)
- ✅ Uses LRS3 SentencePiece model for tokenization
- ✅ Generates audio paths from video paths (.mp4 → .wav)
- ✅ Progress tracking with tqdm
- ✅ Detailed logging and error handling
- ✅ Sample output preview

## Notes

- MaFi videos contain single words (e.g., "hello", "goodbye")
- Audio files should exist alongside video files with same basename
- SentencePiece tokenization converts text to space-separated token IDs
- Script skips records with missing transcripts or tokenization failures