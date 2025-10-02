# WebDataset Converters

Convert preprocessed datasets to WebDataset format for efficient training.

## Installation

```bash
pip install webdataset
```

## Usage

### Auto-AVSR Datasets (LRS2, LRS3, VoxCeleb2, TCD-TIMIT)

```bash
python webData/auto_avsr_to_webD.py \
    --video_root /path/to/dataset_video \
    --text_root /path/to/dataset_text \
    --csv_file /path/to/labels/dataset_train.csv \
    --output_dir /path/to/webdataset \
    --dataset_name lrs2 \
    --samples_per_shard 500
```

### MuAViC Dataset

```bash
python webData/muavic_to_webD.py \
    --video_root /path/to/muavic/muavic_video \
    --text_root /path/to/muavic/muavic_text \
    --csv_file /path/to/muavic/labels/muavic_ar_train.csv \
    --output_dir /path/to/webdataset \
    --dataset_name muavic_ar \
    --samples_per_shard 500
```

## Parameters

- `--video_root`: Directory containing video files
- `--text_root`: Directory containing text files
- `--csv_file`: CSV metadata file from preprocessing
- `--output_dir`: Output directory for WebDataset shards
- `--dataset_name`: Name for output files
- `--samples_per_shard`: Samples per shard (default: 500)

## Output

Creates sharded tar files:
```
output_dir/
├── dataset_train-000000.tar
├── dataset_train-000001.tar
└── ...
```

Each shard contains:
- Video files (MP4 with audio)
- Transcripts
- Metadata (JSON)

## Loading in Training

```python
import webdataset as wds

dataset = wds.WebDataset("/path/to/dataset_train-{000000..000010}.tar")
dataset = dataset.decode("rgb").to_tuple("video", "label", "json")

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## Notes

- Process each split (train/valid/test) separately
- Adjust `samples_per_shard` based on dataset size
- MuAViC script preserves language and timing metadata
