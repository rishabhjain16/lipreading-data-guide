
# Pre-processing

We provide a pre-processing pipeline in this repository for detecting and cropping mouth regions of interest (ROIs) as well as corresponding audio waveforms for LRS2, LRS3, and VoxCeleb2.

## Introduction

Before feeding the raw stream into our model, each video sequence has to undergo a specific pre-processing procedure. This involves three critical steps. The first step is to perform face detection. Following that, each individual frame is aligned to a referenced frame, commonly known as the mean face, in order to normalize rotation and size differences across frames. The final step in the pre-processing module is to crop the mouth region from the aligned mouth image.

<div align="center">

<table style="display: inline-table;">
<tr><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/original.gif", width="144"></td><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/detected.gif" width="144"></td><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/transformed.gif" width="144"></td><td><img src="../doc/cropped.gif" width="144"></td></tr>
<tr><td>0. Original</td> <td>1. Detection</td> <td>2. Transformation</td> <td>3. Mouth ROIs</td> </tr>
</table>
</div>

## Setup

1. Install all dependency-packages.

```Shell
pip install -r requirements.txt
pip install torch torchvision torchaudio pytorch-lightning sentencepiece av 
pip install opencv-python==4.6.0.66
```

2. Install [retinaface](./tools) or [mediapipe](https://pypi.org/project/mediapipe/) tracker.

## Pre-processing LRS2 or LRS3

To pre-process the LRS2 or LRS3 dataset, plrase follow these steps:

1. Download the LRS2 or LRS3 dataset from the official website.

2. Download pre-computed landmarks below. If you leave `landmarks-dir` empty, landmarks will be provided with the used of `detector`.

| File Name              | Source URL                                                                              | File Size  |
|------------------------|-----------------------------------------------------------------------------------------|------------|
| LRS3_landmarks.zip     |[GoogleDrive](https://bit.ly/33rEsax) or [BaiduDrive](https://bit.ly/3rwQSph)(key: mi3c) |     18GB   |
| LRS2_landmarks.zip     |[GoogleDrive](https://bit.ly/3jSMMoz) or [BaiduDrive](https://bit.ly/3BuIwBB)(key: 53rc) |     9GB    |

3. Run the following command to pre-process dataset:

```Shell
python preprocess_lrs2lrs3.py \
    --data-dir [data_dir] \
    --landmarks-dir [landmarks_dir] \
    --detector [detector] \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```

This was the original script, however I have modified it to run it more smoothly:

```bash
# Original scripts
With Precomputed Landmarks:
python preprocess_lrs2_in_lrs3_style.py --data-dir /data/ssd2/data_rishabh/lrs2 --landmarks-dir /data/ssd2/data_rishabh/lrs2/LRS2_landmarks --detector mediapipe --root-dir /data/ssd2/data_rishabh/lrs2/training --dataset lrs2 --subset val

Without Precomputed Landmarks:
python preprocess_lrs2_in_lrs3_style.py --data-dir /data/ssd2/data_rishabh/lrs2 --detector mediapipe --root-dir /data/ssd2/data_rishabh/lrs2/training --dataset lrs2 --subset val
```

## Enhanced Preprocessing with Flexible Cropping

I've created an enhanced version (`preprocess_lrs2_enhanced.py`) that supports different cropping options:

### **Cropping Options:**
- **`lips`** (default): Crops to mouth region only (96x96)
- **`face`**: Crops to full face region (224x224 default)  
- **`full`**: No cropping - keeps full video frames

### **Usage Examples:**

```bash
# Lips cropping (original behavior)
python preprocess_lrs2_enhanced.py \
    --data-dir /home/rishabh/Desktop/Datasets/lrs2 \
    --detector mediapipe \
    --root-dir /home/rishabh/Desktop/Datasets/lrs2_processed \
    --dataset lrs2 \
    --subset train \
    --crop-type lips \
    --crop-size 96 96

# Face cropping 
python preprocess_lrs2_enhanced.py \
    --data-dir /home/rishabh/Desktop/Datasets/lrs2 \
    --detector mediapipe \
    --root-dir /home/rishabh/Desktop/Datasets/lrs2_processed \
    --dataset lrs2 \
    --subset train \
    --crop-type face \
    --crop-size 224 224

# Full video (no cropping)
python preprocess_lrs2_enhanced.py \
    --data-dir /home/rishabh/Desktop/Datasets/lrs2 \
    --detector mediapipe \
    --root-dir /home/rishabh/Desktop/Datasets/lrs2_processed \
    --dataset lrs2 \
    --subset train \
    --crop-type full
```

### **Output Structure:**
The enhanced script creates different output directories based on crop type:

```
lrs2_processed/
├── labels/
│   ├── lrs2_train_transcript_lengths_seg16s.csv          # lips
│   ├── lrs2_train_transcript_lengths_seg16s_face.csv     # face  
│   └── lrs2_train_transcript_lengths_seg16s_full.csv     # full
└── lrs2/
    ├── lrs2_video_seg16s/        # lips (default)
    ├── lrs2_video_seg16s_face/   # face cropping
    ├── lrs2_video_seg16s_full/   # full video
    └── lrs2_text_seg16s/         # text (shared)
```

### Arguments (Enhanced Script)
- `data-dir`: Directory of original dataset.
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of dataset. Valid values are: `lrs2` and `lrs3`.
- `subset`: Subset of dataset. For `lrs2`, the subset can be `train`, `val`, and `test`. For `lrs3`, the subset can be `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `16`.
- `crop-type`: Type of cropping. Valid values are: `lips`, `face`, `full`. Default: `lips`.
- `crop-size`: Crop size [width height]. Default: [96, 96] for lips, [224, 224] for face.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group. Valid values are an integer within the range of `[0, n)`.

### Original Arguments
- `data-dir`: Directory of original dataset.
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of dataset. Valid values are: `lrs2` and `lrs3`.
- `subset`: Subset of dataset. For `lrs2`, the subset can be `train`, `val`, and `test`. For `lrs3`, the subset can be `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group. Valid values are an integer within the range of `[0, n)`.

### Benefits of Enhanced Version:
- **🎯 Flexible Cropping**: Choose between lips, face, or full video
- **📐 Custom Sizes**: Specify crop dimensions  
- **📁 Organized Output**: Different directories for different crop types
- **🔄 Backward Compatible**: Default behavior matches original script
- **⚡ Efficient**: Reuses existing landmark detection and alignment

3. Run the following command to merge labels:

```Shell
python merge.py \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n]
```

### Arguments
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of dataset. Valid values are: `lrs2` and `lrs3`.
- `subset`: Subset of the dataset. For LRS2, valid values are `train`, `val`, and `test`. For LRS3, valid values are `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.

## Pre-processing VoxCeleb2

To pre-process the VoxCeleb2 dataset, please follow these steps:

1. Download the VoxCeleb2 dataset from the official website.

2. Download pre-computed landmarks below. Once you've finished downloading the five files, simply merge them into one single file using `zip -FF vox2_landmarks.zip --out single.zip`, and then decompress it. If you leave `landmarks-dir` empty, landmarks will be provided with the used of `detector`.

| File Name              | Source URL                                                                        | File Size |
|------------------------|-----------------------------------------------------------------------------------|-----------|
| vox2_landmarks.zip     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.zip)     | 18GB      |
| vox2_landmarks.z01     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z01)     | 20GB      |
| vox2_landmarks.z02     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z02)     | 20GB      |
| vox2_landmarks.z03     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z03)     | 20GB      |
| vox2_landmarks.z04     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z04)     | 20GB      |

3. Run the following command to pre-process dataset:

```Shell
python preprocess_vox2.py \
    --vid-dir [vid_dir] \
    --aud-dir [aud_dir] \
    --label-dir [label_dir] \
    --landmarks-dir [landmarks_dir] \
    --detector [detector] \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```

### Arguments
- `vid-dir`: Path to the directory containing video files.
- `aud-dir`: Path to the directory containing audio files.
- `label-dir`: Path to the directory containing language-identification label files. Default: ``. For the label file, we use `vox-en.id` provided by [AVHuBERT repository](https://github.com/facebookresearch/av_hubert/tree/5ab235b3d9dac548055670d534b283b5b70212cc/avhubert/preparation/data).
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of dataset. Default: `vox2`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group and should be an integer within the range of `[0, n)`.

This command will preprocess the dataset and store the preprocessed files in the specified `[root_dir]`/`[dataset]`.

4. Install a pre-trained asr model, such as [whisper](https://github.com/openai/whisper).

5. Run the following command to generate transcripts:

```Shell
python asr_infer.py \
    --root-dir [root-dir] \
    --dataset [dataset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```

### Arguments
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of dataset. Valid value is: `vox2`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups the dataset was split into during preprocessing.
- `job-index`: Job index for the current group.

6. Run the following command to merge labels. (Same as the merge solution at [preprocessing-lrs2-or-lrs3](#preprocessing-lrs2-or-lrs3))

```Shell
python merge.py \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n]
```

### Arguments
- `root-dir`: Root directory of preprocessed dataset.
- `dataset`: Name of the dataset. Valid value is: `vox2`
- `subset`: The subset name of the dataset. For `vox2`, valid value is `train`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.
