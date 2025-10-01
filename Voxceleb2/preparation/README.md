# Note: 

1. This repo is orked from: https://github.com/mpc001/auto_avsr/tree/main/preparation.

2. Please refer to previous page Readme Instructions for updated easy to use scripts.


## Pre-processing

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

## Pre-processing VoxCeleb2

To pre-process the VoxCeleb2 dataset, please follow these steps:

1. Download the VoxCeleb2 dataset from the official website.

### Expected VoxCeleb2 Directory Structure

After downloading VoxCeleb2, your directory structure should look like this:

```
voxceleb2/
├── dev/
│   └── mp4/
│       ├── id00012/
│       │   ├── video_folder1/
│       │   │   ├── 00001.mp4
│       │   │   ├── 00002.mp4
│       │   │   └── ...
│       │   └── video_folder2/
│       │       └── ...
│       ├── id00015/
│       └── ...
└── aac/
    ├── id00012/
    │   ├── video_folder1/
    │   │   ├── 00001.m4a
    │   │   ├── 00002.m4a
    │   │   └── ...
    │   └── video_folder2/
    │       └── ...
    ├── id00015/
    └── ...
```

**Key points about the structure**:
- Video files are in `.mp4` format under `dev/mp4/` and `test/mp4/`
- Audio files are in `.m4a` format under `aac/`
- Each speaker has an ID (e.g., `id00012`)
- Each speaker has multiple video folders with segment files
- The `vox-en.id` file contains paths like `dev/id05668/XWKC1JUM_ow/00053`

2. Download pre-computed landmarks below. Once you've finished downloading the five files, simply merge them into one single file using `zip -FF vox2_landmarks.zip --out single.zip`, and then decompress it. If you leave `landmarks-dir` empty, landmarks will be provided with the used of `detector`.

| File Name              | Source URL                                                                        | File Size |
|------------------------|-----------------------------------------------------------------------------------|-----------|
| vox2_landmarks.zip     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.zip)     | 18GB      |
| vox2_landmarks.z01     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z01)     | 20GB      |
| vox2_landmarks.z02     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z02)     | 20GB      |
| vox2_landmarks.z03     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z03)     | 20GB      |
| vox2_landmarks.z04     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z04)     | 20GB      |

3. **Important**: The script expects `.m4a` audio files (not `.wav`). If you have the standard VoxCeleb2 download, the audio files should be in `.m4a` format in the `aac` directory.

4. **Fix the vox-en.id file**: The provided `vox-en.id` file contains paths with `dev/` prefix, but your video directory structure may not include this. You need to remove the `dev/` prefix:

```Shell
# Create a fixed version of vox-en.id
sed 's/^dev\///' vox-en.id > vox-en-fixed.id
mv vox-en.id vox-en-original.id
mv vox-en-fixed.id vox-en.id
```

5. Run the following command to pre-process dataset:

```Shell
python preprocess_vox2.py \
    --vid-dir /path/to/voxceleb2/dev/mp4 \
    --aud-dir /path/to/voxceleb2/aac \
    --label-dir /path/to/preparation/folder \
    --detector retinaface \
    --root-dir /path/to/output \
    --dataset vox2
```

**Example with actual paths:**
```Shell
python preprocess_vox2.py \
    --vid-dir /home/user/voxceleb2/dev/mp4 \
    --aud-dir /home/user/voxceleb2/aac \
    --label-dir /home/user/lipreading-data-guide/Voxceleb2/preparation \
    --detector retinaface \
    --root-dir /media/user/SSD/Data/VC2 \
    --dataset vox2
```

### Arguments
- `vid-dir`: Path to the directory containing video files (should point to the `mp4` subdirectory, e.g., `/path/to/voxceleb2/dev/mp4`).
- `aud-dir`: Path to the directory containing audio files (should point to the `aac` directory, e.g., `/path/to/voxceleb2/aac`). **Note**: Audio files must be in `.m4a` format.
- `label-dir`: Path to the directory containing the `vox-en.id` file (usually the preparation folder). For the label file, we use `vox-en.id` provided by [AVHuBERT repository](https://github.com/facebookresearch/av_hubert/tree/5ab235b3d9dac548055670d534b283b5b70212cc/avhubert/preparation/data).
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used. (Optional)
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of dataset. Default: `vox2`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into. (Optional for parallel processing)
- `job-index`: Job index for the current group and should be an integer within the range of `[0, n)`. (Optional for parallel processing)

This command will preprocess the dataset and store the preprocessed files in the specified `[root_dir]`/`[dataset]`.

### Troubleshooting

**Issue**: Script runs but no output files are created.

**Common causes and solutions**:

1. **Wrong directory structure**: Make sure your `--vid-dir` points to the `mp4` subdirectory (e.g., `/path/to/voxceleb2/dev/mp4`) and `--aud-dir` points to the `aac` directory.

2. **Path mismatch in vox-en.id**: The `vox-en.id` file contains paths with `dev/` prefix. If your video files are in `/path/to/voxceleb2/dev/mp4/`, you need to remove the `dev/` prefix from the ID file as shown in step 4 above.

3. **Audio file format**: The script expects `.m4a` audio files. If you have `.wav` files, you'll need to modify the script or convert your audio files.

4. **Missing files**: Verify that both video and audio files exist for the same IDs:
   ```Shell
   # Check if video file exists
   ls /path/to/voxceleb2/dev/mp4/id05668/XWKC1JUM_ow/00053.mp4
   
   # Check if corresponding audio file exists  
   ls /path/to/voxceleb2/aac/id05668/XWKC1JUM_ow/00053.m4a
   ```

**Expected output**: The script should create a directory structure like:
```
/your/root/dir/vox2/vox2_video_seg24s/
├── id00001/
│   ├── video_folder1/
│   │   ├── 00000.mp4
│   │   ├── 00001.mp4
│   │   └── ...
│   └── ...
└── ...
```

6. Install a pre-trained ASR model, such as [whisper](https://github.com/openai/whisper):

```Shell
pip install -U openai-whisper
```

7. Run the following command to generate transcripts:

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

8. Run the following command to merge labels.

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
