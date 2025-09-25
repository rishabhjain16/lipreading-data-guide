# VoxCeleb2 Dataset
Note: This repository provides a simplified guide for preparing the VoxCeleb2 dataset, including links to a publicly available mirror hosted on Hugging Face. However, users are strongly advised to obtain their own licence for VoxCeleb2 from the official source before using the dataset. This ensures compliance with the original licensing terms and helps avoid any legal or ethical issues related to dataset usage.

This repository provides instructions to download, extract, and organize the **VoxCeleb2** dataset from [Hugging Face](https://huggingface.co/datasets/Reverb/voxceleb2).

The dataset contains:

- **Audio:** Split AAC archives (`aac.7z.001` – `aac.7z.015`)  
- **Video:** Split video parts (`vox2_dev_mp4_partaa` – `vox2_dev_mp4_partai`)  
- **Text/metadata:** Transcripts and metadata (`vox2_dev_txt.zip`, `vox2_test_txt.zip`, `vox2_meta.csv`)  


## Prerequisites

Ensure the following are installed:

```bash
sudo apt update
sudo apt install -y p7zip-full unzip tar git-lfs
git lfs install

```

## Downloading the Dataset 

```bash
git clone https://huggingface.co/datasets/Reverb/voxceleb2
cd voxceleb2
git lfs pull
```

## Extracting Files 

### 1. Audio 

Structure: aac.7z.001, aac.7z.002, ..., aac.7z.015
```bash
7z x aac.7z.001 -oaac/
```

-oaac/ extracts files into the folder aac/


### 1. Video 

Structure: vox2_dev_mp4_partaa, vox2_dev_mp4_partab, ..., vox2_dev_mp4_partai
```bash
cat vox2_dev_mp4_part* > vox2_dev_full.tar
```

Video files are split across multiple parts. Above command will keep all parts in the same folder. Keep the .tar extension since the combined file is a tar archive.


```bash
tar -xf vox2_dev_full.tar -C vox2_dev_full/
```
This will extracts the contents into a folder vox2_dev_full.

### 3. Metadata & Text Files 

vox2_meta.csv – metadata for the dataset
vox2_dev_txt.zip, vox2_test_txt.zip – transcript files for dev/test sets

```bash
unzip vox2_dev_txt.zip -d vox2_dev_txt/
unzip vox2_test_txt.zip -d vox2_test_txt/
```

-oaac/ extracts files into the folder aac/



## Folder Structure After Extraction
```bash
voxceleb2/
├── aac/                   # Extracted audio files
├── vox2_dev_full/          # Extracted videos
├── vox2_dev_txt/           # Transcripts
├── vox2_test_txt/          # Test transcripts
├── vox2_meta.csv           # Metadata CSV
└── README.md
```
