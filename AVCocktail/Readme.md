# AVCocktail & AVYT Datasets

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-AVCocktail-blue)](https://huggingface.co/datasets/nguyenvulebinh/AVCocktail)

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-AVYT-blue)](https://huggingface.co/datasets/nguyenvulebinh/AVYT)

This repository provides access to two audiovisual datasets: **AVSRCocktail** and **AVYT**. Below are instructions on how to download and utilize these datasets for your projects.


---

## 📦 Dataset Overview

### 1. **AVSRCocktail**

- **Hosted on**: [Hugging Face](https://huggingface.co/datasets/nguyenvulebinh/AVSRCocktail)
- **Modality**: Audiovisual
- **Subsets**:
  - `video_0` to `video_50`: Individual video sessions
- **Splits**:
  - `asd_chunk`
  - `fixed_chunk`
  - `gold_chunk`

### 2. **AVYT**

- **Hosted on**: [Hugging Face](https://huggingface.co/datasets/nguyenvulebinh/AVYT)
- **Modality**: AudioVisual
- **Subsets**:
  - `avyt`: 
    - Splits: talking, silent
  - `avyt-mix`: 
    - Splits: train, test
  - `lrs2`:
    - Splits: train, valid, pretrain, test: [test_snr_{0/5/10/n5}\_interferer\_{1/2}
  - `vox2`: 
    - Splits: dev, test
---

## Method 1: Using Hugging Face `datasets` Library

### Installation

Make sure Python is installed, then install the `datasets` library:

```bash
pip install datasets
```

### Loading the AVCocktail Dataset
```bash
from datasets import load_dataset

# Load the AVCocktail dataset
dataset = load_dataset("nguyenvulebinh/AVCocktail")

# Example: Access the training split
train_data = dataset["train"]
print(train_data)
```
### Loading the AVCocktail Dataset
```bash
from datasets import load_dataset

# Load AVYT dataset
avyt = load_dataset("nguyenvulebinh/AVYT")

# Accessing splits
talking_data = avyt["talking"]
silent_data = avyt["silent"]
```



## Method 2:  Download via Git LFS

### Install Git LFS

Make sure Python is installed, then install the `datasets` library:

```bash
sudo apt-get install git-lfs   # Debian/Ubuntu
git lfs install
```

### Clone the AVCocktail Dataset
```bash
git clone https://huggingface.co/datasets/nguyenvulebinh/AVCocktail
cd AVCocktail
git lfs pull

```

### Clone AVYT dataset
```bash
git clone https://huggingface.co/datasets/nguyenvulebinh/AVYT
cd AVYT
git lfs pull
```

## References:
1. AVCocktail Dataset: https://huggingface.co/datasets/nguyenvulebinh/AVCocktail
2. AVCocktail Github: https://github.com/nguyenvulebinh/AVSRCocktail
3. AVCocktail HF: https://huggingface.co/nguyenvulebinh/AVSRCocktail
4. AVYT Dataset: https://huggingface.co/datasets/nguyenvulebinh/AVYT