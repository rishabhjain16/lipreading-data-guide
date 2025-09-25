# AVCocktail Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-AVCocktail-blue)](https://huggingface.co/datasets/nguyenvulebinh/AVCocktail)

This repository provides instructions for accessing and using the **AVCocktail** audiovisual dataset.

---

## Dataset Overview

The **AVCocktail** dataset is an audiovisual dataset hosted on [Hugging Face](https://huggingface.co/datasets/nguyenvulebinh/AVCocktail). It contains audiovisual data for research and experimentation.

---

## Method 1: Using Hugging Face `datasets` Library

### Installation

Make sure Python is installed, then install the `datasets` library:

```bash
pip install datasets
```

### Loading the Dataset
```bash
from datasets import load_dataset

# Load the AVCocktail dataset
dataset = load_dataset("nguyenvulebinh/AVCocktail")

# Example: Access the training split
train_data = dataset["train"]
print(train_data)
```

## Method 2:  Download via Git LFS

### Install Git LFS

Make sure Python is installed, then install the `datasets` library:

```bash
sudo apt-get install git-lfs   # Debian/Ubuntu
git lfs install
```

### Loading the Dataset
```bash
git clone https://huggingface.co/datasets/nguyenvulebinh/AVCocktail
cd AVCocktail
git lfs pull

```


## References:
1. AVCocktail Dataset: https://huggingface.co/datasets/nguyenvulebinh/AVCocktail
2. AVCocktail Github: https://github.com/nguyenvulebinh/AVSRCocktail
3. AVCocktail HF: https://huggingface.co/nguyenvulebinh/AVSRCocktail