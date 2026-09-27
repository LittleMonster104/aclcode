#!/usr/bin/env python
"""
Download and prepare Flickr8K dataset for image-text retrieval
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "flickr8k"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Flickr8K Dataset Preparation")
print("="*80)

# Flickr8K dataset info
# Note: Flickr8K requires manual download from Kaggle
# https://www.kaggle.com/datasets/adityajn105/flickr8k

print("\n[INFO] Flickr8K Dataset Information:")
print("- Images: 8,000")
print("- Captions: 5 per image (40,000 total)")
print("- Train: 6,000 images")
print("- Val: 1,000 images")
print("- Test: 1,000 images")

print("\n[IMPORTANT] Flickr8K requires manual download from Kaggle:")
print("1. Go to: https://www.kaggle.com/datasets/adityajn105/flickr8k")
print("2. Download the dataset")
print("3. Extract to:", DATA_DIR)
print("\nExpected structure:")
print("  data/flickr8k/")
print("    ├── Images/")
print("    ├── captions.txt")
print("    └── (optional) Flickr_8k.trainImages.txt")

# Check if already downloaded
if (DATA_DIR / "Images").exists() and (DATA_DIR / "captions.txt").exists():
    print("\n✅ Flickr8K dataset found!")
    
    # Count images
    num_images = len(list((DATA_DIR / "Images").glob("*.jpg")))
    print(f"   Images: {num_images}")
    
    # Check captions
    with open(DATA_DIR / "captions.txt", 'r') as f:
        num_captions = len(f.readlines()) - 1  # -1 for header
    print(f"   Captions: {num_captions}")
    
else:
    print("\n⚠️  Flickr8K dataset not found!")
    print("\nAlternative: Use Flickr8K from HuggingFace")
    print("Run: pip install datasets")
    print("Then the training script will auto-download from HuggingFace")

print("\n" + "="*80)
