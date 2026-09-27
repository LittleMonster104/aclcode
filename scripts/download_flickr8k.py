"""
Download Flickr8K as a lighter-weight alternative (~2GB vs ~16GB).
Uses torchvision built-in loader which handles everything automatically.
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "flickr8k"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=== Downloading Flickr8K via torchvision ===")
print(f"Data directory: {DATA_DIR}")

try:
    from torchvision.datasets import Flickr8k
    from torchvision.transforms import Compose, ToTensor
    
    print("Downloading images and annotations...")
    dataset = Flickr8k(
        root=str(DATA_DIR),
        transform=Compose([ToTensor()]),
        download=True
    )
    
     # Flickr8k doesn't have a __len__ in older torchvision, check attributes
    has_len = hasattr(dataset, "__len__")
    img_dir = DATA_DIR / "images"
    
    print(f"Dataset loaded: {len(dataset) if has_len else 'unknown'} samples")
    print(f"Images directory: {img_dir}")
    print(f"Annotations files:")
    for f in DATA_DIR.glob("*.txt"):
        size = f.stat().st_size
        print(f"  {f.name} ({size} bytes)")
        
except ImportError as e:
    print(f"torchvision Flickr8k not available: {e}")
    print("Flickr8K was added in torchvision 0.16+")
    
except Exception as e:
    print(f"Download failed: {e}")
    import traceback
    traceback.print_exc()
