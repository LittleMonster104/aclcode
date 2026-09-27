#!/usr/bin/env python
"""
预计算 CIFAR-10 子集的 CLIP 特征并保存
"""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Device: {device}")
print("[INFO] Loading CLIP...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("[OK] CLIP loaded")

# CIFAR-10 classes
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CAPTION_TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a photo of a small {}",
    "a photo of a large {}",
]

# Load CIFAR-10
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

print("[INFO] Loading CIFAR-10...")
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

# 取子集：train 500, test 100
np.random.seed(42)
train_indices = np.random.choice(len(train_dataset), 500, replace=False)
test_indices = np.random.choice(len(test_dataset), 100, replace=False)

print(f"[INFO] Extracting features for {len(train_indices)} train + {len(test_indices)} test samples...")

def extract_features(dataset, indices):
    vis_feats = []
    txt_feats = []
    labels = []
    
    for idx in indices:
        img, label = dataset[idx]
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Image features
        with torch.no_grad():
            vis_feat = clip_model.get_image_features(pixel_values=img_tensor)
        
        # Text features (unique caption per sample)
        template = CAPTION_TEMPLATES[idx % len(CAPTION_TEMPLATES)]
        caption = template.format(CIFAR10_CLASSES[label])
        
        text_inputs = processor.tokenizer(caption, padding=True, truncation=True,
                                          max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            txt_feat = clip_model.get_text_features(**text_inputs)
        
        vis_feats.append(vis_feat.cpu())
        txt_feats.append(txt_feat.cpu())
        labels.append(label)
        
        if (len(vis_feats)) % 50 == 0:
            print(f"  Processed {len(vis_feats)} samples...")
    
    vis_feats = torch.cat(vis_feats, dim=0)  # [N, 512]
    txt_feats = torch.cat(txt_feats, dim=0)  # [N, 512]
    labels = torch.tensor(labels, dtype=torch.long)
    
    return vis_feats, txt_feats, labels

# Extract
train_vis, train_txt, train_labels = extract_features(train_dataset, train_indices)
test_vis, test_txt, test_labels = extract_features(test_dataset, test_indices)

# Save
save_path = DATA_DIR / "cifar10_clip_features_500.pt"
torch.save({
    "train_vis": train_vis,
    "train_txt": train_txt,
    "train_labels": train_labels,
    "test_vis": test_vis,
    "test_txt": test_txt,
    "test_labels": test_labels,
}, save_path)

print(f"\n[OK] Features saved to {save_path}")
print(f"  Train: {train_vis.shape}, Test: {test_vis.shape}")
print(f"  Feature dim: {train_vis.shape[1]}")
