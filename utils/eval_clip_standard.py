#!/usr/bin/env python3
"""
标准CLIP评估脚本（参考ITRA实现）
目标：复现CLIP ViT-B/32在COCO上50%的性能
"""
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

device = "cpu"  # 改为mps如果要用GPU
print(f"设备: {device}")

print("="*80)
print("标准CLIP评估 - 目标：I2T R@1 ≈ 50%")
print("="*80)

# 加载CLIP
print("\n加载CLIP ViT-B/32...")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()
print(f"  模型输入分辨率: {model.visual.input_resolution}")

# 加载数据
DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")

print("\n加载Karpathy split...")
with open(DATA_DIR / "dataset_coco.json") as f:
    data = json.load(f)

test_images = [img for img in data['images'] if img['split'] == 'test']
print(f"  Test images: {len(test_images)}")

# 提取特征（正确方式）
print("\n提取特征...")

image_features_list = []
text_features_list = []
image_ids = []

for img_data in tqdm(test_images, desc="处理图像"):  # 全部5000张
    try:
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            continue
        
        # 加载并预处理图像
        image = Image.open(img_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # 提取图像特征（每张图只提取一次）
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = F.normalize(image_feature, dim=-1).cpu()
        
        # 提取该图的所有caption特征
        captions = [sent['raw'] for sent in img_data['sentences']]
        text_inputs = clip.tokenize(captions, truncate=True).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=-1).cpu()
        
        # 存储
        image_features_list.append(image_feature.squeeze(0))
        text_features_list.extend([text_features[i] for i in range(len(captions))])
        image_ids.extend([len(image_features_list)-1] * len(captions))
    
    except Exception as e:
        continue

# 转为tensor
image_features = torch.stack(image_features_list)  # [N_images, 512]
text_features = torch.stack(text_features_list)    # [N_texts, 512]
image_ids = torch.tensor(image_ids)                # [N_texts]

n_images = len(image_features)
n_texts = len(text_features)

print(f"\n数据统计:")
print(f"  Images: {n_images}")
print(f"  Texts: {n_texts}")
print(f"  平均每图captions: {n_texts / n_images:.1f}")

# 评估 - Image to Text
print("\n" + "="*80)
print("Image → Text Retrieval")
print("="*80)

sims = image_features @ text_features.T  # [N_images, N_texts]

ranks = []
for i in range(n_images):
    # 找到属于第i张图的所有text indices
    correct_text_indices = (image_ids == i).nonzero(as_tuple=True)[0]
    
    # 计算rank
    sim_row = sims[i]
    max_correct_sim = sim_row[correct_text_indices].max().item()
    rank = (sim_row >= max_correct_sim).sum().item() - 1
    ranks.append(rank)

ranks = np.array(ranks)
i2t_r1 = (ranks == 0).mean() * 100
i2t_r5 = (ranks < 5).mean() * 100
i2t_r10 = (ranks < 10).mean() * 100

print(f"\nImage → Text:")
print(f"  R@1:  {i2t_r1:.2f}%")
print(f"  R@5:  {i2t_r5:.2f}%")
print(f"  R@10: {i2t_r10:.2f}%")

# 评估 - Text to Image
print("\n" + "="*80)
print("Text → Image Retrieval")
print("="*80)

sims_t2i = sims.T  # [N_texts, N_images]

ranks_t2i = []
for txt_idx in range(n_texts):
    correct_img_idx = image_ids[txt_idx].item()
    sim_row = sims_t2i[txt_idx]
    correct_sim = sim_row[correct_img_idx].item()
    rank = (sim_row >= correct_sim).sum().item() - 1
    ranks_t2i.append(rank)

ranks_t2i = np.array(ranks_t2i)
t2i_r1 = (ranks_t2i == 0).mean() * 100
t2i_r5 = (ranks_t2i < 5).mean() * 100
t2i_r10 = (ranks_t2i < 10).mean() * 100

print(f"\nText → Image:")
print(f"  R@1:  {t2i_r1:.2f}%")
print(f"  R@5:  {t2i_r5:.2f}%")
print(f"  R@10: {t2i_r10:.2f}%")

# 对比期望值
print("\n" + "="*80)
print("对比期望值")
print("="*80)

print(f"\n期望 (CLIP ViT-B/32 on COCO 5K):")
print(f"  I2T R@1: 50-56%")
print(f"  T2I R@1: 35-42%")

print(f"\n实际:")
print(f"  I2T R@1: {i2t_r1:.2f}%")
print(f"  T2I R@1: {t2i_r1:.2f}%")

if i2t_r1 > 45:
    print(f"\n✅ Baseline正常！")
elif i2t_r1 > 35:
    print(f"\n⚠️  接近但偏低，检查预处理")
else:
    print(f"\n❌ Baseline异常低，检查实现")
    print(f"\n可能的问题:")
    print(f"  1. 图像预处理不对")
    print(f"  2. 评估代码有bug")
    print(f"  3. CLIP权重未正确加载")

print("="*80)
