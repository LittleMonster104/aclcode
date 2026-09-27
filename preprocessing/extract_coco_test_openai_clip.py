#!/usr/bin/env python3
"""
使用OpenAI官方CLIP提取COCO Test特征
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import clip

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"设备: {device}")

DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")
OUTPUT_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data")

print("="*80)
print("使用OpenAI官方CLIP提取COCO Test特征")
print("="*80)

# 加载OpenAI CLIP
print("\n步骤1: 加载OpenAI CLIP模型...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("  ✅ OpenAI CLIP ViT-B/32 加载完成")

# 加载Karpathy split
print("\n步骤2: 加载Karpathy split...")
with open(DATA_DIR / "dataset_coco.json", 'r') as f:
    karpathy_data = json.load(f)

test_images = [img for img in karpathy_data['images'] if img['split'] == 'test']
print(f"  Test images: {len(test_images)}")

# 提取特征
print("\n步骤3: 提取Test特征（每张图5条captions）...")
vision_features = []
text_features = []
failed = 0

for img_data in tqdm(test_images, desc="Test"):
    try:
        # 加载图像
        filepath = img_data['filepath']
        filename = img_data['filename']
        img_path = DATA_DIR / filepath / filename
        
        if not img_path.exists():
            failed += 1
            continue
        
        image = Image.open(img_path).convert('RGB')
        
        # 对每条caption
        for sent in img_data['sentences']:
            caption = sent['raw']
            
            # OpenAI CLIP预处理
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_input = clip.tokenize([caption], truncate=True).to(device)
            
            # 提取特征
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_feature = model.encode_text(text_input)
            
            vision_features.append(image_features.cpu().squeeze(0))
            text_features.append(text_feature.cpu().squeeze(0))
    
    except Exception as e:
        failed += 1
        if failed <= 3:
            print(f"\n  警告: {filename} 失败: {e}")
        continue

if failed > 0:
    print(f"\n  处理失败: {failed} 张")

vision_features = torch.stack(vision_features)
text_features = torch.stack(text_features)

print(f"\n  ✅ 提取完成:")
print(f"     Vision: {vision_features.shape}")
print(f"     Text: {text_features.shape}")

# 保存
print("\n步骤4: 保存特征...")
torch.save({
    'vision_features': vision_features,
    'text_features': text_features,
    'num_images': len(test_images),
    'num_texts': len(vision_features)
}, OUTPUT_DIR / "coco_test_openai_clip_features.pt")

print(f"  ✅ 已保存: coco_test_openai_clip_features.pt")

# 评估OpenAI CLIP baseline
print("\n步骤5: 评估OpenAI CLIP baseline...")

vision_features = F.normalize(vision_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

n_images = len(test_images)
image_features = vision_features[::5]

# Image-to-Text
sims = image_features @ text_features.T
ranks = []
for i in range(n_images):
    sim_row = sims[i]
    correct_indices = list(range(i * 5, i * 5 + 5))
    max_correct_sim = sim_row[correct_indices].max().item()
    rank = (sim_row >= max_correct_sim).sum().item() - 1
    ranks.append(rank)

import numpy as np
ranks = np.array(ranks)
r1 = (ranks == 0).mean() * 100
r5 = (ranks < 5).mean() * 100
r10 = (ranks < 10).mean() * 100

print(f"\nOpenAI CLIP (Image→Text):")
print(f"  R@1:  {r1:.2f}%")
print(f"  R@5:  {r5:.2f}%")
print(f"  R@10: {r10:.2f}%")

print(f"\n文献参考值:")
print(f"  R@1:  58.4%")

if abs(r1 - 58.4) < 3:
    print(f"\n✅ 性能正常，与文献一致！")
    print(f"   可以开始训练SHL-CDSA")
elif r1 > 50:
    print(f"\n✅ 性能接近文献（差距{58.4-r1:.1f}%）")
    print(f"   可接受，可以开始训练")
else:
    print(f"\n⚠️  仍然低于预期")

print("\n" + "="*80)
print("完成！")
print("="*80)
