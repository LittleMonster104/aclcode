#!/usr/bin/env python3
"""
诊断CLIP baseline问题
目标: 找出为什么我们的CLIP 50.1% << 论文58.4%
"""
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import json
from PIL import Image
import numpy as np
from tqdm import tqdm

print("="*70)
print("CLIP Baseline诊断")
print("="*70)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n[INFO] Device: {device}")

workspace = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace")
coco_dir = workspace / "data" / "coco"

# 1. 检查现有特征
print("\n[CHECK 1] 检查现有COCO特征...")
cache_file = workspace / "data" / "mscoco_1k_clip_features.pt"
data = torch.load(cache_file)
img_feats_old = data['image_features']
txt_feats_old = data['text_features']

print(f"  图像特征: {img_feats_old.shape}")
print(f"  文本特征: {txt_feats_old.shape}")
print(f"  特征范围: [{img_feats_old.min():.3f}, {img_feats_old.max():.3f}]")
print(f"  特征均值: {img_feats_old.mean():.3f}")
print(f"  特征标准差: {img_feats_old.std():.3f}")

# 检查是否归一化
img_norms = torch.norm(img_feats_old, dim=1)
print(f"  图像特征L2范数: min={img_norms.min():.3f}, max={img_norms.max():.3f}, mean={img_norms.mean():.3f}")

# 2. 重新提取特征（使用官方方法）
print("\n[CHECK 2] 使用官方CLIP重新提取特征...")
print("  加载CLIP模型...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
clip_model.eval()

print("  ✓ CLIP加载完成")
print(f"  CLIP vision output dim: {clip_model.config.vision_config.hidden_size}")
print(f"  CLIP text output dim: {clip_model.config.text_config.hidden_size}")
print(f"  CLIP projection dim: {clip_model.config.projection_dim}")

# 加载Karpathy split
with open(workspace / "data" / "dataset_coco.json", 'r') as f:
    karpathy_data = json.load(f)

test_imgs = [img for img in karpathy_data['images'] if img['split'] == 'test']
print(f"\n  测试集: {len(test_imgs)}张图")

# 加载captions
with open(coco_dir / "annotations" / "captions_val2014.json", 'r') as f:
    captions_data = json.load(f)

img_to_captions = {}
for ann in captions_data['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_captions:
        img_to_captions[img_id] = []
    img_to_captions[img_id].append(ann['caption'])

# 重新提取前100个样本的特征进行对比
print("\n  重新提取前100张图的特征...")
img_feats_new = []
txt_feats_new = []

for i, img_info in enumerate(tqdm(test_imgs[:100], desc="提取特征")):
    try:
        # 加载图像
        img_path = coco_dir / "val2014" / img_info['filename']
        image = Image.open(img_path).convert('RGB')
        
        # 获取图像ID
        if 'cocoid' in img_info:
            img_id = img_info['cocoid']
        else:
            img_id = int(img_info['filename'].split('_')[-1].split('.')[0])
        
        # 获取captions
        if img_id in img_to_captions:
            captions = img_to_captions[img_id][:5]
        else:
            captions = [s['raw'] for s in img_info['sentences'][:5]]
        
        while len(captions) < 5:
            captions.append(captions[0])
        captions = captions[:5]
        
        with torch.no_grad():
            # 方法1: 使用vision_model (我们之前的方法)
            img_inputs = clip_processor(images=image, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            img_outputs = clip_model.vision_model(**img_inputs)
            img_feat_v1 = clip_model.visual_projection(img_outputs.pooler_output)
            img_feat_v1 = F.normalize(img_feat_v1, dim=-1).cpu()
            
            # 方法2: 使用get_image_features (官方推荐)
            img_feat_v2 = clip_model.get_image_features(**img_inputs)
            img_feat_v2 = F.normalize(img_feat_v2, dim=-1).cpu()
            
            # 对比两种方法
            if i == 0:
                print(f"\n  方法对比:")
                print(f"    vision_model: {img_feat_v1[0][:5]}")
                print(f"    get_image_features: {img_feat_v2[0][:5]}")
                print(f"    差异: {(img_feat_v1 - img_feat_v2).abs().max():.6f}")
            
            img_feats_new.append(img_feat_v2)
            
            # 提取文本特征
            for caption in captions:
                txt_inputs = clip_processor(text=caption, return_tensors="pt", padding=True, truncation=True)
                txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
                txt_feat = clip_model.get_text_features(**txt_inputs)
                txt_feat = F.normalize(txt_feat, dim=-1).cpu()
                txt_feats_new.append(txt_feat)
    
    except Exception as e:
        print(f"\n  错误: 样本{i}失败: {e}")
        continue

img_feats_new = torch.cat(img_feats_new, dim=0)
txt_feats_new = torch.cat(txt_feats_new, dim=0)

print(f"\n  新特征形状:")
print(f"    图像: {img_feats_new.shape}")
print(f"    文本: {txt_feats_new.shape}")

# 3. 对比评估
print("\n[CHECK 3] 对比评估结果...")

def evaluate(img_feats, txt_feats, name):
    N = len(img_feats)
    sims = img_feats @ txt_feats.T
    ranks = []
    for i in range(N):
        correct_indices = list(range(5*i, 5*i+5))
        sorted_indices = torch.argsort(sims[i], descending=True).tolist()
        rank = min([sorted_indices.index(idx) for idx in correct_indices])
        ranks.append(rank)
    ranks = np.array(ranks)
    r1 = (ranks < 1).mean() * 100
    r5 = (ranks < 5).mean() * 100
    r10 = (ranks < 10).mean() * 100
    print(f"\n  {name}:")
    print(f"    R@1  = {r1:.1f}%")
    print(f"    R@5  = {r5:.1f}%")
    print(f"    R@10 = {r10:.1f}%")
    return r1

# 旧特征（前100）
r1_old = evaluate(img_feats_old[:100], txt_feats_old[:500], "旧特征提取方法")

# 新特征
r1_new = evaluate(img_feats_new, txt_feats_new, "新特征提取方法")

print("\n" + "="*70)
print("诊断结果")
print("="*70)
print(f"\n旧方法 R@1: {r1_old:.1f}%")
print(f"新方法 R@1: {r1_new:.1f}%")
print(f"差异: {r1_new - r1_old:+.1f}%")

if abs(r1_new - r1_old) < 2:
    print("\n结论: 特征提取方法没有问题")
    print("问题可能在其他地方:")
    print("  1. 数据集版本不同")
    print("  2. 评估协议不同")
    print("  3. CLIP版本不同")
else:
    print("\n结论: 特征提取方法有差异！")
    print("建议使用新方法（get_image_features）重新提取全部特征")

print("\n" + "="*70)
print("下一步")
print("="*70)
print("\n如果新方法有提升:")
print("  → 重新提取完整5K测试集特征")
print("  → 重新测试CLIP baseline和AWFF")
print("\n如果没有提升:")
print("  → 问题在评估协议或数据集")
print("  → 需要进一步调查")
