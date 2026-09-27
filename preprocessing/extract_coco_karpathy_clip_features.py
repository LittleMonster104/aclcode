#!/usr/bin/env python3
"""
提取COCO Karpathy Split的CLIP特征
合并train和restval作为完整训练集
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"设备: {device}")

DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")
OUTPUT_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data")

print("="*80)
print("提取COCO Karpathy CLIP特征")
print("="*80)

# 加载Karpathy split
print("\n步骤1: 加载Karpathy split...")
with open(DATA_DIR / "dataset_coco.json", 'r') as f:
    karpathy_data = json.load(f)

# 按split分组，合并train和restval
split_images = {
    'train': [],  # 将包含 train + restval
    'val': [],
    'test': []
}

for img in karpathy_data['images']:
    split = img['split']
    if split in ['train', 'restval']:
        split_images['train'].append(img)
    elif split in ['val', 'test']:
        split_images[split].append(img)

print(f"\n合并后的split:")
print(f"  train (train+restval): {len(split_images['train']):,} 张")
print(f"  val:                   {len(split_images['val']):,} 张")
print(f"  test:                  {len(split_images['test']):,} 张")

# 统计captions
for split_name in ['train', 'val', 'test']:
    n_captions = sum(len(img['sentences']) for img in split_images[split_name])
    print(f"  {split_name} captions: {n_captions:,}")

# 加载CLIP模型
print("\n步骤2: 加载CLIP模型...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("  ✅ CLIP模型加载完成")

def extract_features_for_split(split_name, images_data, max_samples=None):
    """
    提取指定split的CLIP特征
    
    注意: 每张图有5条captions，我们只使用第一条
    """
    if max_samples:
        images_data = images_data[:max_samples]
    
    vision_features = []
    text_features = []
    failed = 0
    
    print(f"\n提取 {split_name} 特征: {len(images_data)} 张图像")
    
    for img_data in tqdm(images_data, desc=f"{split_name}"):
        try:
            # 加载图像
            filepath = img_data['filepath']  # 'train2014' 或 'val2014'
            filename = img_data['filename']
            img_path = DATA_DIR / filepath / filename
            
            if not img_path.exists():
                failed += 1
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            # 获取第一条caption
            caption = img_data['sentences'][0]['raw']
            
            # 处理输入
            image_inputs = processor(images=image, return_tensors="pt")
            text_inputs = processor(text=caption, return_tensors="pt", 
                                   padding="max_length", truncation=True, max_length=77)
            
            # 提取特征
            with torch.no_grad():
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                outputs = clip_model(**image_inputs, **text_inputs)
                
                v_feat = outputs.image_embeds.cpu().squeeze(0)
                t_feat = outputs.text_embeds.cpu().squeeze(0)
            
            vision_features.append(v_feat)
            text_features.append(t_feat)
            
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n  警告: 处理失败 {filename}: {e}")
            continue
    
    if failed > 0:
        print(f"\n  ⚠️  {failed} 张图像处理失败")
    
    vision_features = torch.stack(vision_features)
    text_features = torch.stack(text_features)
    
    print(f"  ✅ 完成: vision {vision_features.shape}, text {text_features.shape}")
    
    return vision_features, text_features

# 提取特征 - 由于训练集很大，先只提取test用于快速验证
print("\n步骤3: 提取特征...")

# 先提取test集（用于验证）
print("\n" + "="*80)
print("提取Test集（优先，用于验证）")
print("="*80)

test_v, test_t = extract_features_for_split('test', split_images['test'])

# 保存test特征
torch.save({
    'vision_features': test_v,
    'text_features': test_t,
    'num_samples': len(test_v)
}, OUTPUT_DIR / "coco_karpathy_test_clip_features.pt")

print(f"\n✅ Test特征已保存")

# 评估CLIP baseline
print("\n" + "="*80)
print("步骤4: 评估CLIP baseline（Test集）")
print("="*80)

test_v_norm = F.normalize(test_v, dim=-1)
test_t_norm = F.normalize(test_t, dim=-1)
sims = test_v_norm @ test_t_norm.T

ranks = []
for i in range(len(test_v)):
    sim_i = sims[i]
    rank = (sim_i >= sim_i[i]).sum().item() - 1
    ranks.append(rank)

ranks = torch.tensor(ranks)
r1 = (ranks == 0).float().mean().item() * 100
r5 = (ranks < 5).float().mean().item() * 100
r10 = (ranks < 10).float().mean().item() * 100

print(f"\nCLIP (Zero-Shot) on COCO Test:")
print(f"  R@1:  {r1:.2f}%")
print(f"  R@5:  {r5:.2f}%")
print(f"  R@10: {r10:.2f}%")

print(f"\n文献参考值（COCO 5K test, image-to-text）:")
print(f"  CLIP ViT-B/32: R@1 约 58-62%")

if r1 > 55 and r1 < 65:
    print(f"\n✅ CLIP性能正常，数据正确！")
elif r1 > 50:
    print(f"\n⚠️  CLIP性能略低，但可接受")
else:
    print(f"\n❌ CLIP性能异常，请检查数据")

# 询问是否继续提取训练集
print("\n" + "="*80)
print("下一步")
print("="*80)

print(f"""
Test集特征已提取并验证。

要提取完整训练集（113K images），需要约2-3小时。

选项:
1. 立即提取完整训练集 → 可以在标准数据上训练AWFF
2. 先用10K子集训练 → 快速验证（30分钟）
3. 用已有的Flickr8K训练 → 跨数据集验证

建议: 先验证AWFF在Test上的性能（加载已训练的模型）
然后再决定是否需要完整训练集。

运行: python evaluate_awff_on_coco_test.py
""")

print("="*80)
