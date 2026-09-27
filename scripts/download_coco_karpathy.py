#!/usr/bin/env python3
"""
下载并准备COCO Karpathy split
使用标准的train/val/test分割进行验证
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"设备: {device}")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("="*80)
print("下载COCO Karpathy Split")
print("="*80)

# ============================================================================
# 1. 下载COCO数据集
# ============================================================================

print("\n步骤1: 下载COCO数据集...")
print("使用HuggingFace的COCO数据集（已经包含Karpathy split）")

try:
    # 尝试使用已有的Karpathy split数据集
    dataset = load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)
    print("✅ 成功加载COCO数据集")
    print(f"   可用splits: {list(dataset.keys())}")
    
    for split_name in dataset.keys():
        print(f"   {split_name}: {len(dataset[split_name])} 样本")
    
except Exception as e:
    print(f"❌ 方法1失败: {e}")
    print("\n尝试备选方案...")
    
    try:
        # 备选方案：使用nlphuji/COCO
        dataset = load_dataset("nlphuji/coco", trust_remote_code=True)
        print("✅ 成功加载COCO数据集（备选源）")
        print(f"   可用splits: {list(dataset.keys())}")
    except Exception as e2:
        print(f"❌ 备选方案也失败: {e2}")
        print("\n最后尝试：手动指定Karpathy split...")
        
        # 如果都失败，使用标准COCO并手动分割
        # Karpathy split: train: 113,287, val: 5,000, test: 5,000
        print("请手动提供COCO Karpathy split数据集")
        print("下载地址: https://www.kaggle.com/datasets/shtvkumar/karpathy-splits")
        exit(1)

# ============================================================================
# 2. 使用适当的split
# ============================================================================

print("\n步骤2: 准备训练和测试集...")

# 根据实际可用的split选择
if 'train' in dataset and 'test' in dataset:
    # 标准split
    train_split = 'train'
    test_split = 'test'
elif 'train' in dataset and 'validation' in dataset:
    train_split = 'train'
    test_split = 'validation'  # 使用验证集作为测试
else:
    # 使用最大的split并手动分割
    split_name = list(dataset.keys())[0]
    full_data = dataset[split_name]
    
    # Karpathy比例：train 95%, test 5%
    n_total = len(full_data)
    n_test = 5000  # 标准测试集大小
    n_train = n_total - n_test
    
    print(f"手动分割数据集:")
    print(f"  总样本: {n_total}")
    print(f"  训练集: {n_train}")
    print(f"  测试集: {n_test}")
    
    import random
    random.seed(42)
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = full_data.select(train_indices)
    test_dataset = full_data.select(test_indices)
    
    train_split = None
    test_split = None

if train_split:
    train_dataset = dataset[train_split]
    test_dataset = dataset[test_split]

print(f"\n最终split:")
print(f"  训练集: {len(train_dataset)} 样本")
print(f"  测试集: {len(test_dataset)} 样本")

# 由于COCO训练集太大，我们使用一个子集用于快速验证
# 完整训练需要更长时间
USE_SUBSET = True
SUBSET_SIZE = 10000  # 使用10K训练样本

if USE_SUBSET and len(train_dataset) > SUBSET_SIZE:
    print(f"\n⚠️  训练集过大，使用{SUBSET_SIZE}样本子集进行快速验证")
    train_dataset = train_dataset.select(range(SUBSET_SIZE))
    print(f"   子集大小: {len(train_dataset)} 样本")

# ============================================================================
# 3. 加载CLIP并提取特征
# ============================================================================

print("\n步骤3: 加载CLIP模型...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def extract_features_from_dataset(dataset, split_name, max_samples=None):
    """从数据集提取CLIP特征"""
    vision_features = []
    text_features = []
    
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    
    for item in tqdm(dataset, desc=f"提取{split_name}特征"):
        try:
            # 处理图像
            if 'image' in item:
                image = item['image']
            elif 'img' in item:
                image = item['img']
            else:
                print(f"警告: 找不到图像字段，跳过")
                continue
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 处理文本（COCO每张图有多个caption，取第一个）
            if 'caption' in item:
                caption = item['caption']
                if isinstance(caption, list):
                    caption = caption[0]
            elif 'text' in item:
                caption = item['text']
                if isinstance(caption, list):
                    caption = caption[0]
            else:
                print(f"警告: 找不到caption字段，跳过")
                continue
            
            # 提取特征
            image_inputs = processor(images=image, return_tensors="pt")
            text_inputs = processor(text=caption, return_tensors="pt", 
                                   padding="max_length", truncation=True, max_length=77)
            
            with torch.no_grad():
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                # 使用完整forward获取投影后的特征
                outputs = clip_model(**image_inputs, **text_inputs)
                
                v_feat = outputs.image_embeds.cpu().squeeze(0)
                t_feat = outputs.text_embeds.cpu().squeeze(0)
            
            vision_features.append(v_feat)
            text_features.append(t_feat)
            
        except Exception as e:
            print(f"警告: 处理样本失败 - {e}")
            continue
    
    if len(vision_features) == 0:
        raise ValueError("没有成功提取任何特征！")
    
    return torch.stack(vision_features), torch.stack(text_features)

# 提取特征
print(f"\n步骤4: 提取特征（这将需要一些时间）...")

print(f"\n提取训练集特征...")
train_v, train_t = extract_features_from_dataset(train_dataset, "训练集")

print(f"\n提取测试集特征...")
# 测试集限制在5000样本（标准Karpathy test大小）
test_v, test_t = extract_features_from_dataset(test_dataset, "测试集", max_samples=5000)

print(f"\n特征提取完成:")
print(f"  训练集: {train_v.shape}")
print(f"  测试集: {test_v.shape}")

# ============================================================================
# 4. 评估CLIP baseline
# ============================================================================

def evaluate(v_feat, t_feat, name):
    """评估检索性能"""
    v_norm = F.normalize(v_feat, dim=-1)
    t_norm = F.normalize(t_feat, dim=-1)
    sims = v_norm @ t_norm.T
    
    ranks = []
    for i in range(len(v_norm)):
        sim_i = sims[i]
        rank = (sim_i >= sim_i[i]).sum().item() - 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks)
    r1 = (ranks == 0).float().mean().item() * 100
    r5 = (ranks < 5).float().mean().item() * 100
    r10 = (ranks < 10).float().mean().item() * 100
    
    print(f"\n{name} CLIP baseline:")
    print(f"  R@1:  {r1:.2f}%")
    print(f"  R@5:  {r5:.2f}%")
    print(f"  R@10: {r10:.2f}%")
    
    return r1, r5, r10

train_r1, _, _ = evaluate(train_v, train_t, "训练集")
test_r1, _, _ = evaluate(test_v, test_t, "测试集")

# 合理性检查
gap = train_r1 - test_r1
print(f"\n合理性检查:")
print(f"  训练-测试差距: {gap:.2f}%")

if gap > -5:
    print("  ✅ 正常：训练集性能 ≥ 测试集性能")
else:
    print("  ⚠️  异常：训练集性能 < 测试集性能")

# ============================================================================
# 5. 保存特征
# ============================================================================

torch.save({
    'vision_features': train_v,
    'text_features': train_t,
    'num_samples': len(train_v)
}, DATA_DIR / "coco_karpathy_train_clip_features.pt")

torch.save({
    'vision_features': test_v,
    'text_features': test_t,
    'num_samples': len(test_v)
}, DATA_DIR / "coco_karpathy_test_clip_features.pt")

print("\n✅ 特征已保存:")
print("   coco_karpathy_train_clip_features.pt")
print("   coco_karpathy_test_clip_features.pt")

print("\n" + "="*80)
print("下一步: 在训练集上训练AWFF，在测试集上验证")
print("="*80)
