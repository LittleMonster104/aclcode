#!/usr/bin/env python3
"""
Step 1: 验证COCO Karpathy数据完整性
"""
import json
from pathlib import Path
import os

print("="*80)
print("验证COCO Karpathy数据完整性")
print("="*80)

DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")

# 检查目录结构
print("\n步骤1: 检查目录结构...")
required_items = {
    'dataset_coco.json': 'file',
    'train2014': 'dir',
    'val2014': 'dir'
}

all_ok = True
for item_name, item_type in required_items.items():
    item_path = DATA_DIR / item_name
    if item_type == 'file':
        if item_path.is_file():
            size = item_path.stat().st_size / 1024 / 1024
            print(f"  ✅ {item_name}: {size:.1f} MB")
        else:
            print(f"  ❌ {item_name}: 不存在")
            all_ok = False
    else:  # dir
        if item_path.is_dir():
            count = len(list(item_path.glob("*.jpg")))
            print(f"  ✅ {item_name}/: {count:,} 张图像")
        else:
            print(f"  ❌ {item_name}/: 不存在")
            all_ok = False

if not all_ok:
    print("\n❌ 数据不完整，请检查下载")
    exit(1)

# 检查图像数量
print("\n步骤2: 验证图像数量...")
train_images = list((DATA_DIR / "train2014").glob("*.jpg"))
val_images = list((DATA_DIR / "val2014").glob("*.jpg"))

print(f"  train2014: {len(train_images):,} 张（预期: 82,783）")
print(f"  val2014:   {len(val_images):,} 张（预期: 40,504）")

if len(train_images) != 82783:
    print(f"  ⚠️  train2014数量不对！")
if len(val_images) != 40504:
    print(f"  ⚠️  val2014数量不对！")

# 解析Karpathy split
print("\n步骤3: 解析Karpathy split...")
with open(DATA_DIR / "dataset_coco.json", 'r') as f:
    karpathy_data = json.load(f)

print(f"  总图像数: {len(karpathy_data['images']):,}")

# 按split分组
splits = {'train': 0, 'val': 0, 'test': 0, 'restval': 0}
split_images = {'train': [], 'val': [], 'test': [], 'restval': []}

for img in karpathy_data['images']:
    split = img['split']
    if split in splits:
        splits[split] += 1
        split_images[split].append(img)

print(f"\nKarpathy split统计:")
print(f"  train:   {splits['train']:,} 张（预期: 113,287）")
print(f"  val:     {splits['val']:,} 张（预期: 5,000）")
print(f"  test:    {splits['test']:,} 张（预期: 5,000）")
print(f"  restval: {splits['restval']:,} 张")

# 验证是否符合标准
if splits['train'] == 113287 and splits['val'] == 5000 and splits['test'] == 5000:
    print("\n✅ 完全符合标准Karpathy split！")
else:
    print("\n⚠️  数量与标准Karpathy split不完全一致")

# 统计captions
total_captions = sum(len(img['sentences']) for img in karpathy_data['images'])
train_captions = sum(len(img['sentences']) for img in split_images['train'])
test_captions = sum(len(img['sentences']) for img in split_images['test'])

print(f"\nCaption统计:")
print(f"  总captions: {total_captions:,}")
print(f"  train captions: {train_captions:,} (预期: ~566K)")
print(f"  test captions:  {test_captions:,} (预期: 25K)")

# 检查图像文件是否存在
print("\n步骤4: 抽查图像文件是否存在...")
sample_size = 100
import random
random.seed(42)
sample_images = random.sample(karpathy_data['images'], sample_size)

missing = 0
for img in sample_images:
    filepath = img['filepath']  # 如 'val2014'
    filename = img['filename']  # 如 'COCO_val2014_000000123456.jpg'
    full_path = DATA_DIR / filepath / filename
    
    if not full_path.exists():
        missing += 1
        if missing <= 3:  # 只打印前3个
            print(f"  ❌ 缺失: {filepath}/{filename}")

if missing == 0:
    print(f"  ✅ 抽查{sample_size}张图像，全部存在")
else:
    print(f"  ⚠️  抽查{sample_size}张图像，缺失{missing}张")

print("\n" + "="*80)
print("数据验证完成")
print("="*80)

if all_ok and missing == 0:
    print("\n✅ 数据完整，可以开始提取CLIP特征")
    
    print("\n下一步:")
    print("  运行: python extract_coco_karpathy_clip_features.py")
else:
    print("\n⚠️  数据存在问题，请检查")
