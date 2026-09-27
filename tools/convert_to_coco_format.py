#!/usr/bin/env python3
"""
将dataset_coco.json转换为COCO标准格式的karpathy annotations
"""
import json
from pathlib import Path

DATA_DIR = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco")

print("="*80)
print("转换为标准COCO格式")
print("="*80)

# 加载我们的数据
print("\n加载dataset_coco.json...")
with open(DATA_DIR / "dataset_coco.json") as f:
    data = json.load(f)

test_images = [img for img in data['images'] if img['split'] == 'test']
print(f"  Test images: {len(test_images)}")

# 转换为COCO标准格式
print("\n转换为COCO格式...")
coco_format = {
    "images": [],
    "annotations": []
}

ann_id = 0
for img_data in test_images:
    # 图像信息
    img_id = img_data['cocoid']
    filename = img_data['filename']
    
    coco_format['images'].append({
        "id": img_id,
        "file_name": filename
    })
    
    # Caption annotations
    for sent in img_data['sentences']:
        coco_format['annotations'].append({
            "id": ann_id,
            "image_id": img_id,
            "caption": sent['raw']
        })
        ann_id += 1

print(f"  Images: {len(coco_format['images'])}")
print(f"  Annotations: {len(coco_format['annotations'])}")

# 保存
output_path = DATA_DIR / "coco_test_karpathy.json"
print(f"\n保存到: {output_path}")
with open(output_path, 'w') as f:
    json.dump(coco_format, f)

print("  ✅ 完成")

# 验证
print("\n验证格式:")
with open(output_path) as f:
    loaded = json.load(f)
    print(f"  Images: {len(loaded['images'])}")
    print(f"  Annotations: {len(loaded['annotations'])}")
    print(f"  每图captions: {len(loaded['annotations']) / len(loaded['images']):.1f}")

print("\n示例:")
print(f"  Image: {loaded['images'][0]}")
print(f"  Annotation: {loaded['annotations'][0]}")

print("\n" + "="*80)
