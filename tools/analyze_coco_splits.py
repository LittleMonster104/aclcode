import json
from pathlib import Path

data = json.load(open('/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/dataset_coco.json'))
coco_dir = Path('/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco/val2014')

print(f'标注文件记录数: {len(data["images"])}')
print(f'val2014 目录图像数: {len(list(coco_dir.glob("*.jpg")))}')
print()

# 检查每个 split 的文件存在情况
splits_total = {}
splits_exists = {}

for img in data['images']:
    split = img['split']
    splits_total[split] = splits_total.get(split, 0) + 1
    
    img_path = coco_dir / img['filename']
    if img_path.exists():
        splits_exists[split] = splits_exists.get(split, 0) + 1

print('Split 统计:')
print(f'{"Split":<15} {"标注数":<10} {"存在数":<10} {"存在率"}')
print('-' * 50)
for split in sorted(splits_total.keys()):
    total = splits_total[split]
    exists = splits_exists.get(split, 0)
    rate = exists / total * 100 if total > 0 else 0
    print(f'{split:<15} {total:<10} {exists:<10} {rate:.1f}%')

print()
print('结论:')
print(f'只有 val2014 中的 {sum(splits_exists.values())} 张图像可用')
print(f'其余 {len(data["images"]) - sum(splits_exists.values())} 张来自 train2014（未下载）')
