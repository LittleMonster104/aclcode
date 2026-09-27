import json
from pathlib import Path
from collections import Counter

data = json.load(open('/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/dataset_coco.json'))
coco_dir = Path('/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/coco/val2014')

all_available = []
for img in data['images']:
    img_path = coco_dir / img['filename']
    if img_path.exists() and img['sentences']:
        all_available.append(img['split'])

print(f'实际可用样本数: {len(all_available)}')
print('Split 分布:')
for split, count in Counter(all_available).items():
    print(f'  {split}: {count}')
