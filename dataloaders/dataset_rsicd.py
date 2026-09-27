#!/usr/bin/env python3
"""
RSICD 数据集加载器

支持格式：
1. 标准格式：JSON文件 + 图像目录
   - annotations.json: [{"image_id": "xxx", "caption": "..."}, ...]
   - images/: xxx.jpg
   
2. 合成数据（用于测试）
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RSICDDataset(Dataset):
    """RSICD 遥感图像描述数据集"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        processor = None,
        max_samples: Optional[int] = None,
        use_synthetic: bool = False
    ):
        """
        Args:
            data_root: 数据根目录
            split: "train", "val", or "test"
            processor: Qwen-VL processor (用于图像预处理)
            max_samples: 最大样本数（用于快速测试）
            use_synthetic: 是否使用合成数据（数据未准备好时）
        """
        self.data_root = Path(data_root)
        self.split = split
        self.processor = processor
        self.max_samples = max_samples
        
        # 加载数据
        if use_synthetic or not self._check_data_exists():
            print(f"⚠️  真实数据不存在，使用合成数据")
            self.samples = self._create_synthetic_data()
        else:
            self.samples = self._load_real_data()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"✅ 加载 {len(self.samples)} 个样本 (split={split})")
    
    def _check_data_exists(self) -> bool:
        """检查真实数据是否存在"""
        ann_file = self.data_root / f"{self.split}_annotations.json"
        img_dir = self.data_root / "images"
        return ann_file.exists() and img_dir.exists()
    
    def _load_real_data(self) -> List[Dict]:
        """加载真实 RSICD 数据"""
        ann_file = self.data_root / f"{self.split}_annotations.json"
        
        with open(ann_file) as f:
            annotations = json.load(f)
        
        # 转换为标准格式
        samples = []
        for ann in annotations:
            samples.append({
                "image_id": ann["image_id"],
                "image_path": str(self.data_root / "images" / f"{ann['image_id']}.jpg"),
                "caption": ann["caption"]
            })
        
        return samples
    
    def _create_synthetic_data(self) -> List[Dict]:
        """创建合成数据用于测试"""
        templates = [
            "A remote sensing image showing {object} in {location}.",
            "Aerial view of {object} with {feature}.",
            "Satellite imagery depicting {object} and {feature}.",
            "{object} visible in this overhead photograph.",
            "High resolution remote sensing data of {object}."
        ]
        
        objects = ["buildings", "roads", "vegetation", "water bodies", "agricultural fields",
                   "urban areas", "forests", "rivers", "parking lots", "industrial zones"]
        
        locations = ["urban area", "rural region", "coastal zone", "mountainous terrain",
                     "desert landscape", "residential district", "commercial center"]
        
        features = ["dense vegetation", "clear boundaries", "complex patterns",
                   "regular structures", "natural formations", "geometric layouts"]
        
        samples = []
        num_samples = self.max_samples or 100
        
        for i in range(num_samples):
            template = templates[i % len(templates)]
            obj = objects[i % len(objects)]
            loc = locations[i % len(locations)]
            feat = features[i % len(features)]
            
            caption = template.format(object=obj, location=loc, feature=feat)
            
            samples.append({
                "image_id": f"synthetic_{i:05d}",
                "image_path": None,  # 合成图像在 __getitem__ 中生成
                "caption": caption
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        if sample["image_path"] and Path(sample["image_path"]).exists():
            image = Image.open(sample["image_path"]).convert("RGB")
        else:
            # 创建合成图像（随机噪声）
            image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
        
        # 预处理（如果有 processor）
        if self.processor:
            # Qwen-VL processor 会处理图像和文本
            # 这里简化处理，实际使用时需要根据 Qwen-VL 的接口调整
            processed = {
                "pixel_values": torch.from_numpy(np.array(image)).float() / 255.0,
                "input_ids": self._tokenize_caption(sample["caption"])
            }
            return processed
        else:
            # 返回原始数据
            return {
                "image": image,
                "caption": sample["caption"],
                "image_id": sample["image_id"]
            }
    
    def _tokenize_caption(self, caption: str):
        """简单的 tokenization（实际使用 processor.tokenizer）"""
        # 这里是占位符，实际应该用 Qwen-VL 的 tokenizer
        return torch.randint(0, 50000, (20,))  # 假设 20 个 token


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    use_synthetic: bool = False
):
    """创建训练、验证、测试的 DataLoader"""
    from torch.utils.data import DataLoader
    
    datasets = {}
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        datasets[split] = RSICDDataset(
            data_root=data_root,
            split=split,
            max_samples=max_samples,
            use_synthetic=use_synthetic
        )
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train")
        )
    
    return dataloaders


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 RSICD 数据加载器")
    print("=" * 60)
    
    # 使用合成数据测试
    dataset = RSICDDataset(
        data_root="./data/RSICD",
        split="train",
        max_samples=10,
        use_synthetic=True
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载
    sample = dataset[0]
    print(f"\n第一个样本:")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Caption: {sample['caption']}")
    print(f"  Image shape: {sample['image'].size}")
    
    # 创建 DataLoader
    dataloaders = create_dataloaders(
        data_root="./data/RSICD",
        batch_size=4,
        max_samples=10,
        use_synthetic=True
    )
    
    print(f"\n✅ DataLoader 创建成功")
    print(f"  Train: {len(dataloaders['train'])} batches")
    print(f"  Val: {len(dataloaders['val'])} batches")
    print(f"  Test: {len(dataloaders['test'])} batches")
