#!/usr/bin/env python3
"""
数据加载器：支持 RSICD, UCM-Captions, Flickr8K 等图像描述数据集
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageCaptionDataset(Dataset):
    """通用图像描述数据集加载器"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",  # train, val, test
        processor=None,
        max_samples: Optional[int] = None,
        dataset_name: str = "RSICD"
    ):
        """
        Args:
            data_root: 数据集根目录
            split: 数据分割 (train/val/test)
            processor: 模型的 processor（用于预处理）
            max_samples: 最大样本数（用于快速测试）
            dataset_name: 数据集名称 (RSICD, UCM, Flickr8K)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.processor = processor
        self.dataset_name = dataset_name
        
        # 加载数据
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"✅ {dataset_name} {split} split: {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict]:
        """加载样本（根据数据集格式自动适配）"""
        
        if self.dataset_name == "RSICD":
            return self._load_rsicd()
        elif self.dataset_name == "UCM":
            return self._load_ucm()
        elif self.dataset_name == "Flickr8K":
            return self._load_flickr8k()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_rsicd(self) -> List[Dict]:
        """加载 RSICD 数据集
        
        预期目录结构：
        data/RSICD/
          ├── images/
          │   ├── 00001.jpg
          │   └── ...
          └── dataset_rsicd.json  # 或 captions.json
        """
        # 尝试多种可能的文件名
        caption_files = [
            self.data_root / "dataset_rsicd.json",
            self.data_root / "captions.json",
            self.data_root / f"{self.split}_captions.json"
        ]
        
        caption_file = None
        for f in caption_files:
            if f.exists():
                caption_file = f
                break
        
        if caption_file is None:
            print(f"⚠️  未找到 RSICD 标注文件，使用合成数据")
            return self._generate_synthetic_data(100)
        
        with open(caption_file) as f:
            data = json.load(f)
        
        samples = []
        for item in data.get("images", data):
            if item.get("split") == self.split or "split" not in item:
                img_path = self.data_root / "images" / item["filename"]
                for caption in item.get("sentences", item.get("captions", [])):
                    samples.append({
                        "image_path": str(img_path),
                        "caption": caption if isinstance(caption, str) else caption.get("raw", "")
                    })
        
        return samples
    
    def _load_ucm(self) -> List[Dict]:
        """加载 UCM Captions 数据集"""
        # TODO: 实现 UCM 加载逻辑
        print("⚠️  UCM 加载器待实现，使用合成数据")
        return self._generate_synthetic_data(100)
    
    def _load_flickr8k(self) -> List[Dict]:
        """加载 Flickr8K 数据集
        
        预期目录结构：
        data/Flickr8K/
          ├── Images/
          └── captions.txt  # image_name\tcaption
        """
        caption_file = self.data_root / f"Flickr8k.token.txt"
        
        if not caption_file.exists():
            print(f"⚠️  未找到 Flickr8K 标注文件，使用合成数据")
            return self._generate_synthetic_data(100)
        
        samples = []
        with open(caption_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    img_path = self.data_root / "Images" / img_name
                    samples.append({
                        "image_path": str(img_path),
                        "caption": caption
                    })
        
        return samples
    
    def _generate_synthetic_data(self, n_samples: int) -> List[Dict]:
        """生成合成数据用于测试"""
        templates = [
            "A satellite view of {} with {} features",
            "An aerial image showing {} and {}",
            "Remote sensing image of {} region",
            "A bird's eye view of {} with visible {}"
        ]
        
        objects = ["urban", "forest", "water", "agricultural", "residential", "industrial"]
        features = ["buildings", "roads", "vegetation", "structures", "patterns"]
        
        import random
        samples = []
        for i in range(n_samples):
            template = random.choice(templates)
            if "{}" in template:
                caption = template.format(
                    random.choice(objects),
                    random.choice(features)
                )
            else:
                caption = template
            
            samples.append({
                "image_path": f"synthetic_{i}.jpg",
                "caption": caption,
                "is_synthetic": True
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载图像
        if sample.get("is_synthetic"):
            # 合成数据：创建随机图像
            image = Image.new('RGB', (224, 224), color=(73, 109, 137))
        else:
            try:
                image = Image.open(sample["image_path"]).convert('RGB')
            except Exception as e:
                print(f"⚠️  图像加载失败: {sample['image_path']}, 使用占位符")
                image = Image.new('RGB', (224, 224), color=(73, 109, 137))
        
        caption = sample["caption"]
        
        # 如果有 processor，进行预处理
        if self.processor:
            processed = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "caption": caption
            }
        else:
            return {
                "image": image,
                "caption": caption
            }


def create_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 32,
    processor=None,
    max_samples: Optional[int] = None,
    dataset_name: str = "RSICD",
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器"""
    
    dataset = ImageCaptionDataset(
        data_root=data_root,
        split=split,
        processor=processor,
        max_samples=max_samples,
        dataset_name=dataset_name
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器...")
    
    # 测试 RSICD（合成数据）
    dataset = ImageCaptionDataset(
        data_root="./data/RSICD",
        split="train",
        max_samples=10,
        dataset_name="RSICD"
    )
    
    print(f"\n样本示例:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  {i+1}. Caption: {sample['caption'][:60]}...")
