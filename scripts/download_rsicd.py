#!/usr/bin/env python3
"""
下载 RSICD 遥感数据集（10K 图像-文本对）
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, dest_path):
    """带进度条的文件下载"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def download_rsicd():
    """
    RSICD 数据集信息:
    - 10,921 张遥感图像
    - 每张图像 5 个标注
    - 总共 54,605 个图像-文本对
    - 大小: ~2.5GB
    """
    data_dir = Path("./data/rsicd")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("RSICD 数据集下载指南:")
    print("\n方案1: 从 GitHub 下载（可能需要代理）")
    print("  仓库: https://github.com/201528014227051/RSICD_optimal")
    print("  下载: git clone https://github.com/201528014227051/RSICD_optimal.git")
    
    print("\n方案2: 从百度网盘下载（国内推荐）")
    print("  链接: https://pan.baidu.com/s/1bp71tE3")
    print("  提取码: (在原论文或 GitHub README 中查找)")
    
    print("\n方案3: 从原始来源下载")
    print("  论文: Remote Sensing Image Captioning Dataset (RSICD)")
    print("  链接: https://github.com/201528014227051/RSICD_optimal")
    
    print("\n下载后请确保目录结构:")
    print("  data/rsicd/")
    print("  ├── images/          # 10,921 张 .jpg 图像")
    print("  ├── dataset.json     # 标注文件")
    print("  └── README.md")
    
    print("\n\n快速开始（如果您已有数据）:")
    print("  将图像放到: data/rsicd/images/")
    print("  将标注放到: data/rsicd/dataset.json")
    
    return data_dir

def verify_rsicd(data_dir="./data/rsicd"):
    """验证数据集完整性"""
    data_dir = Path(data_dir)
    
    images_dir = data_dir / "images"
    annotations = data_dir / "dataset.json"
    
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        return False
    
    if not annotations.exists():
        print(f"❌ 标注文件不存在: {annotations}")
        return False
    
    num_images = len(list(images_dir.glob("*.jpg")))
    print(f"✅ 找到 {num_images} 张图像")
    
    import json
    with open(annotations, 'r') as f:
        data = json.load(f)
    
    if 'images' in data:
        print(f"✅ 标注文件包含 {len(data['images'])} 条记录")
    
    if num_images > 10000:
        print("✅ RSICD 数据集验证通过!")
        return True
    else:
        print("⚠️  图像数量不足，请检查下载完整性")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RSICD 遥感数据集下载工具")
    print("=" * 60)
    
    data_dir = download_rsicd()
    
    print("\n" + "=" * 60)
    print("下载完成后，运行验证:")
    print("  python download_rsicd.py --verify")
    print("=" * 60)
    
    import sys
    if "--verify" in sys.argv:
        print("\n验证数据集...")
        verify_rsicd()
