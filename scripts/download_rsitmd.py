#!/usr/bin/env python3
"""
下载 RSITMD 数据集
Remote Sensing Image Text Matching Dataset
"""

import os
import requests
from pathlib import Path
import zipfile
import json

print("=" * 70)
print("下载 RSITMD 数据集")
print("=" * 70)

data_dir = Path("/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data/RSITMD")
data_dir.mkdir(parents=True, exist_ok=True)

# RSITMD 数据集信息
print("\n[1/3] 准备下载 RSITMD...")
print("  来源: GitHub - xiaoyuan1996/RSITMD")
print("  包含: 遥感图像 + 文本描述")

# 方法 1: 尝试从 GitHub 下载
github_urls = [
    "https://github.com/xiaoyuan1996/RSITMD/archive/refs/heads/master.zip",
    "https://codeload.github.com/xiaoyuan1996/RSITMD/zip/refs/heads/master"
]

print("\n[2/3] 开始下载...")
downloaded = False

for i, url in enumerate(github_urls, 1):
    try:
        print(f"  尝试方法 {i}: {url}")
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            zip_path = data_dir / "rsitmd.zip"
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"  文件大小: {total_size / (1024*1024):.1f} MB")
            
            with open(zip_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = downloaded_size / total_size * 100
                            print(f"\r  进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n  ✅ 下载完成: {zip_path}")
            
            # 解压
            print("\n[3/3] 解压文件...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print(f"  ✅ 解压完成")
            
            # 删除 zip
            zip_path.unlink()
            
            downloaded = True
            break
            
    except Exception as e:
        print(f"\n  ⚠️  方法 {i} 失败: {e}")
        continue

if not downloaded:
    print("\n❌ 自动下载失败")
    print("\n手动下载方法:")
    print("  1. 访问: https://github.com/xiaoyuan1996/RSITMD")
    print("  2. 点击 'Code' -> 'Download ZIP'")
    print(f"  3. 解压到: {data_dir}")
    print("\n或者使用 git:")
    print(f"  cd {data_dir.parent}")
    print(f"  git clone https://github.com/xiaoyuan1996/RSITMD.git")
else:
    # 检查下载的内容
    print("\n" + "=" * 70)
    print("✅ RSITMD 下载完成")
    print("=" * 70)
    
    # 列出内容
    print("\n内容:")
    for item in data_dir.iterdir():
        if item.is_dir():
            n_files = len(list(item.rglob('*')))
            print(f"  📁 {item.name}/ ({n_files} 文件)")
        else:
            size = item.stat().st_size / 1024
            print(f"  📄 {item.name} ({size:.1f} KB)")
