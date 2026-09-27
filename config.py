#!/usr/bin/env python3
"""
ACE Project Configuration
统一的路径和参数配置
"""
from pathlib import Path
import os

# ============================================
# 路径配置
# ============================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

# 数据集路径
COCO_DIR = DATA_ROOT / "coco"
RSITMD_DIR = DATA_ROOT / "RSITMD"
RSICD_DIR = DATA_ROOT / "RSICD"
FLICKR8K_DIR = DATA_ROOT / "flickr8k"
EDUCATIONAL_DIR = DATA_ROOT / "educational"

# 特征缓存路径
FEATURES_DIR = DATA_ROOT / "features"

# 模型权重路径
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================
# 模型配置
# ============================================
CLIP_MODELS = ["ViT-B/32", "ViT-L/14"]
SIGLIP_MODELS = ["ViT-B-16-SigLIP", "ViT-SO400M-14-SigLIP"]

# ============================================
# 训练配置
# ============================================
CONSISTENCY_EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# ============================================
# 评估配置
# ============================================
TOPK_VALUES = [1, 5, 10]

# ============================================
# 工具函数
# ============================================
def get_data_path(dataset_name: str) -> Path:
    """获取数据集路径"""
    mapping = {
        "coco": COCO_DIR,
        "rsitmd": RSITMD_DIR,
        "rsicd": RSICD_DIR,
        "flickr8k": FLICKR8K_DIR,
        "educational": EDUCATIONAL_DIR,
    }
    return mapping.get(dataset_name.lower(), DATA_ROOT / dataset_name)

def ensure_dirs():
    """确保所有必要目录存在"""
    for dir_path in [DATA_ROOT, RESULTS_ROOT, FEATURES_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("ACE Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Results Root: {RESULTS_ROOT}")
    print("=" * 50)
    ensure_dirs()
    print("✅ All directories ensured")
