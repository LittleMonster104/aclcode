# 数据集准备指南

本目录用于存放实验所需的数据集。

## 📁 目录结构

```
data/
├── coco/               # COCO Karpathy split
├── RSITMD/            # RSITMD 遥感数据集
├── RSICD/             # RSICD 遥感数据集（可选）
├── flickr8k/          # Flickr8K 数据集（可选）
├── educational/       # 教育数据集（已提供）
└── features/          # 预计算的特征缓存
```

## 📥 数据下载

### 1. COCO Dataset (必需)
```bash
python3 ../scripts/download_coco_karpathy.py
```
预期大小：~13GB  
预期位置：`data/coco/`

### 2. RSITMD Dataset (必需)
```bash
python3 ../scripts/download_rsitmd.py
```
预期大小：~2GB  
预期位置：`data/RSITMD/`

### 3. Flickr8K Dataset (可选)
```bash
python3 ../scripts/download_flickr8k.py
```
预期大小：~1GB  
预期位置：`data/flickr8k/`

### 4. RSICD Dataset (可选)
```bash
python3 ../scripts/download_rsicd.py
```
预期大小：~4GB  
预期位置：`data/RSICD/`

## 🔍 验证数据完整性

```bash
# 检查 COCO 数据集
python3 ../tools/verify_coco_karpathy_data.py

# 检查 COCO splits
python3 ../tools/check_coco_splits.py
```

## 💾 特征预计算（推荐）

为了加速实验，可以预先提取CLIP特征：

```bash
# 提取 COCO 特征
python3 ../preprocessing/extract_coco_karpathy_clip_features.py --model ViT-B/32
python3 ../preprocessing/extract_coco_karpathy_clip_features.py --model ViT-L/14

# 提取 RSITMD 特征
python3 ../preprocessing/precompute_clip_features.py --dataset RSITMD --model ViT-B/32
```

特征将保存在 `data/features/` 目录。

## ⚠️ 注意事项

1. **磁盘空间**: 确保至少有 20GB 可用空间
2. **网络连接**: 下载可能需要较长时间，建议使用稳定的网络连接
3. **HuggingFace访问**: 如果在中国大陆，可能需要设置镜像或代理

## 🆘 遇到问题？

- 数据下载失败：检查网络连接，或尝试手动下载
- 特征提取错误：确保安装了正确版本的 torch 和 clip
- 路径找不到：运行 `python3 -c "import config; config.ensure_dirs()"`

