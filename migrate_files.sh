#!/bin/bash
# 代码迁移自动化脚本
# 从 workspace 迁移必要文件到 code_data

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 目录定义
WORKSPACE_DIR="/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace"
CODE_DATA_DIR="/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI-ACL/acl/code_data"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ACE 代码迁移脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查源目录
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo -e "${RED}❌ 错误: workspace 目录不存在: $WORKSPACE_DIR${NC}"
    exit 1
fi

# 进入目标目录
cd "$CODE_DATA_DIR"
echo -e "${GREEN}✅ 当前目录: $(pwd)${NC}"
echo ""

# ============================================
# Phase 1: 创建目录结构
# ============================================
echo -e "${YELLOW}[Phase 1] 创建目录结构...${NC}"

mkdir -p dataloaders
mkdir -p preprocessing
mkdir -p utils
mkdir -p scripts
mkdir -p tools
mkdir -p data/features
mkdir -p results/raw

echo -e "${GREEN}✅ 目录结构创建完成${NC}"
echo ""

# ============================================
# Phase 2: 迁移 P0 优先级文件（核心依赖）
# ============================================
echo -e "${YELLOW}[Phase 2] 迁移 P0 优先级文件（核心依赖）...${NC}"

# 数据加载器
echo "  📁 复制数据加载器..."
cp "$WORKSPACE_DIR/data_loader.py" dataloaders/ && echo "    ✓ data_loader.py"
cp "$WORKSPACE_DIR/caption_dataset.py" dataloaders/ && echo "    ✓ caption_dataset.py"
cp "$WORKSPACE_DIR/dataset_rsicd.py" dataloaders/ && echo "    ✓ dataset_rsicd.py"
touch dataloaders/__init__.py

# 特征提取
echo "  🔧 复制特征提取脚本..."
cp "$WORKSPACE_DIR/extract_coco_karpathy_clip_features.py" preprocessing/ && echo "    ✓ extract_coco_karpathy_clip_features.py"

# 评估工具
echo "  📊 复制评估工具..."
if [ -f "$WORKSPACE_DIR/eval_retrieval.py" ]; then
    cp "$WORKSPACE_DIR/eval_retrieval.py" utils/ && echo "    ✓ eval_retrieval.py"
else
    echo "    ⚠️  eval_retrieval.py 不存在，跳过"
fi
touch utils/__init__.py

echo -e "${GREEN}✅ P0 文件迁移完成${NC}"
echo ""

# ============================================
# Phase 3: 迁移 P1 优先级文件（扩展功能）
# ============================================
echo -e "${YELLOW}[Phase 3] 迁移 P1 优先级文件（扩展功能）...${NC}"

# 下载脚本
echo "  📥 复制数据下载脚本..."
for script in download_coco_karpathy download_rsitmd download_flickr8k download_rsicd; do
    if [ -f "$WORKSPACE_DIR/${script}.py" ]; then
        cp "$WORKSPACE_DIR/${script}.py" scripts/ && echo "    ✓ ${script}.py"
    else
        echo "    ⚠️  ${script}.py 不存在，跳过"
    fi
done

# 预处理脚本
echo "  🔧 复制预处理脚本..."
if [ -f "$WORKSPACE_DIR/extract_coco_test_openai_clip.py" ]; then
    cp "$WORKSPACE_DIR/extract_coco_test_openai_clip.py" preprocessing/ && echo "    ✓ extract_coco_test_openai_clip.py"
fi
if [ -f "$WORKSPACE_DIR/precompute_clip_features.py" ]; then
    cp "$WORKSPACE_DIR/precompute_clip_features.py" preprocessing/ && echo "    ✓ precompute_clip_features.py"
fi

# 评估工具扩展
echo "  📊 复制评估工具扩展..."
if [ -f "$WORKSPACE_DIR/eval_clip_standard.py" ]; then
    cp "$WORKSPACE_DIR/eval_clip_standard.py" utils/ && echo "    ✓ eval_clip_standard.py"
fi

echo -e "${GREEN}✅ P1 文件迁移完成${NC}"
echo ""

# ============================================
# Phase 4: 迁移 P2 优先级文件（辅助工具）
# ============================================
echo -e "${YELLOW}[Phase 4] 迁移 P2 优先级文件（辅助工具）...${NC}"

# 诊断工具
echo "  🔍 复制诊断工具..."
for tool in check_coco_splits check_clip_features verify_coco_karpathy_data diagnose_clip_baseline; do
    if [ -f "$WORKSPACE_DIR/${tool}.py" ]; then
        cp "$WORKSPACE_DIR/${tool}.py" tools/ && echo "    ✓ ${tool}.py"
    fi
done

# 数据处理工具
echo "  🛠️  复制数据处理工具..."
for tool in convert_to_coco_format prepare_flickr8k analyze_coco_splits; do
    if [ -f "$WORKSPACE_DIR/${tool}.py" ]; then
        cp "$WORKSPACE_DIR/${tool}.py" tools/ && echo "    ✓ ${tool}.py"
    fi
done

# 结果分析工具
echo "  📈 复制结果分析工具..."
for tool in summarize_results compare_results generate_figures; do
    if [ -f "$WORKSPACE_DIR/${tool}.py" ]; then
        cp "$WORKSPACE_DIR/${tool}.py" tools/ && echo "    ✓ ${tool}.py"
    fi
done

echo -e "${GREEN}✅ P2 文件迁移完成${NC}"
echo ""

# ============================================
# Phase 5: 创建配置文件
# ============================================
echo -e "${YELLOW}[Phase 5] 创建配置文件...${NC}"

# 创建 config.py
cat > config.py << 'EOF'
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
EOF

echo -e "${GREEN}✅ config.py 创建完成${NC}"

# 创建 requirements.txt
cat > requirements.txt << 'EOF'
# ACE Project Dependencies

# Core Dependencies
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.23.0
Pillow>=9.0.0

# CLIP Models
clip @ git+https://github.com/openai/CLIP.git
open_clip_torch>=2.20.0

# Utilities
tqdm>=4.65.0
transformers>=4.30.0

# Optional: For analysis
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.5.0
scipy>=1.10.0
EOF

echo -e "${GREEN}✅ requirements.txt 创建完成${NC}"

# 创建数据下载说明
cat > data/README.md << 'EOF'
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

EOF

echo -e "${GREEN}✅ data/README.md 创建完成${NC}"
echo ""

# ============================================
# Phase 6: 更新现有脚本的导入路径
# ============================================
echo -e "${YELLOW}[Phase 6] 更新导入路径（需要手动检查）...${NC}"

echo "  ⚠️  以下文件需要手动更新导入路径："
echo "    - experiments/coco_table1_main_results.py"
echo "    - experiments/rsitmd_table2_cross_domain.py"
echo "    - baselines/*.py"
echo ""
echo "  添加以下导入："
echo "    from dataloaders.data_loader import ImageCaptionDataset"
echo "    from dataloaders.caption_dataset import CaptionDataset"
echo "    from utils.eval_retrieval import evaluate_retrieval"
echo "    import config"
echo ""

# ============================================
# Phase 7: 迁移结果文件
# ============================================
echo -e "${YELLOW}[Phase 7] 迁移结果文件...${NC}"

echo "  📊 复制结果 JSON 文件..."
for result_file in "$WORKSPACE_DIR/data"/*_results.json "$WORKSPACE_DIR/data"/ablation*.json; do
    if [ -f "$result_file" ]; then
        filename=$(basename "$result_file")
        cp "$result_file" results/raw/ && echo "    ✓ $filename"
    fi
done

echo -e "${GREEN}✅ 结果文件迁移完成${NC}"
echo ""

# ============================================
# Phase 8: 创建快速测试脚本
# ============================================
echo -e "${YELLOW}[Phase 8] 创建测试脚本...${NC}"

cat > test_migration.sh << 'EOF'
#!/bin/bash
# 迁移验证测试脚本

echo "=========================================="
echo "测试迁移完成度"
echo "=========================================="
echo ""

# 测试 Python 导入
echo "[1/5] 测试 Python 导入..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from dataloaders.data_loader import ImageCaptionDataset
    print('  ✅ dataloaders.data_loader')
except Exception as e:
    print(f'  ❌ dataloaders.data_loader: {e}')

try:
    from dataloaders.caption_dataset import CaptionDataset
    print('  ✅ dataloaders.caption_dataset')
except Exception as e:
    print(f'  ⚠️  dataloaders.caption_dataset: {e}')

try:
    import config
    print('  ✅ config')
except Exception as e:
    print(f'  ❌ config: {e}')

try:
    from core.dual_end_fusion import compute_dual_end_fusion
    print('  ✅ core.dual_end_fusion')
except Exception as e:
    print(f'  ❌ core.dual_end_fusion: {e}')
"
echo ""

# 测试目录结构
echo "[2/5] 测试目录结构..."
for dir in dataloaders preprocessing utils scripts tools data results; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ (不存在)"
    fi
done
echo ""

# 测试核心文件
echo "[3/5] 测试核心文件..."
for file in config.py requirements.txt data/README.md MIGRATION_CHECKLIST.md; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (不存在)"
    fi
done
echo ""

# 统计迁移文件
echo "[4/5] 统计迁移文件..."
echo "  dataloaders: $(ls dataloaders/*.py 2>/dev/null | wc -l) 个文件"
echo "  preprocessing: $(ls preprocessing/*.py 2>/dev/null | wc -l) 个文件"
echo "  utils: $(ls utils/*.py 2>/dev/null | wc -l) 个文件"
echo "  scripts: $(ls scripts/*.py 2>/dev/null | wc -l) 个文件"
echo "  tools: $(ls tools/*.py 2>/dev/null | wc -l) 个文件"
echo ""

# 测试快速运行
echo "[5/5] 测试快速运行..."
if [ -f "baselines/quick_test.sh" ]; then
    echo "  ✅ 可以运行: bash baselines/quick_test.sh"
else
    echo "  ⚠️  baselines/quick_test.sh 不存在"
fi

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
EOF

chmod +x test_migration.sh
echo -e "${GREEN}✅ test_migration.sh 创建完成${NC}"
echo ""

# ============================================
# 完成
# ============================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ 迁移脚本执行完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}下一步操作：${NC}"
echo "  1. 运行测试脚本验证迁移："
echo "     ${GREEN}bash test_migration.sh${NC}"
echo ""
echo "  2. 手动更新导入路径："
echo "     编辑 experiments/*.py 和 baselines/*.py"
echo "     添加: from dataloaders.data_loader import ..."
echo ""
echo "  3. 安装依赖："
echo "     ${GREEN}pip install -r requirements.txt${NC}"
echo ""
echo "  4. 准备数据集："
echo "     查看 data/README.md 获取下载指南"
echo ""
echo "  5. 运行快速测试："
echo "     ${GREEN}cd baselines && bash quick_test.sh${NC}"
echo ""
echo -e "${YELLOW}详细信息请查看: MIGRATION_CHECKLIST.md${NC}"
