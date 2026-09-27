# 代码迁移清单：workspace → code_data

**源目录**：`/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/`  
**目标目录**：`/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI-ACL/acl/code_data/`

---

## 📊 总体分析

### workspace 目录统计
- **Python文件总数**：474个
- **核心算法文件**：已迁移（3个）
- **实验脚本**：已迁移（2个）
- **基线方法**：已迁移（14个）
- **教育数据集评估**：已迁移（5个）

### code_data 目录现状
```
code_data/
├── core/                    ✅ 已有核心算法（3个）
├── experiments/             ✅ 已有主实验（2个）
├── baselines/               ✅ 已有基线方法（14个）
├── eduppt/                  ✅ 已有教育数据集（5个）
├── analysis/                ✅ 已有分析脚本（2个）
├── data/                    ⚠️  仅有educational子目录
│   └── educational/
└── results/                 ⚠️  仅有少量结果文件
```

---

## 🔥 优先级1：必须迁移（核心依赖）

### 1.1 数据加载器（Data Loaders）
这些文件被所有实验脚本依赖，必须迁移：

| 文件名 | 路径 | 依赖说明 | 优先级 |
|-------|------|---------|--------|
| `data_loader.py` | `workspace/` | 通用数据加载器，支持RSICD/UCM/Flickr8K等多数据集 | 🔴 P0 |
| `caption_dataset.py` | `workspace/` | Caption数据集加载器，支持Karpathy格式 | 🔴 P0 |
| `dataset_rsicd.py` | `workspace/` | RSICD遥感数据集专用加载器 | 🔴 P0 |

**迁移建议**：
```bash
mkdir -p code_data/dataloaders
cp workspace/data_loader.py code_data/dataloaders/
cp workspace/caption_dataset.py code_data/dataloaders/
cp workspace/dataset_rsicd.py code_data/dataloaders/
```

**需要修改**：
- 所有实验脚本中的导入路径：`from dataloaders.data_loader import ...`
- 数据路径配置：检查硬编码的路径引用

---

### 1.2 特征提取脚本（Feature Extraction）
用于预计算CLIP/SigLIP特征：

| 文件名 | 路径 | 用途 | 优先级 |
|-------|------|------|--------|
| `extract_coco_karpathy_clip_features.py` | `workspace/` | 提取COCO Karpathy split的CLIP特征 | 🔴 P0 |
| `extract_coco_test_openai_clip.py` | `workspace/` | 提取COCO测试集OpenAI CLIP特征 | 🟡 P1 |
| `precompute_clip_features.py` | `workspace/` | 通用CLIP特征预计算脚本 | 🟡 P1 |

**迁移建议**：
```bash
mkdir -p code_data/preprocessing
cp workspace/extract_coco_karpathy_clip_features.py code_data/preprocessing/
cp workspace/extract_coco_test_openai_clip.py code_data/preprocessing/
cp workspace/precompute_clip_features.py code_data/preprocessing/
```

---

### 1.3 评估工具（Evaluation Utils）
核心评估指标计算：

| 文件名 | 路径 | 用途 | 优先级 |
|-------|------|------|--------|
| `eval_retrieval.py` | `workspace/` | 检索评估（R@1/R@5/R@10计算） | 🔴 P0 |
| `eval_clip_standard.py` | `workspace/` | 标准CLIP评估协议 | 🟡 P1 |

**迁移建议**：
```bash
mkdir -p code_data/utils
cp workspace/eval_retrieval.py code_data/utils/
cp workspace/eval_clip_standard.py code_data/utils/
```

---

## 🟡 优先级2：强烈推荐（完整复现）

### 2.1 数据下载脚本（Data Download）
帮助用户准备数据集：

| 文件名 | 用途 | 优先级 |
|-------|------|--------|
| `download_coco_karpathy.py` | 下载COCO Karpathy split | 🟡 P1 |
| `download_flickr8k.py` | 下载Flickr8K数据集 | 🟡 P1 |
| `download_rsitmd.py` | 下载RSITMD遥感数据集 | 🟡 P1 |
| `download_rsicd.py` | 下载RSICD遥感数据集 | 🟡 P1 |

**迁移建议**：
```bash
mkdir -p code_data/scripts
cp workspace/download_*.py code_data/scripts/
```

---

### 2.2 基线方法扩展（Additional Baselines）
workspace中有更多基线实现：

| 文件名 | 方法 | 优先级 |
|-------|------|--------|
| `eval_clip_baseline.py` | 标准CLIP基线 | 🟡 P1 |
| `eval_siglip_baseline.py` | ✅ 已迁移 | - |
| `eval_vitb16_baseline.py` | ✅ 已迁移 | - |
| `eval_blip_coco.py` | BLIP基线 | 🟢 P2 |

---

### 2.3 模型训练脚本（Training Scripts）
如果需要重新训练projector：

| 文件名 | 用途 | 优先级 |
|-------|------|--------|
| `train_awff.py` | 训练AWFF模型 | 🟡 P1 |
| `train_consistency.py` | （可能不存在，需确认） | 🟡 P1 |

---

## 🟢 优先级3：可选迁移（辅助工具）

### 3.1 诊断/调试工具（Diagnostic Tools）

| 文件名 | 用途 | 优先级 |
|-------|------|--------|
| `check_coco_splits.py` | 检查COCO数据集split | 🟢 P2 |
| `check_clip_features.py` | 检查CLIP特征维度 | 🟢 P2 |
| `verify_coco_karpathy_data.py` | 验证Karpathy数据完整性 | 🟢 P2 |
| `diagnose_clip_baseline.py` | 诊断CLIP基线问题 | 🟢 P2 |

**迁移建议**：
```bash
mkdir -p code_data/tools
cp workspace/check_*.py code_data/tools/
cp workspace/verify_*.py code_data/tools/
cp workspace/diagnose_*.py code_data/tools/
```

---

### 3.2 数据处理工具（Data Processing）

| 文件名 | 用途 | 优先级 |
|-------|------|--------|
| `convert_to_coco_format.py` | 转换为COCO格式 | 🟢 P2 |
| `prepare_flickr8k.py` | 准备Flickr8K数据 | 🟢 P2 |
| `analyze_coco_splits.py` | 分析COCO split统计 | 🟢 P3 |

---

### 3.3 结果分析工具（Analysis Tools）

| 文件名 | 用途 | 优先级 |
|-------|------|--------|
| `summarize_results.py` | 汇总实验结果 | 🟢 P2 |
| `compare_results.py` | 对比不同方法结果 | 🟢 P2 |
| `generate_figures.py` | 生成论文图表 | 🟢 P3 |

---

## 📁 数据目录迁移方案

### workspace/data/ 目录结构
```
workspace/data/
├── RSICD/              # RSICD遥感数据集
├── RSITMD/             # RSITMD遥感数据集
├── cifar-10-*/         # ❌ 大量CIFAR文件（不需要）
├── *.pth               # ❌ 模型checkpoint（不迁移）
├── *.json              # ✅ 结果文件（选择性迁移）
└── educational/        # ✅ 已迁移
```

### 迁移建议

#### ✅ 需要迁移的数据
```bash
# 创建数据目录结构
mkdir -p code_data/data/{RSICD,RSITMD,coco,flickr8k}

# 迁移数据集（如果存在且不超过1GB）
cp -r workspace/data/RSICD code_data/data/       # 检查大小
cp -r workspace/data/RSITMD code_data/data/      # 检查大小

# 迁移结果JSON文件（小文件）
cp workspace/data/*_results.json code_data/results/
cp workspace/data/ablation*.json code_data/results/
```

#### ❌ 不需要迁移的数据
- **CIFAR文件**：大量png文件（10000+），占用空间大，论文不需要
- **模型checkpoint**：`*.pth`文件（可能很大）
- **日志文件**：`*.log`文件
- **临时文件**：`.cache/`, `__pycache__/`

#### 📝 数据下载说明文件
创建 `code_data/data/README.md`：
```markdown
# 数据集准备

## COCO Dataset
下载脚本：`../scripts/download_coco_karpathy.py`
预期位置：`data/coco/`

## RSITMD Dataset
下载脚本：`../scripts/download_rsitmd.py`
预期位置：`data/RSITMD/`

## Flickr8K Dataset
下载脚本：`../scripts/download_flickr8k.py`
预期位置：`data/flickr8k/`
```

---

## 🔧 需要修改的配置

### 1. 路径配置统一
创建 `code_data/config.py`：
```python
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

# 数据集路径
COCO_DIR = DATA_ROOT / "coco"
RSITMD_DIR = DATA_ROOT / "RSITMD"
FLICKR8K_DIR = DATA_ROOT / "flickr8k"
EDUCATIONAL_DIR = DATA_ROOT / "educational"

# 特征缓存路径
FEATURES_DIR = DATA_ROOT / "features"
```

### 2. 导入路径修改

**现有实验脚本**：
```python
# experiments/coco_table1_main_results.py (当前)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.dual_end_fusion import compute_dual_end_fusion
```

**迁移后需要添加**：
```python
# experiments/coco_table1_main_results.py (修改后)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.dual_end_fusion import compute_dual_end_fusion
from dataloaders.data_loader import ImageCaptionDataset  # 新增
from utils.eval_retrieval import evaluate_retrieval      # 新增
import config  # 新增
```

---

## 📋 迁移执行计划

### Phase 1: 核心依赖（Day 1）
```bash
# 1. 创建目录结构
cd /Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI-ACL/acl/code_data
mkdir -p dataloaders preprocessing utils scripts tools

# 2. 迁移P0优先级文件
cp ../../../workspace/data_loader.py dataloaders/
cp ../../../workspace/caption_dataset.py dataloaders/
cp ../../../workspace/dataset_rsicd.py dataloaders/
cp ../../../workspace/extract_coco_karpathy_clip_features.py preprocessing/
cp ../../../workspace/eval_retrieval.py utils/

# 3. 创建配置文件
cat > config.py << 'EOF'
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
COCO_DIR = DATA_ROOT / "coco"
RSITMD_DIR = DATA_ROOT / "RSITMD"
FLICKR8K_DIR = DATA_ROOT / "flickr8k"
EDUCATIONAL_DIR = DATA_ROOT / "educational"
FEATURES_DIR = DATA_ROOT / "features"
EOF
```

### Phase 2: 扩展功能（Day 2-3）
```bash
# 4. 迁移P1优先级文件
cp ../../../workspace/download_coco_karpathy.py scripts/
cp ../../../workspace/download_rsitmd.py scripts/
cp ../../../workspace/download_flickr8k.py scripts/
cp ../../../workspace/precompute_clip_features.py preprocessing/

# 5. 迁移辅助工具
cp ../../../workspace/check_*.py tools/
cp ../../../workspace/verify_*.py tools/
```

### Phase 3: 测试与验证（Day 4）
```bash
# 6. 测试基本导入
python3 -c "from dataloaders.data_loader import ImageCaptionDataset; print('✅ Import OK')"

# 7. 运行快速测试
cd baselines
bash quick_test.sh

# 8. 验证主实验脚本
cd ../experiments
python3 coco_table1_main_results.py --quick_test
```

---

## ⚠️ 关键注意事项

### 1. 路径依赖问题
**问题**：workspace中的脚本可能硬编码了路径
```python
# 常见硬编码路径
data_root = "/Users/jiazhu/Documents/ZJNU/EvoScientist/workspace/data"
model_path = "../workspace/models/checkpoint.pth"
```

**解决方案**：
- 使用 `grep -r "/Users/jiazhu" code_data/` 查找硬编码路径
- 替换为相对路径或使用 `config.py` 中的路径变量

### 2. 模型权重文件
**workspace中的模型文件**：
```
workspace/data/
├── awff_best.pth                    # ~100MB
├── awff_coco_best.pth              # ~100MB
├── anchor_bridge_best.pth          # ~50MB
└── checkpoint_ep*.pth              # 多个checkpoint
```

**迁移策略**：
- ✅ 如果文件<50MB：直接复制
- ⚠️ 如果文件>50MB：上传到云存储，提供下载脚本
- 📝 在README中说明如何获取预训练权重

### 3. Python依赖管理
创建 `code_data/requirements.txt`：
```txt
torch>=1.13.0
torchvision>=0.14.0
clip @ git+https://github.com/openai/CLIP.git
open_clip_torch>=2.20.0
transformers>=4.30.0
Pillow>=9.0.0
numpy>=1.23.0
tqdm>=4.65.0
```

---

## 📊 迁移完成度检查清单

### 核心功能（必须100%）
- [ ] ✅ 数据加载器（3个文件）
- [ ] ✅ 特征提取脚本（1个核心+2个扩展）
- [ ] ✅ 评估工具（1个核心）
- [ ] ✅ 配置文件（config.py）
- [ ] ✅ 依赖文件（requirements.txt）

### 实验复现（必须100%）
- [x] ✅ 核心算法（3个）- 已完成
- [x] ✅ 主实验脚本（2个）- 已完成
- [x] ✅ 基线方法（14个）- 已完成
- [x] ✅ 分析脚本（2个）- 已完成
- [ ] ⚠️ 数据下载脚本（4个）- 待迁移

### 辅助工具（可选）
- [ ] 🟡 诊断工具（4个）
- [ ] 🟡 数据处理（3个）
- [ ] 🟡 结果分析（3个）

---

## 🎯 最终目录结构预览

```
code_data/
├── README.md                        # ✅ 已有
├── MIGRATION_CHECKLIST.md          # 📝 本文件
├── config.py                       # 📝 待创建
├── requirements.txt                # 📝 待创建
├── run_all_experiments.sh          # ✅ 已有
│
├── core/                           # ✅ 已完成（3个文件）
│   ├── dual_end_fusion.py
│   ├── consistency_regularization.py
│   └── adaptive_topk.py
│
├── dataloaders/                    # 📝 待创建
│   ├── __init__.py
│   ├── data_loader.py              # 🔴 P0
│   ├── caption_dataset.py          # 🔴 P0
│   └── dataset_rsicd.py            # 🔴 P0
│
├── preprocessing/                  # 📝 待创建
│   ├── extract_coco_karpathy_clip_features.py  # 🔴 P0
│   ├── extract_coco_test_openai_clip.py        # 🟡 P1
│   └── precompute_clip_features.py             # 🟡 P1
│
├── utils/                          # 📝 待创建
│   ├── __init__.py
│   ├── eval_retrieval.py           # 🔴 P0
│   └── eval_clip_standard.py       # 🟡 P1
│
├── scripts/                        # 📝 待创建
│   ├── download_coco_karpathy.py   # 🟡 P1
│   ├── download_rsitmd.py          # 🟡 P1
│   ├── download_flickr8k.py        # 🟡 P1
│   └── download_rsicd.py           # 🟡 P1
│
├── tools/                          # 📝 待创建（可选）
│   ├── check_coco_splits.py
│   ├── verify_coco_karpathy_data.py
│   └── diagnose_clip_baseline.py
│
├── experiments/                    # ✅ 已完成（2个文件）
│   ├── coco_table1_main_results.py
│   └── rsitmd_table2_cross_domain.py
│
├── baselines/                      # ✅ 已完成（14个文件）
│   ├── run_baselines.py
│   ├── eval_*.py (多个)
│   └── ...
│
├── eduppt/                         # ✅ 已完成（5个文件）
│   └── eval_*.py
│
├── analysis/                       # ✅ 已完成（2个文件）
│   ├── error_correlation_appendix_a1.py
│   └── hyperparameter_sensitivity_appendix_d.py
│
├── data/                           # ⚠️ 部分完成
│   ├── README.md                   # 📝 待创建
│   ├── educational/                # ✅ 已有
│   ├── coco/                       # 📝 待下载
│   ├── RSITMD/                     # 📝 待下载
│   └── flickr8k/                   # 📝 待下载
│
└── results/                        # ⚠️ 部分完成
    ├── baseline_*.json             # ✅ 已有部分
    └── [实验结果将保存在此]
```

---

## 💡 执行建议

### 立即执行（今天）
1. 迁移P0优先级文件（6个核心文件）
2. 创建 `config.py` 和 `requirements.txt`
3. 测试基本导入是否正常

### 本周完成
4. 迁移P1优先级文件（下载脚本+扩展工具）
5. 更新所有实验脚本的导入路径
6. 运行 `quick_test.sh` 验证基线方法
7. 编写 `data/README.md` 数据下载指南

### 可选（时间允许）
8. 迁移诊断工具到 `tools/`
9. 添加结果分析脚本
10. 完善文档和示例

---

## 📞 遇到问题？

### 常见问题排查

**Q1: 导入错误 `ModuleNotFoundError: No module named 'dataloaders'`**
```bash
# 解决方案：检查目录结构和__init__.py
ls -la code_data/dataloaders/
touch code_data/dataloaders/__init__.py
```

**Q2: 数据路径找不到**
```bash
# 解决方案：使用config.py中的路径
python3 -c "import config; print(config.DATA_ROOT)"
```

**Q3: CLIP模型加载失败**
```bash
# 解决方案：检查网络连接，或使用本地缓存
export TORCH_HOME=/path/to/model/cache
```

---

## ✅ 验收标准

迁移完成后，应该能够：

1. **一键运行所有实验**：
   ```bash
   bash run_all_experiments.sh
   ```

2. **独立运行单个实验**：
   ```bash
   python3 experiments/coco_table1_main_results.py
   ```

3. **快速测试基线**：
   ```bash
   cd baselines && bash quick_test.sh
   ```

4. **数据加载正常**：
   ```python
   from dataloaders.data_loader import ImageCaptionDataset
   dataset = ImageCaptionDataset(data_root="data/RSITMD", split="test")
   print(f"✅ Loaded {len(dataset)} samples")
   ```

---

**生成时间**: 2026-09-27  
**文档版本**: 1.0  
**状态**: 待执行
