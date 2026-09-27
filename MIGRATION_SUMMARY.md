# 代码迁移摘要 - 快速参考

**生成时间**: 2026-09-27  
**状态**: 待执行

---

## 🚀 快速开始

### 一键迁移（推荐）
```bash
cd /Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI-ACL/acl/code_data
bash migrate_files.sh
```

### 验证迁移
```bash
bash test_migration.sh
```

---

## 📊 迁移统计

### 现有文件（已完成）
- ✅ **核心算法**: 3个 (`core/`)
- ✅ **主实验**: 2个 (`experiments/`)
- ✅ **基线方法**: 14个 (`baselines/`)
- ✅ **教育数据集**: 5个 (`eduppt/`)
- ✅ **分析脚本**: 2个 (`analysis/`)

### 待迁移文件
- 🔴 **P0 (必须)**: 6个核心文件
  - 3个数据加载器
  - 1个特征提取
  - 1个评估工具
  - 1个配置文件

- 🟡 **P1 (推荐)**: ~10个扩展文件
  - 4个下载脚本
  - 2个预处理脚本
  - 1个评估扩展

- 🟢 **P2 (可选)**: ~15个辅助工具
  - 诊断工具
  - 数据处理
  - 结果分析

---

## 📁 迁移后目录结构

```
code_data/
├── config.py                    # 📝 新增 - 统一配置
├── requirements.txt             # 📝 新增 - 依赖管理
├── migrate_files.sh             # 📝 新增 - 迁移脚本
├── test_migration.sh            # 📝 新增 - 测试脚本
│
├── dataloaders/                 # 📝 新增目录
│   ├── data_loader.py           # 通用数据加载器
│   ├── caption_dataset.py       # Caption数据集
│   └── dataset_rsicd.py         # RSICD数据集
│
├── preprocessing/               # 📝 新增目录
│   ├── extract_coco_karpathy_clip_features.py
│   └── precompute_clip_features.py
│
├── utils/                       # 📝 新增目录
│   ├── eval_retrieval.py        # 检索评估
│   └── eval_clip_standard.py    # CLIP标准评估
│
├── scripts/                     # 📝 新增目录
│   ├── download_coco_karpathy.py
│   ├── download_rsitmd.py
│   └── download_flickr8k.py
│
├── tools/                       # 📝 新增目录（可选）
│   ├── check_*.py              # 检查工具
│   ├── verify_*.py             # 验证工具
│   └── diagnose_*.py           # 诊断工具
│
├── core/                        # ✅ 已有
├── experiments/                 # ✅ 已有
├── baselines/                   # ✅ 已有
├── eduppt/                      # ✅ 已有
├── analysis/                    # ✅ 已有
├── data/                        # ⚠️ 需要补充
│   └── README.md               # 📝 新增 - 数据下载指南
└── results/                     # ⚠️ 需要补充
```

---

## ✅ 完成后检查清单

### 1. 文件迁移
- [ ] P0优先级文件全部迁移（6个）
- [ ] P1优先级文件迁移（推荐10个）
- [ ] 配置文件创建完成（config.py, requirements.txt）
- [ ] 文档文件创建完成（data/README.md）

### 2. 代码修改
- [ ] 更新 `experiments/coco_table1_main_results.py` 导入路径
- [ ] 更新 `experiments/rsitmd_table2_cross_domain.py` 导入路径
- [ ] 更新 `baselines/*.py` 导入路径（如需要）

### 3. 功能测试
- [ ] 运行 `test_migration.sh` 通过
- [ ] Python导入测试通过
- [ ] 快速测试 `baselines/quick_test.sh` 可运行

### 4. 数据准备
- [ ] 阅读 `data/README.md`
- [ ] 下载COCO数据集（必需）
- [ ] 下载RSITMD数据集（必需）
- [ ] 预计算特征（推荐）

### 5. 依赖安装
- [ ] 运行 `pip install -r requirements.txt`
- [ ] 验证CLIP模型可以加载

---

## 🔧 常见问题

### Q1: migrate_files.sh 执行失败
**原因**: 文件权限不足  
**解决**: `chmod +x migrate_files.sh`

### Q2: 导入错误 ModuleNotFoundError
**原因**: Python路径问题  
**解决**: 
```python
import sys
sys.path.insert(0, '/path/to/code_data')
```

### Q3: 数据路径找不到
**原因**: 未使用config.py  
**解决**: 
```python
import config
data_path = config.COCO_DIR
```

### Q4: CLIP模型下载失败
**原因**: 网络问题  
**解决**: 设置镜像或使用缓存
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📞 需要帮助？

1. **查看详细清单**: `MIGRATION_CHECKLIST.md`
2. **查看项目README**: `README.md`
3. **运行测试**: `bash test_migration.sh`
4. **检查导入**: `python3 -c "import config; print(config.PROJECT_ROOT)"`

---

## 🎯 下一步行动

### 今天完成
1. ✅ 运行迁移脚本: `bash migrate_files.sh`
2. ✅ 验证迁移: `bash test_migration.sh`
3. ✅ 安装依赖: `pip install -r requirements.txt`

### 本周完成
4. ⚠️ 更新导入路径（手动编辑2-3个文件）
5. ⚠️ 下载COCO数据集（~13GB）
6. ⚠️ 下载RSITMD数据集（~2GB）
7. ✅ 运行快速测试: `cd baselines && bash quick_test.sh`

### 可选任务
8. 🟢 预计算CLIP特征（加速实验）
9. 🟢 迁移P2优先级工具
10. 🟢 完善文档和示例

---

**准备好了吗？运行迁移脚本开始吧！**

```bash
bash migrate_files.sh
```
