# 代码迁移文档索引

欢迎查看ACE项目代码迁移文档。本目录包含从workspace到code_data的完整迁移指南。

---

## 📚 文档列表

### 1. 快速开始
- **文件**: `MIGRATION_SUMMARY.md`
- **用途**: 快速参考，包含一键迁移命令
- **适合**: 想要快速完成迁移的用户
- **阅读时间**: 3分钟

### 2. 详细清单
- **文件**: `MIGRATION_CHECKLIST.md`
- **用途**: 完整的迁移清单，包含所有文件分类和优先级
- **适合**: 需要了解每个文件作用的用户
- **阅读时间**: 15分钟

### 3. 自动化脚本
- **文件**: `migrate_files.sh`
- **用途**: 自动执行文件迁移
- **适合**: 所有用户（推荐使用）
- **执行时间**: 2-5分钟

### 4. 验证脚本
- **文件**: `test_migration.sh` (由migrate_files.sh自动生成)
- **用途**: 验证迁移是否成功
- **适合**: 迁移完成后使用
- **执行时间**: 1分钟

---

## 🚀 推荐流程

### 第一次迁移（完整流程）

```bash
# 1. 查看快速摘要（了解整体情况）
cat MIGRATION_SUMMARY.md

# 2. 运行迁移脚本（自动化执行）
chmod +x migrate_files.sh
bash migrate_files.sh

# 3. 验证迁移结果
bash test_migration.sh

# 4. 安装依赖
pip install -r requirements.txt

# 5. 查看详细清单（可选，了解细节）
cat MIGRATION_CHECKLIST.md
```

### 快速迁移（只看摘要）

```bash
# 直接运行迁移脚本
bash migrate_files.sh

# 验证结果
bash test_migration.sh
```

---

## 📊 迁移进度追踪

你可以使用以下命令查看迁移状态：

```bash
# 查看已迁移的文件数量
echo "Dataloaders: $(ls dataloaders/*.py 2>/dev/null | wc -l)"
echo "Preprocessing: $(ls preprocessing/*.py 2>/dev/null | wc -l)"
echo "Utils: $(ls utils/*.py 2>/dev/null | wc -l)"
echo "Scripts: $(ls scripts/*.py 2>/dev/null | wc -l)"
echo "Tools: $(ls tools/*.py 2>/dev/null | wc -l)"

# 或者运行测试脚本
bash test_migration.sh
```

---

## 🎯 核心文件清单（P0优先级）

迁移完成后，确保以下核心文件存在：

### 必需文件（6个）
- [ ] `dataloaders/data_loader.py`
- [ ] `dataloaders/caption_dataset.py`
- [ ] `dataloaders/dataset_rsicd.py`
- [ ] `preprocessing/extract_coco_karpathy_clip_features.py`
- [ ] `utils/eval_retrieval.py` (如果存在于workspace)
- [ ] `config.py`

### 必需配置（2个）
- [ ] `requirements.txt`
- [ ] `data/README.md`

---

## ⚠️ 重要提示

### 迁移前
1. **备份workspace目录**（建议）
2. **检查磁盘空间**：至少需要500MB用于迁移文件
3. **确认Python环境**：Python 3.8+

### 迁移后
1. **手动更新导入路径**：修改experiments/和baselines/中的导入语句
2. **准备数据集**：查看data/README.md获取下载指南
3. **测试运行**：执行baselines/quick_test.sh验证功能

### 常见错误
- **路径不存在**：workspace目录路径可能需要调整（编辑migrate_files.sh的第13-14行）
- **权限不足**：运行`chmod +x migrate_files.sh`添加执行权限
- **导入失败**：检查Python路径设置，使用`sys.path.insert(0, '.')`

---

## 📞 获取帮助

### 问题排查顺序
1. 运行 `test_migration.sh` 查看具体错误
2. 查看 `MIGRATION_CHECKLIST.md` 了解文件作用
3. 检查 `migrate_files.sh` 输出日志
4. 查看 Python import 错误信息

### 文档链接
- 项目主README: `README.md`
- 数据准备指南: `data/README.md`
- 基线方法说明: `baselines/README_BASELINE.md`

---

## 📈 迁移后的目录结构预览

```
code_data/
├── 📄 README.md                 # 项目主文档
├── 📄 MIGRATION_SUMMARY.md      # 快速摘要（本文件的详细版）
├── 📄 MIGRATION_CHECKLIST.md    # 详细清单
├── 📄 MIGRATION_INDEX.md        # 本文件
├── 📄 config.py                 # 配置文件
├── 📄 requirements.txt          # 依赖文件
├── 🔧 migrate_files.sh          # 迁移脚本
├── 🔧 test_migration.sh         # 测试脚本
├── 🔧 run_all_experiments.sh    # 运行所有实验
│
├── 📁 core/                     # ✅ 核心算法（3个文件）
├── 📁 dataloaders/              # 📝 数据加载（3个文件）
├── 📁 preprocessing/            # 📝 预处理（1-3个文件）
├── 📁 utils/                    # 📝 工具函数（1-2个文件）
├── 📁 scripts/                  # 📝 下载脚本（4个文件）
├── 📁 tools/                    # 📝 辅助工具（可选）
│
├── 📁 experiments/              # ✅ 主实验（2个文件）
├── 📁 baselines/                # ✅ 基线方法（14个文件）
├── 📁 eduppt/                   # ✅ 教育数据集（5个文件）
├── 📁 analysis/                 # ✅ 分析脚本（2个文件）
│
├── 📁 data/                     # 数据目录
│   ├── 📄 README.md            # 数据下载指南
│   ├── 📁 educational/         # ✅ 教育数据集
│   ├── 📁 coco/                # 待下载
│   ├── 📁 RSITMD/              # 待下载
│   └── 📁 features/            # 特征缓存
│
└── 📁 results/                  # 结果目录
    └── 📁 raw/                 # 原始结果文件
```

---

## ✅ 完成确认

迁移完成后，你应该能够：

1. ✅ 导入核心模块：`from dataloaders.data_loader import ImageCaptionDataset`
2. ✅ 使用配置文件：`import config; print(config.DATA_ROOT)`
3. ✅ 运行快速测试：`bash baselines/quick_test.sh`
4. ✅ 执行主实验：`python3 experiments/coco_table1_main_results.py`

如果以上4项都能成功，恭喜！迁移已完成。

---

**开始迁移吧！祝顺利！🚀**

```bash
bash migrate_files.sh
```
