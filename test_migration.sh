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
