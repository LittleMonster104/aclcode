#!/bin/bash
# 每个学科目录只保留10张图片

IMAGES_DIR="/Users/jiazhu/Documents/ZJNU/EvoScientist/AAAI-ACL/acl/code_data/data/educational/images"

cd "$IMAGES_DIR"

echo "========================================="
echo "清理教育图片目录 - 每个学科保留10张"
echo "========================================="
echo ""

total_deleted=0
total_kept=0

# 遍历所有学科目录
for subject_dir in */; do
    if [ -d "$subject_dir" ]; then
        subject_name="${subject_dir%/}"
        
        # 统计当前目录中的图片数量
        image_count=$(find "$subject_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l | tr -d ' ')
        
        if [ "$image_count" -gt 10 ]; then
            echo "📁 处理: $subject_name"
            echo "   原图片数: $image_count"
            
            # 保留前10张，删除其余
            kept=0
            deleted=0
            
            find "$subject_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | while read img; do
                if [ $kept -lt 10 ]; then
                    kept=$((kept + 1))
                else
                    rm "$img"
                    deleted=$((deleted + 1))
                fi
            done
            
            # 重新统计
            new_count=$(find "$subject_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l | tr -d ' ')
            deleted_count=$((image_count - new_count))
            
            echo "   删除数量: $deleted_count"
            echo "   保留数量: $new_count"
            echo ""
            
            total_deleted=$((total_deleted + deleted_count))
            total_kept=$((total_kept + new_count))
        else
            echo "✓ 跳过: $subject_name (只有 $image_count 张图片)"
            total_kept=$((total_kept + image_count))
        fi
    fi
done

echo "========================================="
echo "清理完成！"
echo "========================================="
echo "总共删除: $total_deleted 张图片"
echo "总共保留: $total_kept 张图片"
echo ""
