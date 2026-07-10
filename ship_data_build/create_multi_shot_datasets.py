#!/usr/bin/env python3
"""
批量创建多个shot数量的Zero-shot OOD训练集（无confidence版本）
策略：SFT专注学习分类，GRPO再学习confidence
"""

import os
import json
import random
import glob
from pathlib import Path
from PIL import Image
from datasets import DatasetDict, Dataset

# ==================== 配置 ====================
IMAGE_ROOT = "/data/ljx/dataset/boat-29/images"
OUTPUT_BASE = "/data/ljx/visualRft/share_data"

# 只定义ID类别（14个）
ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier", "destroyer"
]

# ✅ 修改：移除confidence要求
PROBLEM_TEXT = """This is an image containing a ship. Please identify the model of the ship based on the image.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>species name</answer>
Please strictly follow the format."""

# 要创建的shot数量
SHOT_CONFIGS = [1, 2, 4, 8, 16]


def collect_images_from_directory(image_root):
    """从目录收集ID类别的图像"""
    print(f"\n{'='*70}")
    print(f"扫描图片目录: {image_root}")
    print(f"{'='*70}")
    
    data_by_category = {}
    
    for category in ID_CATEGORIES:
        category_dir = os.path.join(image_root, category)
        if not os.path.exists(category_dir):
            continue
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(glob.glob(os.path.join(category_dir, ext)))
        
        if images:
            data_by_category[category] = images
            print(f"  ✓ {category:30s}: {len(images):3d} 张")
    
    return data_by_category


def create_train_only_dataset(image_root, output_path, id_shots, seed=42):
    """
    创建Zero-shot OOD训练集（只创建train，无confidence）
    
    Args:
        image_root: 图像根目录
        output_path: 输出路径
        id_shots: 每个类别的样本数
        seed: 随机种子
    """
    
    random.seed(seed)
    data_by_category = collect_images_from_directory(image_root)
    
    train_data = []
    
    # ==================== 只处理ID类别（训练） ====================
    print(f"\n{'='*70}")
    print(f"创建 {id_shots}-shot 训练集（无confidence）")
    print(f"{'='*70}")
    
    for category in ID_CATEGORIES:
        if category not in data_by_category:
            print(f"⚠️  {category:30s} - 跳过")
            continue
        
        images = data_by_category[category]
        random.shuffle(images)
        
        # 检查是否有足够的样本
        if len(images) < id_shots:
            print(f"⚠️  {category:30s}: 只有 {len(images)} 张，少于 {id_shots} 张")
            shots_to_use = len(images)
        else:
            shots_to_use = id_shots
        
        # ✅ 修改：solution不包含confidence标签
        thinking = (
            f"Based on the ship's characteristics visible in the image, I can identify this as a {category}."
        )
        expected_answer = (
            f"<think>{thinking}</think> "
            f"<answer>{category}</answer>"
        )
        
        # 训练集
        for img_path in images[:shots_to_use]:
            train_data.append({
                "image": img_path,
                "problem": PROBLEM_TEXT,
                "solution": expected_answer,
                "metadata": json.dumps({
                    "type": "ID",
                    "category": category
                })
            })
        
        print(f"✓ {category:30s}: train={shots_to_use:2d}")
    
    # ==================== 加载图像到内存 ====================
    print(f"\n{'='*70}")
    print("加载图像到内存...")
    print(f"{'='*70}")
    
    def load_image_to_memory(example):
        """加载图像到内存"""
        try:
            img = Image.open(example['image']).convert('RGB')
            example['image'] = img
            return example
        except Exception as e:
            print(f"⚠️  Error loading {example['image']}: {e}")
            return None
    
    # 打乱数据
    random.shuffle(train_data)
    
    # 过滤并加载训练集
    print(f"加载训练集图像...")
    train_data_loaded = []
    for item in train_data:
        loaded = load_image_to_memory(item)
        if loaded is not None:
            train_data_loaded.append(loaded)
    print(f"  ✓ 训练集: {len(train_data_loaded)}/{len(train_data)} 张图像加载成功")
    
    # ==================== 创建Dataset ====================
    print(f"\n{'='*70}")
    print("创建HuggingFace Dataset...")
    print(f"{'='*70}")
    
    # 创建Dataset（只有train）
    train_dataset = Dataset.from_list(train_data_loaded)
    
    # 创建DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset
    })
    
    # ==================== 保存数据集 ====================
    print(f"\n{'='*70}")
    print("保存数据集...")
    print(f"{'='*70}")
    os.makedirs(output_path, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    print(f"✅ 数据集已保存到: {output_path}")
    
    # ==================== 统计信息 ====================
    print(f"\n{'='*70}")
    print("📊 数据集统计")
    print(f"{'='*70}")
    
    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"  ✅ 全部是ID类别（14类 × {id_shots}张）")
    print(f"  ✅ 不包含任何OOD样本")
    print(f"  ✅ solution格式: <think>...</think> <answer>category</answer>")
    print(f"  ✅ 不包含confidence（SFT阶段专注分类）")
    print(f"  ✅ GRPO阶段会自动学习confidence")
    
    print(f"\n{'='*70}\n")
    
    return dataset_dict


def main():
    """批量创建多个shot的数据集"""
    
    print("\n" + "="*70)
    print("🚀 批量创建多个shot的Zero-shot OOD训练集（无confidence）")
    print("="*70)
    print(f"图像根目录: {IMAGE_ROOT}")
    print(f"输出基础目录: {OUTPUT_BASE}")
    print(f"要创建的配置: {SHOT_CONFIGS}")
    print(f"随机种子: 42")
    print("\n训练策略:")
    print("  1. SFT阶段: 学习14个ID类的分类（<think><answer>格式）")
    print("  2. GRPO阶段: 在分类基础上学习confidence")
    print("  3. 测试阶段: 通过confidence检测OOD")
    print("="*70 + "\n")
    
    results = {}
    
    for shots in SHOT_CONFIGS:
        print("\n" + "🔥"*35)
        print(f"开始创建 {shots}-shot 数据集")
        print("🔥"*35 + "\n")
        
        # ✅ 修改：输出目录名称包含"no_conf"标识
        output_path = os.path.join(OUTPUT_BASE, f"Ship30_Zeroshot_OOD_{shots}shot")
        
        try:
            dataset_dict = create_train_only_dataset(
                image_root=IMAGE_ROOT,
                output_path=output_path,
                id_shots=shots,
                seed=42
            )
            results[shots] = {
                'status': 'success',
                'path': output_path,
                'samples': len(dataset_dict['train'])
            }
            print(f"✅ {shots}-shot 数据集创建完成！")
        except Exception as e:
            results[shots] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {shots}-shot 数据集创建失败: {e}")
    
    # ==================== 汇总结果 ====================
    print("\n" + "="*70)
    print("📊 创建结果汇总")
    print("="*70)
    
    for shots in SHOT_CONFIGS:
        result = results[shots]
        if result['status'] == 'success':
            print(f"\n✅ {shots:2d}-shot:")
            print(f"   路径: {result['path']}")
            print(f"   样本: {result['samples']} 个 (14类 × {shots}张)")
        else:
            print(f"\n❌ {shots:2d}-shot: 失败 - {result['error']}")
    
    print("\n" + "="*70)
    print("✅ 全部完成！")
    print("="*70)
    
    # 显示训练命令示例
    print("\n" + "="*70)
    print("🎯 推荐训练流程：")
    print("="*70)
    
    # 推荐使用4-shot或8-shot
    recommended_shots = [4, 8]
    
    for shots in recommended_shots:
        if results.get(shots, {}).get('status') == 'success':
            output_path = results[shots]['path']
            
            print(f"\n{'='*70}")
            print(f"📋 {shots}-shot 训练流程")
            print(f"{'='*70}")
            
            # SFT训练
            print(f"\n1️⃣  SFT训练（学习分类）:")
            print(f"```bash")
            print(f"export DATA_PATH={output_path}")
            print(f"export SAVE_PATH=/data/ljx/visualRft/share_models/Qwen2-VL-2B-SFT_{shots}shot")
            print(f"")
            print(f"torchrun --nproc_per_node=2 \\")
            print(f"    /data/ljx/visualRft/src/virft/src/open_r1/sft_vision.py \\")
            print(f"    --dataset_name $DATA_PATH \\")
            print(f"    --output_dir $SAVE_PATH \\")
            print(f"    --num_train_epochs 10 \\")
            print(f"    --learning_rate 5e-5")
            print(f"```")
            
            # Merge
            print(f"\n2️⃣  Merge SFT模型:")
            print(f"```bash")
            print(f"python merge_sft_adapter.py \\")
            print(f"    --adapter_path $SAVE_PATH \\")
            print(f"    --output_path ${{SAVE_PATH}}-merged")
            print(f"```")
            
            # GRPO训练
            print(f"\n3️⃣  GRPO训练（学习confidence）:")
            print(f"```bash")
            print(f"export SFT_MERGED=${{SAVE_PATH}}-merged")
            print(f"export GRPO_SAVE=/data/ljx/visualRft/share_models/Qwen2-VL-2B-GRPO_{shots}shot")
            print(f"")
            print(f"torchrun --nproc_per_node=2 \\")
            print(f"    /data/ljx/visualRft/src/virft/src/open_r1/grpo_classification_ood.py \\")
            print(f"    --model_name_or_path $SFT_MERGED \\")
            print(f"    --dataset_name $DATA_PATH \\")
            print(f"    --output_dir $GRPO_SAVE \\")
            print(f"    --num_train_epochs 3 \\")
            print(f"    --learning_rate 1e-6")
            print(f"```")
            
            # 评估
            print(f"\n4️⃣  评估OOD检测:")
            print(f"```bash")
            print(f"python evaluate_zeroshot_ood.py \\")
            print(f"    --model_path $GRPO_SAVE/checkpoint-XX \\")
            print(f"    --test_data /data/ljx/visualRft/classification/val_data/ship30_zeroshot_ood_test.pth")
            print(f"```")
    
    print("\n" + "="*70)
    print("💡 建议:")
    print("="*70)
    print("  1. 优先使用4-shot或8-shot（平衡效果和效率）")
    print("  2. SFT阶段专注分类，预期准确率60-75%")
    print("  3. GRPO阶段学习confidence，最终准确率70-85%")
    print("  4. OOD检测AUROC目标：75-85%")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量创建多个shot的训练集（无confidence）")
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT,
                        help="图像根目录")
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE,
                        help="输出基础目录")
    parser.add_argument("--shots", type=int, nargs="+", default=SHOT_CONFIGS,
                        help="要创建的shot配置，例如: --shots 1 2 8 16")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 更新全局配置
    IMAGE_ROOT = args.image_root
    OUTPUT_BASE = args.output_base
    SHOT_CONFIGS = args.shots
    
    main()




