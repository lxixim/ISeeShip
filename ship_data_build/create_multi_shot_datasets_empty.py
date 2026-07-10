#!/usr/bin/env python3
"""
批量创建多个shot数量的训练集（带空confidence和judge标签）
策略：标签占位但不指定值，让GRPO自己学习
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

# ✅ 简化：只要求confidence（空标签占位）
PROBLEM_TEXT = """This is an image containing a ship. Please identify the model of the ship based on the image.

Output format (MUST follow):
<confidence></confidence> <think>your reasoning</think> <answer>species name</answer>

- <confidence>: Your confidence level (0-100%)
- <think>: Your reasoning process
- <answer>: The species name

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
    创建训练集（带空confidence和judge标签）
    
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
    print(f"创建 {id_shots}-shot 训练集（空标签占位）")
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
        
        # ✅ 简化：solution只包含空confidence标签
        thinking = (
            f"Based on the ship's characteristics visible in the image, I can identify this as a {category}."
        )
        expected_answer = (
            f"<confidence></confidence> "  # ⭐ 空标签占位
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
    print(f"  ✅ solution格式: <confidence></confidence> <think>...</think> <answer>...</answer>")
    print(f"  ✅ confidence是空标签（占位但不指定值）")
    print(f"  ✅ 模型会学会输出confidence，但通过GRPO学习正确的值")
    
    print(f"\n💡 为什么这样设计:")
    print(f"  1. 空标签让模型学会格式（不会忘记输出）")
    print(f"  2. 不指定值让模型通过GRPO探索（不会记住固定值）")
    print(f"  3. 推理时用真实标签判断ID/OOD，只需要模型的confidence")
    
    print(f"\n{'='*70}\n")
    
    return dataset_dict


def main():
    """批量创建多个shot的数据集"""
    
    print("\n" + "="*70)
    print("🚀 批量创建训练集（带空confidence标签）")
    print("="*70)
    print(f"图像根目录: {IMAGE_ROOT}")
    print(f"输出基础目录: {OUTPUT_BASE}")
    print(f"要创建的配置: {SHOT_CONFIGS}")
    print(f"随机种子: 42")
    print("\n训练策略:")
    print("  1. 数据格式: <confidence></confidence> <think>...</think> <answer>...</answer>")
    print("  2. 空标签占位: 模型学会格式，但不记住固定值")
    print("  3. GRPO学习: 通过奖励函数学习正确的confidence值")
    print("  4. 推理时: 用真实标签判断ID/OOD，只看模型confidence")
    print("="*70 + "\n")
    
    results = {}
    
    for shots in SHOT_CONFIGS:
        print("\n" + "🔥"*35)
        print(f"开始创建 {shots}-shot 数据集")
        print("🔥"*35 + "\n")
        
        # 输出目录名称
        output_path = os.path.join(OUTPUT_BASE, f"Ship30_Empty_Tags_{shots}shot")
        
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
            print(f"   格式: <confidence></confidence> <think>...</think> <answer>...</answer>")
        else:
            print(f"\n❌ {shots:2d}-shot: 失败 - {result['error']}")
    
    print("\n" + "="*70)
    print("✅ 全部完成！")
    print("="*70)
    
    # 显示训练命令示例
    print("\n" + "="*70)
    print("🎯 推荐训练流程：")
    print("="*70)
    
    recommended_shots = [4, 8]
    
    for shots in recommended_shots:
        if results.get(shots, {}).get('status') == 'success':
            output_path = results[shots]['path']
            
            print(f"\n{'='*70}")
            print(f"📋 {shots}-shot 训练流程")
            print(f"{'='*70}")
            
            # SFT训练
            print(f"\n1️⃣  SFT训练:")
            print(f"```bash")
            print(f"export DATA_PATH={output_path}")
            print(f"export SAVE_PATH=/data/ljx/visualRft/share_models/Qwen2-VL-2B-SFT_{shots}shot")
            print(f"")
            print(f"torchrun --nproc_per_node=2 \\")
            print(f"    train_sft.py \\")
            print(f"    --dataset_name $DATA_PATH \\")
            print(f"    --output_dir $SAVE_PATH \\")
            print(f"    --num_train_epochs 10")
            print(f"```")
            
            print(f"\n  预期: 模型学会输出<confidence>标签（但值是随机的）")
            
            # Merge
            print(f"\n2️⃣  Merge SFT:")
            print(f"```bash")
            print(f"python merge_sft.py --adapter $SAVE_PATH --output ${{SAVE_PATH}}-merged")
            print(f"```")
            
            # GRPO训练
            print(f"\n3️⃣  GRPO训练（学习正确的confidence值）:")
            print(f"```bash")
            print(f"export GRPO_SAVE=/data/ljx/visualRft/share_models/Qwen2-VL-2B-GRPO_{shots}shot")
            print(f"")
            print(f"python train_grpo_simple.py \\")
            print(f"    --model ${{SAVE_PATH}}-merged \\")
            print(f"    --dataset $DATA_PATH \\")
            print(f"    --output $GRPO_SAVE")
            print(f"```")
            
            print(f"\n  预期:")
            print(f"    - ID样本: <confidence>80-95%</confidence>")
            print(f"    - 模型学会: 确定时高confidence，不确定时低confidence")
            
            # 评估
            print(f"\n4️⃣  评估:")
            print(f"```bash")
            print(f"python your_inference_script.py")
            print(f"```")
            
            print(f"\n  预期:")
            print(f"    - ID分类: 70%+")
            print(f"    - OOD检测: 70%+")
            print(f"    - Confidence差异: 0.35+")
    
    print("\n" + "="*70)
    print("💡 关键设计说明:")
    print("="*70)
    print("\n1. 为什么用空标签？")
    print("   - 模型看到标签格式 → 学会输出")
    print("   - 但标签是空的 → 不会记住固定值")
    print("   - 通过GRPO奖励 → 学习正确的值")
    print("\n2. 推理时如何检测OOD？")
    print("   - 用真实标签判断是ID还是OOD")
    print("   - 只需要模型的confidence")
    print("   - ID期望高conf，OOD期望低conf")
    print("\n3. 计算OOD score:")
    print("   ```python")
    print("   # 方法1: 直接用confidence")
    print("   if 真实是ID:")
    print("       ood_score = 1 - conf  # conf高→score低→不是OOD")
    print("   else:")
    print("       ood_score = conf      # conf低→score低，但真实是OOD")
    print("")
    print("   # 方法2: 简化版（推荐）")
    print("   # 用阈值检测: conf < 0.5 → OOD")
    print("   # AUROC用confidence和真实ID/OOD标签计算")
    print("   ```")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量创建训练集（空标签）")
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE)
    parser.add_argument("--shots", type=int, nargs="+", default=SHOT_CONFIGS)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    IMAGE_ROOT = args.image_root
    OUTPUT_BASE = args.output_base
    SHOT_CONFIGS = args.shots
    
    main()