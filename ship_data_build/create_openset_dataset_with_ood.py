#!/usr/bin/env python3
"""
批量创建开集训练集 - 纯训练集版本

核心要点：
✅ 只创建训练集（测试集已有，使用全部30类）
✅ Solution只包含<answer>标签（<think>由模型生成）
✅ 16个OOD类 = 10训练 + 6保留用于泛化测试
✅ OOD训练集涵盖Easy/Medium/Hard难度梯度
✅ 保守OOD比例（20-25%），优先ID分类准确率
"""

import os
import json
import random
import glob
from pathlib import Path
from PIL import Image
from datasets import DatasetDict, Dataset
from collections import defaultdict

# ==================== 配置 ====================
IMAGE_ROOT = "/data/ljx/dataset/boat-29/images"
OUTPUT_BASE = "/data/ljx/visualRft/share_data"

# ID类别（14个）
ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier", "destroyer"
]

# 🔥 OOD训练类别（10个 - 涵盖Easy/Medium/Hard难度）
OOD_TRAIN_CATEGORIES = [
    # Easy难度（明显不同）
    "aircraft_carrier",      # 航母
    "frigate",               # 护卫舰
    "submarine",             # 潜艇
    "firefighting",          # 消防船
    "tugboat",               # 拖船
    
    # Medium难度（中等相似）
    "reefer",                # 冷藏船
    "vehicles_carrier",      # 汽车运输船
    "passenger_ro-ro_ship",  # 客滚船
    
    # Hard难度（细粒度相似）
    "fso",                   # 浮式储油船
    "cruise",                # 游轮
]

# 🔥 OOD保留类别（6个 - 不用于训练，仅用于测试时评估泛化）
OOD_RESERVED_CATEGORIES = [
    "bitumen",               # 沥青船（极似oil_products_tanker）
    "passenger_ship",        # 客船（类似passenger_cargo_ship）
    "catamaran_yacht",       # 双体游艇（类似sailing_trimaran）
    "monohull_yacht",        # 单体游艇
    "monohull_sailboat",     # 单体帆船
    "sailing_catamaran",     # 帆船双体船
]

# 全部16类OOD（验证用）
ALL_OOD_CATEGORIES = [
    "aircraft_carrier", "bitumen", "catamaran_yacht", "cruise",
    "firefighting", "frigate", "fso", "monohull_sailboat",
    "monohull_yacht", "passenger_ro-ro_ship", "passenger_ship",
    "reefer", "sailing_catamaran", "submarine", "tugboat", "vehicles_carrier"
]

# 验证划分完整性
assert len(OOD_TRAIN_CATEGORIES) == 10, "OOD训练类应为10个"
assert len(OOD_RESERVED_CATEGORIES) == 6, "OOD保留类应为6个"
assert set(OOD_TRAIN_CATEGORIES) | set(OOD_RESERVED_CATEGORIES) == set(ALL_OOD_CATEGORIES)
assert len(set(OOD_TRAIN_CATEGORIES) & set(OOD_RESERVED_CATEGORIES)) == 0

# 要创建的shot数量
SHOT_CONFIGS = [1, 2, 4, 8, 16]

# 固定的PROBLEM模板（与你的推理代码完全一致）
PROBLEM_TEXT = """This is an image containing a ship. Please identify the type of the ship based on the image.

You can ONLY output ONE of these exact answers:
bulk_carrier, container_ship, general_cargo_ship, passenger_cargo_ship,
fishing_boat, sailing_trimaran, chemical_tanker, crude_oil_tanker,
oil_products_tanker, LNG_tanker, LPG_tanker, kayak, heavy_load_carrier, destroyer, unknown

If the ship matches one of the 14 types → output that type
If the ship does NOT match ANY of the 14 types → output "unknown"

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>type or unknown</answer>
Please strictly follow the format."""



# ==================== OOD比例策略（保守版）====================
def get_ood_config(k_shot):
    """
    保守版OOD配置
    
    策略：优先保证ID分类准确率，同时学习OOD拒识能力
    """
    configs = {
        1: {
            'ood_ratio': 0.25,
            'min_ood_samples': 5,
            'min_ood_coverage': 0.5,  # 至少覆盖50%的OOD训练类
        },
        2: {
            'ood_ratio': 0.25,
            'min_ood_samples': 9,
            'min_ood_coverage': 0.7,  # 70%
        },
        4: {
            'ood_ratio': 0.40,
            'min_ood_samples': 30,
            'min_ood_coverage': 1.0,  # 100%全覆盖
        },
        8: {
            'ood_ratio': 0.35,
            'min_ood_samples': 40,
            'min_ood_coverage': 1.0,
        },
        16: {
            'ood_ratio': 0.30,
            'min_ood_samples': 68,
            'min_ood_coverage': 1.0,
        }
    }
    return configs.get(k_shot, {'ood_ratio': 0.20, 'min_ood_samples': 20, 'min_ood_coverage': 1.0})


# ==================== 核心函数 ====================
def collect_images_from_directory(image_root, categories):
    """从目录收集指定类别的图像"""
    print(f"\n{'='*70}")
    print(f"扫描图片目录: {image_root}")
    print(f"目标类别: {len(categories)} 个")
    print(f"{'='*70}")
    
    data_by_category = {}
    
    for category in categories:
        category_dir = os.path.join(image_root, category)
        if not os.path.exists(category_dir):
            print(f"  ⚠️  {category:30s}: 目录不存在")
            continue
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(glob.glob(os.path.join(category_dir, ext)))
        
        if images:
            data_by_category[category] = images
            print(f"  ✓ {category:30s}: {len(images):3d} 张")
        else:
            print(f"  ⚠️  {category:30s}: 0 张图片")
    
    return data_by_category


def sample_ood_balanced(ood_data_by_category, target_ood_num, ood_categories, min_coverage=1.0):
    """
    均衡采样OOD样本，确保类别覆盖
    
    策略：
    1. 第一轮：每个类至少1个样本
    2. 第二轮：按类别图片数量比例分配剩余配额
    """
    
    available_categories = [cat for cat in ood_categories 
                           if cat in ood_data_by_category and len(ood_data_by_category[cat]) > 0]
    
    min_coverage_num = int(len(ood_categories) * min_coverage)
    
    selected = []
    remaining_quota = target_ood_num
    
    # 第一轮：保证类别覆盖
    categories_to_cover = random.sample(available_categories, 
                                       min(min_coverage_num, len(available_categories)))
    
    for category in categories_to_cover:
        images = ood_data_by_category[category]
        sample = random.choice(images)
        selected.append((sample, category))
        remaining_quota -= 1
    
    # 第二轮：按比例分配剩余配额
    if remaining_quota > 0:
        total_images = sum(len(ood_data_by_category.get(cat, [])) 
                          for cat in available_categories)
        
        for category in available_categories:
            images = ood_data_by_category[category]
            category_weight = len(images) / total_images
            extra_samples = max(1, int(remaining_quota * category_weight))
            
            # 避免重复采样
            available = [img for img in images 
                        if (img, category) not in selected]
            
            if available:
                n_sample = min(extra_samples, len(available))
                extra = random.sample(available, n_sample)
                selected.extend([(img, category) for img in extra])
    
    return selected[:target_ood_num]


def create_openset_dataset(image_root, output_path, id_shots, seed=42):
    """
    创建开集训练数据集
    
    Args:
        image_root: 图像根目录
        output_path: 输出路径
        id_shots: 每个ID类别的样本数
        seed: 随机种子
    """
    
    random.seed(seed)
    
    # 获取OOD配置
    ood_config = get_ood_config(id_shots)
    ood_ratio = ood_config['ood_ratio']
    min_ood_samples = ood_config['min_ood_samples']
    min_ood_coverage = ood_config['min_ood_coverage']
    
    # 计算样本数
    num_id_total = len(ID_CATEGORIES) * id_shots
    num_ood_target = max(int(num_id_total * ood_ratio / (1 - ood_ratio)), min_ood_samples)
    
    print(f"\n{'='*70}")
    print(f"开集训练集配置 ({id_shots}-shot)")
    print(f"{'='*70}")
    print(f"ID类别数:            {len(ID_CATEGORIES)}")
    print(f"OOD训练类别数:       {len(OOD_TRAIN_CATEGORIES)} (用于学习拒识)")
    print(f"OOD保留类别数:       {len(OOD_RESERVED_CATEGORIES)} (不用于训练，测试时评估泛化)")
    print(f"ID样本数:            {num_id_total} ({len(ID_CATEGORIES)} × {id_shots})")
    print(f"OOD样本数:           {num_ood_target} ({100*ood_ratio:.0f}%)")
    print(f"OOD最小覆盖:         {min_ood_coverage*100:.0f}% ({int(len(OOD_TRAIN_CATEGORIES)*min_ood_coverage)}/{len(OOD_TRAIN_CATEGORIES)}类)")
    print(f"总样本数:            {num_id_total + num_ood_target}")
    print(f"{'='*70}\n")
    
    # 收集ID类别图片
    print("🔍 第1步: 收集ID类别图片")
    id_data_by_category = collect_images_from_directory(image_root, ID_CATEGORIES)
    
    # 🔥 只收集OOD训练类别图片（不包含保留类别）
    print("\n🔍 第2步: 收集OOD训练类别图片（不含保留类别）")
    ood_data_by_category = collect_images_from_directory(image_root, OOD_TRAIN_CATEGORIES)
    
    train_data = []
    
    # ==================== 处理ID样本 ====================
    print(f"\n{'='*70}")
    print(f"📦 第3步: 创建ID训练样本 ({id_shots}-shot)")
    print(f"{'='*70}")
    
    id_sample_count = 0
    for category in ID_CATEGORIES:
        if category not in id_data_by_category:
            print(f"⚠️  {category:30s} - 跳过（无图片）")
            continue
        
        images = id_data_by_category[category]
        random.shuffle(images)
        
        if len(images) < id_shots:
            print(f"⚠️  {category:30s}: 只有 {len(images)} 张，少于 {id_shots} 张")
            shots_to_use = len(images)
        else:
            shots_to_use = id_shots
        
        # 🔥 核心：solution只有<answer>标签
        solution = f"<answer>{category}</answer>"
        
        for img_path in images[:shots_to_use]:
            train_data.append({
                "image": img_path,
                "problem": PROBLEM_TEXT,
                "solution": solution,
                "metadata": json.dumps({
                    "type": "ID",
                    "category": category,
                    "original_category": category
                })
            })
            id_sample_count += 1
        
        print(f"✓ {category:30s}: {shots_to_use:2d} 样本")
    
    print(f"\n总计ID样本: {id_sample_count}")
    
    # ==================== 处理OOD训练样本 ====================
    print(f"\n{'='*70}")
    print(f"📦 第4步: 创建OOD训练样本 (目标{num_ood_target}个)")
    print(f"{'='*70}")
    
    # 均衡采样OOD（仅从训练类别）
    selected_ood = sample_ood_balanced(
        ood_data_by_category,
        num_ood_target,
        OOD_TRAIN_CATEGORIES,
        min_coverage=min_ood_coverage
    )
    
    print(f"成功采样: {len(selected_ood)} 个OOD训练样本")
    
    # 统计OOD样本分布
    ood_category_count = defaultdict(int)
    ood_difficulty_count = {"Easy": 0, "Medium": 0, "Hard": 0}
    
    # 难度映射
    difficulty_map = {
        "aircraft_carrier": "Easy", "frigate": "Easy", "submarine": "Easy",
        "firefighting": "Easy", "tugboat": "Easy",
        "reefer": "Medium", "vehicles_carrier": "Medium", "passenger_ro-ro_ship": "Medium",
        "fso": "Hard", "cruise": "Hard"
    }
    
    ood_sample_count = 0
    for img_path, original_category in selected_ood:
        # 🔥 核心：solution只有<answer>unknown</answer>
        # 模型会自己生成<think>部分
        solution = "<answer>unknown</answer>"
        
        difficulty = difficulty_map.get(original_category, "Unknown")
        ood_difficulty_count[difficulty] += 1
        
        train_data.append({
            "image": img_path,
            "problem": PROBLEM_TEXT,
            "solution": solution,
            "metadata": json.dumps({
                "type": "OOD",
                "category": "unknown",
                "original_category": original_category,
                "ood_split": "train",
                "difficulty": difficulty
            })
        })
        ood_sample_count += 1
        ood_category_count[original_category] += 1
    
    print(f"\n✅ 成功创建 {ood_sample_count} 个OOD训练样本")
    
    # 检查覆盖度
    covered_categories = len(ood_category_count)
    coverage_ratio = covered_categories / len(OOD_TRAIN_CATEGORIES)
    print(f"\nOOD训练类别覆盖度: {covered_categories}/{len(OOD_TRAIN_CATEGORIES)} ({100*coverage_ratio:.0f}%)")
    
    print(f"\nOOD训练样本按难度分布:")
    for difficulty in ["Easy", "Medium", "Hard"]:
        count = ood_difficulty_count[difficulty]
        ratio = count / ood_sample_count if ood_sample_count > 0 else 0
        print(f"  {difficulty:10s}: {count:2d} 个 ({100*ratio:.1f}%)")
    
    print(f"\nOOD训练样本按类别分布:")
    for cat in sorted(ood_category_count.keys(), key=lambda x: difficulty_map.get(x, "Z")):
        count = ood_category_count[cat]
        diff = difficulty_map.get(cat, "Unknown")
        print(f"  [{diff:6s}] {cat:30s}: {count:2d} 个")
    
    # ==================== 准备训练数据 ====================
    print(f"\n{'='*70}")
    print(f"📊 第5步: 准备训练数据")
    print(f"{'='*70}")
    
    # 打乱数据
    random.shuffle(train_data)
    
    # 统计
    id_samples = [s for s in train_data if json.loads(s['metadata'])['type'] == 'ID']
    ood_samples = [s for s in train_data if json.loads(s['metadata'])['type'] == 'OOD']
    
    print(f"训练集:")
    print(f"  ID样本:      {len(id_samples):3d} ({100*len(id_samples)/len(train_data):.1f}%)")
    print(f"  OOD样本:     {len(ood_samples):3d} ({100*len(ood_samples)/len(train_data):.1f}%)")
    print(f"  总计:        {len(train_data):3d}")
    print(f"\n⚠️  注意：")
    print(f"  • 所有数据用于训练")
    print(f"  • 测试集已有（全部30类）")
    print(f"  • 6个OOD保留类不在训练集中（用于评估泛化）")
    
    # ==================== 加载图像到内存 ====================
    print(f"\n{'='*70}")
    print("📥 第6步: 加载图像到内存")
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
    
    print(f"加载训练集图像...")
    train_data_loaded = []
    for item in train_data:
        loaded = load_image_to_memory(item)
        if loaded is not None:
            train_data_loaded.append(loaded)
    print(f"  ✓ 训练集: {len(train_data_loaded)}/{len(train_data)} 张图像加载成功")
    
    # ==================== 创建Dataset ====================
    print(f"\n{'='*70}")
    print("💾 第7步: 创建HuggingFace Dataset")
    print(f"{'='*70}")
    
    train_dataset = Dataset.from_list(train_data_loaded)
    dataset_dict = DatasetDict({'train': train_dataset})
    
    # ==================== 保存数据集 ====================
    print(f"\n{'='*70}")
    print("💾 第8步: 保存数据集")
    print(f"{'='*70}")
    
    os.makedirs(output_path, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    print(f"✅ 数据集已保存到: {output_path}")
    
    # 保存配置文件
    config = {
        "k_shot": id_shots,
        "strategy": "train_only_with_generalization_reserve",
        "num_id_classes": len(ID_CATEGORIES),
        "num_ood_train_classes": len(OOD_TRAIN_CATEGORIES),
        "num_ood_reserved_classes": len(OOD_RESERVED_CATEGORIES),
        "id_categories": ID_CATEGORIES,
        "ood_train_categories": OOD_TRAIN_CATEGORIES,
        "ood_reserved_categories": OOD_RESERVED_CATEGORIES,
        "ood_train_difficulty": difficulty_map,
        "num_id_samples": id_sample_count,
        "num_ood_samples": ood_sample_count,
        "ood_ratio": float(ood_ratio),
        "ood_coverage_actual": float(coverage_ratio),
        "ood_coverage_target": float(min_ood_coverage),
        "ood_difficulty_distribution": ood_difficulty_count,
        "train_samples": {
            "total": len(train_data_loaded),
            "id": len(id_samples),
            "ood": len(ood_samples)
        },
        "problem_template": PROBLEM_TEXT,
        "solution_format": "<answer>category/unknown</answer>",
        "note": "Training set only. Solution contains only <answer> tag. Model generates <think> during training. 6 OOD reserved classes for generalization testing."
    }
    
    config_file = os.path.join(output_path, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ 配置文件已保存到: {config_file}")
    
    # ==================== 统计信息 ====================
    print(f"\n{'='*70}")
    print("📊 数据集统计")
    print(f"{'='*70}")
    
    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"  ✅ ID样本: {len(id_samples)} ({100*len(id_samples)/len(train_dataset):.1f}%)")
    print(f"  ✅ OOD样本: {len(ood_samples)} ({100*len(ood_samples)/len(train_dataset):.1f}%)")
    
    print(f"\nOOD类别划分（16个 = 10训练 + 6保留）:")
    print(f"\n  🎓 训练类别({len(OOD_TRAIN_CATEGORIES)}个) - 学习拒识:")
    print(f"    [Easy]   {', '.join([c for c in OOD_TRAIN_CATEGORIES if difficulty_map.get(c) == 'Easy'])}")
    print(f"    [Medium] {', '.join([c for c in OOD_TRAIN_CATEGORIES if difficulty_map.get(c) == 'Medium'])}")
    print(f"    [Hard]   {', '.join([c for c in OOD_TRAIN_CATEGORIES if difficulty_map.get(c) == 'Hard'])}")
    
    print(f"\n  🔒 保留类别({len(OOD_RESERVED_CATEGORIES)}个) - 不用于训练，评估泛化:")
    for cat in OOD_RESERVED_CATEGORIES:
        print(f"    - {cat}")
    
    print(f"\nOOD训练类别覆盖:")
    print(f"  目标: {min_ood_coverage*100:.0f}%")
    print(f"  实际: {coverage_ratio*100:.0f}% ({covered_categories}/{len(OOD_TRAIN_CATEGORIES)}类)")
    
    print(f"\n数据格式:")
    print(f"  Problem: 要求输出<think>和<answer>")
    print(f"  Solution: 只包含<answer>标签")
    print(f"  <think>: 由模型自己生成（通过GRPO学习）")
    
    print(f"\nSolution示例:")
    print(f"  ID样本:  <answer>bulk_carrier</answer>")
    print(f"  OOD样本: <answer>unknown</answer>")
    
    print(f"\n预期性能（测试时）:")
    print(f"  ID分类准确率:          {get_expected_id_acc(id_shots)}")
    print(f"  OOD已见检测F1:         {get_expected_ood_seen_f1(id_shots)} (10个训练类)")
    print(f"  OOD未见检测F1:         {get_expected_ood_unseen_f1(id_shots)} (6个保留类)")
    print(f"  泛化Gap:               {get_expected_generalization_gap(id_shots)}")
    
    print(f"\n{'='*70}\n")
    
    return dataset_dict, config


def get_expected_id_acc(k_shot):
    """预期ID分类准确率"""
    expectation = {1: "40-50%", 2: "50-60%", 4: "60-70%", 8: "65-75%", 16: "70-80%"}
    return expectation.get(k_shot, "60-75%")

def get_expected_ood_seen_f1(k_shot):
    """预期OOD已见类检测F1（10个训练类）"""
    expectation = {1: "55-65%", 2: "65-75%", 4: "75-80%", 8: "80-85%", 16: "85-90%"}
    return expectation.get(k_shot, "75-85%")

def get_expected_ood_unseen_f1(k_shot):
    """预期OOD未见类检测F1（6个保留类，泛化性）"""
    expectation = {1: "40-50%", 2: "50-60%", 4: "60-70%", 8: "65-75%", 16: "70-80%"}
    return expectation.get(k_shot, "60-75%")

def get_expected_generalization_gap(k_shot):
    """预期泛化Gap（已见-未见的差距）"""
    expectation = {1: "10-15%", 2: "10-15%", 4: "10-15%", 8: "10-15%", 16: "10-15%"}
    return expectation.get(k_shot, "10-15%")


def main():
    """批量创建多个shot的开集训练集"""
    
    print("\n" + "="*70)
    print("🚀 批量创建开集训练集（纯训练集版本）")
    print("="*70)
    print(f"图像根目录: {IMAGE_ROOT}")
    print(f"输出基础目录: {OUTPUT_BASE}")
    print(f"要创建的配置: {SHOT_CONFIGS}")
    print(f"随机种子: 42")
    print("\n核心特点:")
    print("  ✅ 只创建训练集（测试集已有，全部30类）")
    print("  ✅ Solution只包含<answer>标签")
    print("  ✅ <think>由模型自己生成")
    print("  ✅ 16个OOD = 10训练 + 6保留（评估泛化）")
    print("  ✅ 保守OOD比例（20-25%）")
    print("="*70 + "\n")
    
    # 打印类别划分
    print("📋 类别划分详情:")
    print(f"\n1️⃣  ID类别（{len(ID_CATEGORIES)}个）:")
    for i, cat in enumerate(ID_CATEGORIES, 1):
        print(f"   {i:2d}. {cat}")
    
    print(f"\n2️⃣  OOD训练类别（{len(OOD_TRAIN_CATEGORIES)}个 - 用于训练）:")
    difficulty_map = {
        "aircraft_carrier": "Easy", "frigate": "Easy", "submarine": "Easy",
        "firefighting": "Easy", "tugboat": "Easy",
        "reefer": "Medium", "vehicles_carrier": "Medium", "passenger_ro-ro_ship": "Medium",
        "fso": "Hard", "cruise": "Hard"
    }
    for i, cat in enumerate(OOD_TRAIN_CATEGORIES, 1):
        diff = difficulty_map.get(cat, "")
        print(f"   {i:2d}. [{diff:6s}] {cat}")
    
    print(f"\n3️⃣  OOD保留类别（{len(OOD_RESERVED_CATEGORIES)}个 - 不用于训练）:")
    for i, cat in enumerate(OOD_RESERVED_CATEGORIES, 1):
        print(f"   {i:2d}. {cat}")
    
    print("\n" + "="*70 + "\n")
    
    results = {}
    
    for shots in SHOT_CONFIGS:
        print("\n" + "🔥"*35)
        print(f"开始创建 {shots}-shot 开集训练集")
        print("🔥"*35 + "\n")
        
        output_path = os.path.join(OUTPUT_BASE, f"Ship30_OpenSet_{shots}shot")
        
        try:
            dataset_dict, config = create_openset_dataset(
                image_root=IMAGE_ROOT,
                output_path=output_path,
                id_shots=shots,
                seed=42
            )
            results[shots] = {
                'status': 'success',
                'path': output_path,
                'config': config
            }
            print(f"✅ {shots}-shot 训练集创建完成！")
        except Exception as e:
            results[shots] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {shots}-shot 训练集创建失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== 汇总结果 ====================
    print("\n" + "="*70)
    print("📊 创建结果汇总")
    print("="*70)
    
    for shots in SHOT_CONFIGS:
        result = results[shots]
        if result['status'] == 'success':
            config = result['config']
            print(f"\n✅ {shots:2d}-shot:")
            print(f"   路径: {result['path']}")
            print(f"   训练集: {config['train_samples']['total']} 个")
            print(f"     - ID: {config['train_samples']['id']} 个")
            print(f"     - OOD: {config['train_samples']['ood']} 个 ({100*config['ood_ratio']:.0f}%)")
            print(f"       * Easy: {config['ood_difficulty_distribution']['Easy']} 个")
            print(f"       * Medium: {config['ood_difficulty_distribution']['Medium']} 个")
            print(f"       * Hard: {config['ood_difficulty_distribution']['Hard']} 个")
            print(f"   OOD覆盖: {config['ood_coverage_actual']*100:.0f}%")
            print(f"   预期ID准确率: {get_expected_id_acc(shots)}")
        else:
            print(f"\n❌ {shots:2d}-shot: 失败 - {result['error']}")
    
    print("\n" + "="*70)
    print("✅ 全部完成！")
    print("="*70)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量创建开集训练集（纯训练集版本）")
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT,
                        help="图像根目录")
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE,
                        help="输出基础目录")
    parser.add_argument("--shots", type=int, nargs="+", default=SHOT_CONFIGS,
                        help="要创建的shot配置")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 更新全局配置
    IMAGE_ROOT = args.image_root
    OUTPUT_BASE = args.output_base
    SHOT_CONFIGS = args.shots
    
    main()