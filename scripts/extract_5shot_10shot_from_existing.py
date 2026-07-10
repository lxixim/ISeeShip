#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从现有的few-shot数据集中抽取5shot和10shot数据集
"""
import os
import random
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from collections import defaultdict

# 设置随机种子
random.seed(42)

# 从ships.txt读取标准类别列表
SHIPS_TXT_PATH = '/data/ljx/visualRft/classification/val_data/ships.txt'
if os.path.exists(SHIPS_TXT_PATH):
    with open(SHIPS_TXT_PATH, 'r') as f:
        VALID_CATEGORIES = [line.strip() for line in f if line.strip()]
else:
    # 如果文件不存在，使用默认列表
    VALID_CATEGORIES = [
        'aircraft_carrier', 'bitumen', 'bulk_carrier', 'catamaran_yacht',
        'chemical_tanker', 'container_ship', 'crude_oil_tanker', 'cruise',
        'destroyer', 'firefighting', 'fishing_boat', 'frigate', 'fso',
        'general_cargo_ship', 'heavy_load_carrier', 'kayak', 'LNG_tanker',
        'LPG_tanker', 'monohull_sailboat', 'monohull_yacht', 'oil_products_tanker',
        'passenger_cargo_ship', 'passenger_ro-ro_ship', 'passenger_ship',
        'reefer', 'sailing_catamaran', 'sailing_trimaran', 'submarine',
        'tugboat', 'vehicles_carrier'
    ]

# 类别名标准化（处理大小写问题）
CATEGORY_NORMALIZE = {cat.lower(): cat for cat in VALID_CATEGORIES}
print(f"加载了 {len(VALID_CATEGORIES)} 个标准类别")

# 输入数据集路径
INPUT_DATASETS = {
    '1shot': '/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_1shot_cot_new/train',
    '2shot': '/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_2shot_cot_new/train',
    '4shot': '/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_4shot_cot_new/train',
    '8shot': '/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_8shot_cot_new/train',
    '16shot': '/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_16shot_cot_new/train',
}

# 输出路径
OUTPUT_BASE_PATH = "/data/ljx/visualRft/share_data"

def extract_category_from_solution(solution_str):
    """从solution字符串中提取类别名，并验证是否为有效类别"""
    # solution格式通常是: <think>...</think> <answer>category_name</answer>
    if '<answer>' in solution_str and '</answer>' in solution_str:
        start = solution_str.find('<answer>') + len('<answer>')
        end = solution_str.find('</answer>')
        category = solution_str[start:end].strip()
        
        # 标准化类别名（处理大小写）
        category_lower = category.lower()
        if category_lower in CATEGORY_NORMALIZE:
            return CATEGORY_NORMALIZE[category_lower]
        else:
            # 如果不在有效列表中，返回None（可能是模型生成的错误答案）
            return None
    return None

def extract_category_from_image_path(image_obj):
    """尝试从图像路径中提取类别名"""
    # 尝试从PIL Image对象获取路径
    if hasattr(image_obj, 'filename') and image_obj.filename:
        path = image_obj.filename
        # 路径格式通常是: /data/ljx/dataset/boat-29/images/{category}/{image_file}
        parts = Path(path).parts
        for i, part in enumerate(parts):
            if part == 'images' and i + 1 < len(parts):
                category = parts[i + 1]
                category_lower = category.lower()
                if category_lower in CATEGORY_NORMALIZE:
                    return CATEGORY_NORMALIZE[category_lower]
    return None

def group_by_category(dataset):
    """将数据集按类别分组"""
    category_samples = defaultdict(list)
    skipped_count = 0
    path_extracted = 0
    solution_extracted = 0
    
    for i, sample in enumerate(dataset):
        category = None
        
        # 优先从图像路径中提取（更可靠，因为路径是原始数据）
        category = extract_category_from_image_path(sample['image'])
        if category:
            path_extracted += 1
        else:
            # 如果路径提取失败，再从solution中提取（但可能包含错误答案）
            category = extract_category_from_solution(sample['solution'])
            if category:
                solution_extracted += 1
        
        if category:
            category_samples[category].append(sample)
        else:
            skipped_count += 1
            if skipped_count <= 5:  # 只打印前5个跳过的样本
                print(f"  警告: 样本 {i} 无法确定类别，已跳过")
                # 尝试显示图像路径
                img = sample['image']
                if hasattr(img, 'filename') and img.filename:
                    print(f"    图像路径: {img.filename}")
                print(f"    solution: {sample['solution'][:150]}...")
    
    print(f"\n类别提取统计:")
    print(f"  从图像路径提取: {path_extracted} 个样本")
    print(f"  从solution提取: {solution_extracted} 个样本")
    if skipped_count > 5:
        print(f"  跳过的样本: {skipped_count} 个（前5个已显示详情）")
    elif skipped_count > 0:
        print(f"  跳过的样本: {skipped_count} 个")
    
    return category_samples

def create_dataset_from_existing(source_dataset_name, target_shots, output_name):
    """从现有数据集中抽取指定shot数的数据集"""
    print(f"\n{'='*60}")
    print(f"从 {source_dataset_name} 创建 {target_shots}-shot 数据集: {output_name}")
    print(f"{'='*60}")
    
    # 加载源数据集
    source_path = INPUT_DATASETS[source_dataset_name]
    print(f"加载源数据集: {source_path}")
    source_dataset = load_from_disk(source_path)
    print(f"源数据集样本数: {len(source_dataset)}")
    
    # 按类别分组
    print("按类别分组...")
    category_samples = group_by_category(source_dataset)
    print(f"找到 {len(category_samples)} 个类别")
    
    # 检查每个类别的样本数
    for cat, samples in sorted(category_samples.items()):
        print(f"  {cat:30s}: {len(samples)} 个样本")
    
    # 检查是否所有30个类别都存在
    missing_categories = set(VALID_CATEGORIES) - set(category_samples.keys())
    if missing_categories:
        print(f"\n警告: 以下类别在数据集中不存在: {sorted(missing_categories)}")
    
    # 从每个类别中抽取指定数量的样本
    selected_samples = []
    for category in VALID_CATEGORIES:  # 按照标准类别顺序处理
        if category not in category_samples:
            print(f"  警告: 类别 {category} 在数据集中不存在，跳过")
            continue
            
        samples = category_samples[category]
        if len(samples) >= target_shots:
            # 如果样本数足够，随机抽取
            selected = random.sample(samples, target_shots)
        else:
            # 如果样本数不够，使用所有样本
            print(f"  警告: {category} 只有 {len(samples)} 个样本，少于 {target_shots}")
            selected = samples
        selected_samples.extend(selected)
    
    print(f"\n总共抽取了 {len(selected_samples)} 个样本")
    print(f"期望样本数: {len(VALID_CATEGORIES)} 个类别 × {target_shots} shot = {len(VALID_CATEGORIES) * target_shots}")
    print(f"实际类别数: {len(category_samples)}")
    
    # 创建新的数据集
    new_dataset = Dataset.from_list(selected_samples)
    
    # 创建DatasetDict
    dataset_dict = DatasetDict({
        'train': new_dataset
    })
    
    # 保存数据集
    output_path = os.path.join(OUTPUT_BASE_PATH, output_name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n保存数据集到: {output_path}")
    dataset_dict.save_to_disk(output_path)
    print(f"数据集保存成功!")
    
    return dataset_dict

def main():
    print("="*60)
    print("从现有数据集中抽取5shot和10shot数据集")
    print("="*60)
    
    # 检查源数据集是否存在
    print("\n检查源数据集...")
    for name, path in INPUT_DATASETS.items():
        print(f"检查 {name}...")
        if os.path.exists(path):
            try:
                print(f"  加载 {path}...")
                ds = load_from_disk(path)
                print(f"  ✓ {name:8s}: {len(ds):6d} 个样本")
            except Exception as e:
                print(f"  ✗ {name:8s}: 加载失败 - {e}")
        else:
            print(f"  ✗ {name:8s}: 路径不存在: {path}")
    
    # 创建5shot数据集 - 从8shot中抽取（8shot有240个样本，足够抽取5shot=150个样本）
    print("\n" + "="*60)
    print("创建5shot数据集")
    print("="*60)
    create_dataset_from_existing(
        source_dataset_name='8shot',
        target_shots=5,
        output_name='ViRFT_CLS_ship30_5shot_cot_new'
    )
    
    # 创建10shot数据集 - 从16shot中抽取（16shot有480个样本，足够抽取10shot=300个样本）
    print("\n" + "="*60)
    print("创建10shot数据集")
    print("="*60)
    create_dataset_from_existing(
        source_dataset_name='16shot',
        target_shots=10,
        output_name='ViRFT_CLS_ship30_10shot_cot_new'
    )
    
    print("\n" + "="*60)
    print("所有数据集创建完成!")
    print("="*60)
    print("\n创建的数据集:")
    print("  - ViRFT_CLS_ship30_5shot_cot_new")
    print("  - ViRFT_CLS_ship30_10shot_cot_new")

if __name__ == "__main__":
    main()

