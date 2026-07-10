#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure OOD Detection Dataset Preparation

核心思想：
1. 只使用 14 个 ID 类别的图像
2. Prompt 中包含 14 个 ID 类别 + 16 个伪 OOD 类别名（陷阱）
3. 训练时只有 ID 样本，但模型"看到"了伪 OOD 类别名
4. 如果模型输出伪 OOD 类别名 → 奖励函数会惩罚

这样模型学到：
- 不要编造训练集中没见过的类别
- 不确定时输出 {"ood": true}
- 真正的分布外检测能力
"""
import os
import random
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from PIL import Image
import argparse

# Set random seed for reproducibility
random.seed(42)

# Source data path
SOURCE_DATA_PATH = "/data/ljx/dataset/boat-29/images"

# Output base path
OUTPUT_BASE_PATH = "/data/ljx/visualRft/share_data"

# ID classes (14) - 只有这些类别的图像会被使用
ID_CLASSES = [
    'aircraft_carrier', 'bitumen', 'bulk_carrier', 'chemical_tanker',
    'cruise', 'destroyer', 'fishing_ships', 'fso', 'kayak',
    'LNG_tanker', 'passenger_cargo_ship', 'sailing_trimaran',
    'submarine', 'vehicles_carrier'
]

# Pseudo OOD classes (16) - 只有名字，没有图像（用作"陷阱"）
PSEUDO_OOD_CLASSES = [
    'container_ship', 'crude_oil_tanker', 'tugboat', 'frigate',
    'general_cargo_ship', 'reefer', 'passenger_ship', 'LPG_tanker',
    'catamaran_yacht', 'heavy_load_carrier', 'firefighting',
    'monohull_sailboat', 'monohull_yacht', 'oil_products_tanker',
    'passenger_ro-ro_ship', 'sailing_catamaran'
]

# 所有类别（ID + 伪OOD）会出现在 prompt 中
ALL_CLASSES_FOR_PROMPT = ID_CLASSES + PSEUDO_OOD_CLASSES


def generate_problem_prompt():
    """
    生成包含伪 OOD 类别的 prompt
    这是核心：让模型"看到"伪 OOD 类别名，但训练集中没有对应图像
    """
    classes_str = "', '".join(ALL_CLASSES_FOR_PROMPT)
    
    prompt = f"""This is an image containing a ship. Please identify the species of the ship based on the image.

You must compare it against the following known ship types: '{classes_str}'.

IMPORTANT:
- If you are CERTAIN it matches one of the types, output in JSON format: {{"category": "type_name", "ood": false}}
- If you are UNCERTAIN or it doesn't clearly match any type, output: {{"ood": true, "category": "unknown"}}

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output format should be:
<think> ... </think> <answer>{{"category": "ship_type", "ood": false}}</answer>"""
    
    return prompt


def get_solution_for_id_class(category_name):
    """
    为 ID 类别生成 solution（只是类别名，不包含 thinking）
    在 SFT 阶段，convert_example 会负责生成 thinking
    """
    return category_name


def collect_images_from_category(category_path, num_shots):
    """Randomly sample specified number of images from a category"""
    image_files = []
    category_dir = Path(category_path)
    
    if not category_dir.exists():
        print(f"Warning: Category directory does not exist {category_path}")
        return []
    
    # Collect all image files
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(category_dir.glob(ext)))
    
    # If not enough images, return all images
    if len(image_files) < num_shots:
        print(f"Warning: {category_dir.name} category only has {len(image_files)} images, less than required {num_shots}")
        return image_files
    
    # Random sampling
    selected_images = random.sample(image_files, num_shots)
    return selected_images


def create_pure_ood_dataset(num_shots, output_name):
    """
    创建纯 OOD 检测数据集
    - 只包含 14 个 ID 类别的图像
    - Prompt 包含 14 个 ID + 16 个伪 OOD 类别名
    """
    print(f"\n{'='*80}")
    print(f"Creating Pure OOD Detection Dataset: {num_shots}-shot, {output_name}")
    print(f"{'='*80}")
    print(f"ID classes (with images): {len(ID_CLASSES)}")
    print(f"Pseudo OOD classes (name only): {len(PSEUDO_OOD_CLASSES)}")
    print(f"{'='*80}\n")
    
    dataset_samples = []
    problem_prompt = generate_problem_prompt()
    
    # 只遍历 ID 类别（有图像的）
    for category in ID_CLASSES:
        category_path = os.path.join(SOURCE_DATA_PATH, category)
        
        # Sample images from this category
        selected_images = collect_images_from_category(category_path, num_shots)
        
        # Create data sample for each image
        for img_path in selected_images:
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    img.verify()
                
                sample = {
                    'image': str(img_path),
                    'problem': problem_prompt,  # 包含 14 ID + 16 伪OOD 类别名
                    'solution': get_solution_for_id_class(category)  # 只是类别名
                }
                dataset_samples.append(sample)
                
            except Exception as e:
                print(f"Error: Cannot process image {img_path}: {e}")
                continue
    
    print(f"✅ Total collected {len(dataset_samples)} samples (all ID classes)")
    print(f"   Expected: {len(ID_CLASSES)} categories x {num_shots} shots = {len(ID_CLASSES) * num_shots}")
    print(f"   Pseudo OOD classes in prompt: {len(PSEUDO_OOD_CLASSES)} (no images)")
    print()
    
    # Create HuggingFace Dataset
    features = Features({
        'image': HFImage(),
        'problem': Value('string'),
        'solution': Value('string')
    })
    
    dataset = Dataset.from_dict(
        {
            'image': [s['image'] for s in dataset_samples],
            'problem': [s['problem'] for s in dataset_samples],
            'solution': [s['solution'] for s in dataset_samples]
        },
        features=features
    )
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset
    })
    
    # Save dataset
    output_path = os.path.join(OUTPUT_BASE_PATH, output_name)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving dataset to: {output_path}")
    dataset_dict.save_to_disk(output_path)
    print(f"✅ Dataset saved successfully!")
    
    # Print dataset info
    print(f"\n{'='*80}")
    print(f"Dataset Summary:")
    print(f"  - Training samples: {len(dataset)}")
    print(f"  - Samples per ID category: {num_shots}")
    print(f"  - ID categories (with images): {len(ID_CLASSES)}")
    print(f"  - Pseudo OOD categories (name only in prompt): {len(PSEUDO_OOD_CLASSES)}")
    print(f"  - Total categories in prompt: {len(ALL_CLASSES_FOR_PROMPT)}")
    print(f"{'='*80}\n")
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description='Prepare pure OOD detection datasets (ID-only images)')
    parser.add_argument('--shots', type=str, default='all', 
                       help='Number of shots to create, can be 1,2,4,8,16 or all (create all)')
    args = parser.parse_args()
    
    # Define dataset configurations to create
    dataset_configs = {
        1: 'ViRFT_CLS_ship14_PureOOD_1shot',
        2: 'ViRFT_CLS_ship14_PureOOD_2shot',
        4: 'ViRFT_CLS_ship14_PureOOD_4shot',
        8: 'ViRFT_CLS_ship14_PureOOD_8shot',
        16: 'ViRFT_CLS_ship14_PureOOD_16shot'
    }
    
    # Determine which datasets to create
    if args.shots == 'all':
        shots_to_create = list(dataset_configs.keys())
    else:
        shots_to_create = [int(s.strip()) for s in args.shots.split(',')]
    
    print(f"\n{'='*80}")
    print(f"Pure OOD Detection Dataset Preparation")
    print(f"{'='*80}")
    print(f"Will create datasets with following shots: {shots_to_create}")
    print(f"Source data path: {SOURCE_DATA_PATH}")
    print(f"Output path: {OUTPUT_BASE_PATH}")
    print(f"\nStrategy:")
    print(f"  - Use images from {len(ID_CLASSES)} ID classes only")
    print(f"  - Add {len(PSEUDO_OOD_CLASSES)} pseudo OOD class names in prompt (no images)")
    print(f"  - Model sees all {len(ALL_CLASSES_FOR_PROMPT)} class names in prompt")
    print(f"  - But only {len(ID_CLASSES)} classes have training images")
    print(f"  - If model outputs pseudo OOD class → GRPO will penalize!")
    print(f"{'='*80}\n")
    
    # Verify source data path
    if not os.path.exists(SOURCE_DATA_PATH):
        print(f"❌ Error: Source data path does not exist: {SOURCE_DATA_PATH}")
        return
    
    # Create each dataset
    for num_shots in shots_to_create:
        if num_shots not in dataset_configs:
            print(f"⚠️  Warning: Skipping unsupported shot number {num_shots}")
            continue
        
        output_name = dataset_configs[num_shots]
        try:
            create_pure_ood_dataset(num_shots, output_name)
        except Exception as e:
            print(f"❌ Error: Failed to create {num_shots}-shot dataset: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ All Pure OOD datasets created successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()



