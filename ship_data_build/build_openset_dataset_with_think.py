#!/usr/bin/env python3
"""
船舶开集识别数据集 - 添加think标签版本
"""

import os
import json
import random
import glob
from PIL import Image
from datasets import DatasetDict, Dataset

# ==================== 配置 ====================
IMAGE_ROOT = "/data/ljx/dataset/boat-29/images"
OUTPUT_BASE = "/data/ljx/visualRft/share_data"

# ID类别（14个）
ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier",
    "tugboat"
]

# OOD训练类（10个）
OOD_TRAIN_CATEGORIES = [
    "destroyer", "aircraft_carrier", "frigate", "submarine",
    "firefighting", "reefer", "vehicles_carrier", "passenger_ro-ro_ship",
    "fso", "cruise"
]

# OOD保留类（6个）
OOD_RESERVED_CATEGORIES = [
    "bitumen", "passenger_ship", "catamaran_yacht",
    "monohull_yacht", "monohull_sailboat", "sailing_catamaran"
]

ID_LIST_STR = ", ".join(ID_CATEGORIES)

# Prompt模板
PROMPT_TEMPLATE = f"""This is an image containing a ship. Please identify the type of the ship based on the image.

You can ONLY output ONE of these exact answers:
{ID_LIST_STR}, unknown

Rules:
- If the ship matches one of the 14 known types → output that exact type name
- If the ship does NOT match ANY of the 14 types → output "unknown"

Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer>type</answer>
Please strictly follow the format."""


# ==================== Think模板 ====================
# ID类别的think模板（GRPO会探索更多内容）
ID_THINK_TEMPLATES = [
    "This vessel shows typical features of a {category}.",
    "Based on the hull shape and structure, this appears to be a {category}.",
    "The visual characteristics match a {category}.",
    "Analyzing the ship's features, I identify this as a {category}.",
    "This ship has distinctive features of a {category}.",
]

# OOD的think模板
OOD_THINK_TEMPLATES = [
    "This vessel does not match any of the 14 known ship types.",
    "The features of this ship do not correspond to any known category.",
    "This appears to be a ship type outside the known categories.",
    "I cannot identify this vessel as any of the 14 known types.",
    "This ship's characteristics do not fit any known category.",
]


def get_id_think(category):
    """生成ID类别的think内容"""
    template = random.choice(ID_THINK_TEMPLATES)
    return template.format(category=category.replace('_', ' '))


def get_ood_think():
    """生成OOD的think内容"""
    return random.choice(OOD_THINK_TEMPLATES)


def collect_images(image_root, categories):
    """收集图像"""
    data = {}
    for cat in categories:
        cat_dir = os.path.join(image_root, cat)
        if not os.path.exists(cat_dir):
            print(f"  ⚠️ {cat}: 目录不存在")
            continue
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(glob.glob(os.path.join(cat_dir, ext)))
        if images:
            data[cat] = images
            print(f"  ✓ {cat}: {len(images)} 张")
    return data


def create_openset_dataset(image_root, output_path, id_shots, ood_ratio=0.25, seed=42):
    """
    创建开集识别数据集
    """
    random.seed(seed)
    
    print("\n" + "="*70)
    print(f"📦 开集识别数据集 ({id_shots}-shot, OOD比例{100*ood_ratio:.0f}%)")
    print("="*70)
    
    # 收集图像
    print("\n🔍 收集ID图像...")
    id_data = collect_images(image_root, ID_CATEGORIES)
    
    print("\n🔍 收集OOD图像...")
    ood_data = collect_images(image_root, OOD_TRAIN_CATEGORIES)
    
    train_data = []
    
    # ===== ID样本 =====
    print("\n📦 创建ID样本...")
    for category in ID_CATEGORIES:
        if category not in id_data:
            continue
        images = id_data[category]
        random.shuffle(images)
        n = min(id_shots, len(images))
        
        for img_path in images[:n]:
            # 🔥 添加think标签
            think_content = get_id_think(category)
            train_data.append({
                "image": img_path,
                "problem": PROMPT_TEMPLATE,
                "solution": f"<think>{think_content}</think>\n<answer>{category}</answer>",
            })
        print(f"  {category}: {n} 样本")
    
    n_id = len(train_data)
    
    # ===== OOD样本 =====
    n_ood_target = int(n_id * ood_ratio / (1 - ood_ratio))
    ood_per_class = max(1, n_ood_target // len(OOD_TRAIN_CATEGORIES))
    
    print(f"\n📦 创建OOD样本 (目标{n_ood_target}个)...")
    for category in OOD_TRAIN_CATEGORIES:
        if category not in ood_data:
            continue
        images = ood_data[category]
        random.shuffle(images)
        n = min(ood_per_class, len(images))
        
        for img_path in images[:n]:
            # 🔥 添加think标签
            think_content = get_ood_think()
            train_data.append({
                "image": img_path,
                "problem": PROMPT_TEMPLATE,
                "solution": f"<think>{think_content}</think>\n<answer>unknown</answer>",
            })
        print(f"  {category}: {n} 样本 → unknown")
    
    n_ood = len(train_data) - n_id
    
    # 统计
    print(f"\n📊 统计: ID={n_id}, OOD={n_ood}, 总计={len(train_data)}")
    print(f"   OOD实际比例: {100*n_ood/len(train_data):.1f}%")
    
    # 🔥 统计think多样性
    thinks = [item['solution'].split('</think>')[0].replace('<think>', '') for item in train_data]
    unique_thinks = len(set(thinks))
    print(f"   <think>多样性: {unique_thinks} 种不同内容")
    
    random.shuffle(train_data)
    
    # 加载图像
    print(f"\n📥 加载图像...")
    loaded_data = []
    for item in train_data:
        try:
            img = Image.open(item['image']).convert('RGB')
            loaded_data.append({
                'image': img,
                'problem': item['problem'],
                'solution': item['solution'],
            })
        except:
            pass
    
    # 保存
    dataset = DatasetDict({'train': Dataset.from_list(loaded_data)})
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    
    # 配置文件
    config = {
        "task": "open_set_recognition",
        "id_shots": id_shots,
        "ood_ratio": ood_ratio,
        "id_categories": ID_CATEGORIES,
        "ood_train_categories": OOD_TRAIN_CATEGORIES,
        "ood_reserved_categories": OOD_RESERVED_CATEGORIES,
        "id_samples": n_id,
        "ood_samples": n_ood,
        "total_samples": len(loaded_data),
        "think_diversity": unique_thinks,
        "has_think_tag": True,  # 🔥 标记有think标签
    }
    with open(os.path.join(output_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ 已保存: {output_path}")
    
    # 🔥 打印示例
    print("\n📝 Solution示例:")
    for i in range(min(5, len(train_data))):
        print(f"  [{i+1}] {train_data[i]['solution']}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--ood_ratio", type=float, default=0.30)
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--output_base", type=str, default=OUTPUT_BASE)
    args = parser.parse_args()
    
    output_path = os.path.join(args.output_base, f"Ship_OpenSet_{args.shots}shot_addthink")
    create_openset_dataset(args.image_root, output_path, args.shots, args.ood_ratio)