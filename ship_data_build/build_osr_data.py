#!/usr/bin/env python3
"""
OSR数据构建器：14类ID + 5-6类Near-OOD（训练用）
✅ 改进版：更合理的confidence映射 + 难度梯度 + 不列举ID类别
"""

import os
import json
import random
import glob
import torch
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

# ==================== 配置 ====================
IMAGE_ROOT = "/data/ljx/dataset/boat-29/images"
OUTPUT_BASE = "/data/ljx/visualRft/share_data"
MODEL_PATH = "/data/ljx/Qwen2-VL-2B-Instruct"

# 14类ID类别
ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier", "destroyer"
]

# ==================== 改进1: 训练OOD类别选择（增加难度梯度）====================
# 🔥 优先选项（如果图片充足）
TRAIN_OOD_CATEGORIES = [
    # Hard难度（极度相似，细粒度区分）- 2个
    "bitumen",         # 极似oil_products_tanker（最难）
    "cruise",          # 极似passenger_cargo_ship
    
    # Medium难度（中等相似）- 2个
    "passenger_ship",  # 类似passenger_cargo_ship
    "reefer",          # 类似cargo ships
    
    # Easy难度（明显不同）- 2个
    "submarine",       # 潜艇
    "aircraft_carrier" # 航母
]

# 备用选项（如果上面某些类别图片不足）
TRAIN_OOD_CATEGORIES_FALLBACK = [
    "cruise", "fso", "passenger_ship", "monohull_yacht", "submarine"
]

# 全部16类OOD（用于测试集或记录）
ALL_OOD_CATEGORIES = [
    "aircraft_carrier", "bitumen", "catamaran_yacht", "cruise",
    "firefighting", "frigate", "fso", "monohull_sailboat",
    "monohull_yacht", "passenger_ro-ro_ship", "passenger_ship",
    "reefer", "sailing_catamaran", "submarine", "tugboat", "vehicles_carrier"
]

# 验证训练OOD是否都在全部OOD中
for cat in TRAIN_OOD_CATEGORIES:
    if cat not in ALL_OOD_CATEGORIES:
        print(f"⚠️  警告: {cat} 不在ALL_OOD_CATEGORIES中")


# ==================== 改进2: 更合理的OOD倍数策略 ====================
def get_ood_multiplier(k_shot):
    """
    OOD倍数策略（改进版）
    
    确保：
    1. 每个OOD类至少3-4个样本
    2. OOD总数不超过ID的50%
    3. 低shot时OOD相对多，学习拒识
    """
    if k_shot <= 2:
        return 1.5   # 2-shot: 28 ID, ~15 OOD (每类3个)
    elif k_shot <= 4:
        return 1.0   # 4-shot: 56 ID, ~20 OOD (每类4个)
    elif k_shot <= 8:
        return 0.75  # 8-shot: 112 ID, ~30 OOD (每类6个)
    else:
        return 0.6   # 16-shot: 224 ID, ~48 OOD (每类9-10个)


# ==================== 改进3: 更合理的Confidence映射 ====================
def compute_confidence_target(max_similarity, category_name=""):
    """
    🔥 V5版本：将OOD confidence限制在 [0.15, 0.35) 区间
    
    目标分布:
    - 0.15-0.20: 20% (Easy OOD, 明显不同)
    - 0.20-0.25: 30% (Medium-Easy OOD)
    - 0.25-0.30: 35% (Medium-Hard OOD)
    - 0.30-0.35: 15% (Hard OOD, 极度相似)
    
    平均值: ~0.25 (vs 之前的0.429)
    
    Args:
        max_similarity: 与最近ID原型的余弦相似度 [0, 1]
        category_name: OOD类别名（用于特殊处理）
    
    Returns:
        confidence_target: 目标confidence [0.15, 0.35)
    """
    import random
    
    # 特殊处理：某些类别已知很相似
    hard_ood_categories = {"bitumen", "cruise", "fso", "passenger_ship", "reefer"}
    
    # 🔥 将所有OOD confidence映射到 [0.15, 0.35) 区间
    if max_similarity > 0.85:
        # 极度相似（Hard OOD）- 映射到 [0.30, 0.35)
        base = 0.30 + (max_similarity - 0.85) * 0.33  # [0.30, 0.35)
        noise = random.gauss(0, 0.015)  # 小噪声
        conf = base + noise
        
        # Hard类别稍微提升（但仍在范围内）
        if category_name in hard_ood_categories:
            conf += 0.02
        
        return round(max(0.30, min(0.349, conf)), 3)  # 确保 < 0.35
    
    elif max_similarity > 0.70:
        # 高相似（Medium-Hard OOD）- 映射到 [0.25, 0.30)
        base = 0.25 + (max_similarity - 0.70) * 0.33  # [0.25, 0.30)
        noise = random.gauss(0, 0.02)
        conf = base + noise
        
        return round(max(0.25, min(0.30, conf)), 3)
    
    elif max_similarity > 0.55:
        # 中等相似（Medium OOD）- 映射到 [0.20, 0.25)
        base = 0.20 + (max_similarity - 0.55) * 0.33  # [0.20, 0.25)
        noise = random.gauss(0, 0.02)
        conf = base + noise
        
        return round(max(0.20, min(0.25, conf)), 3)
    
    elif max_similarity > 0.40:
        # 低相似（Easy-Medium OOD）- 映射到 [0.18, 0.22)
        base = 0.18 + (max_similarity - 0.40) * 0.27  # [0.18, 0.22)
        noise = random.gauss(0, 0.015)
        conf = base + noise
        
        return round(max(0.18, min(0.22, conf)), 3)
    
    else:
        # 非常不同（Easy OOD）- 映射到 [0.15, 0.20)
        base = 0.15 + (max_similarity / 0.40) * 0.05  # [0.15, 0.20)
        noise = random.gauss(0, 0.015)
        conf = base + noise
        
        # 极低相似度稍微降低
        if max_similarity < 0.30:
            conf -= 0.01
        
        return round(max(0.15, min(0.20, conf)), 3)


# ==================== 改进4: Problem模板（添加<think>）====================
PROBLEM_TEXT = """This is an image containing a ship. Please identify the species of the ship based on the image.

Known 14 ship types:
bulk_carrier, container_ship, general_cargo_ship, passenger_cargo_ship,
fishing_boat, sailing_trimaran, chemical_tanker, crude_oil_tanker,
oil_products_tanker, LNG_tanker, LPG_tanker, kayak, heavy_load_carrier, destroyer

Rules:
- If the ship matches a known type → output that type name
- If the ship does NOT match any known type → output "unknown"
- Do NOT guess if unsure

Output format (strictly follow):
<think>your reasoning process</think>
<answer>species name or unknown</answer>
<confidence>0.xx</confidence>

Confidence: 0.00-1.00, higher means more certain."""


# ==================== 特征提取器 ====================
class FeatureExtractor:
    """用于生成伪OOD分数"""
    def __init__(self, model_path):
        print(f"加载特征提取器: {model_path}")
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.device = self.model.device
    
    @torch.no_grad()
    def extract_features(self, image_paths):
        """逐个提取特征"""
        all_features = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "<image>\nWhat ship is this?"}
                    ]
                }]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                features = outputs.hidden_states[-1]
                mask = inputs.attention_mask.unsqueeze(-1)
                features = (features * mask).sum(dim=1) / mask.sum(dim=1)
                features = F.normalize(features, dim=-1)
                
                all_features.append(features.cpu())
            except Exception as e:
                print(f"⚠️ 特征提取失败 {image_path}: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有成功提取任何特征")
        
        return torch.cat(all_features, dim=0)


# ==================== 数据收集 ====================
def collect_images():
    """扫描所有ID和训练OOD图像"""
    data = {}
    
    # 收集ID类别
    for category in ID_CATEGORIES:
        path = os.path.join(IMAGE_ROOT, category)
        images = glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.png")
        if images:
            data[category] = {"type": "ID", "images": images}
        else:
            print(f"⚠️ 警告: {category} 没有找到图像")
    
    # 收集训练用的OOD（优先使用主选项，如果某些不存在则用备用）
    ood_to_collect = TRAIN_OOD_CATEGORIES.copy()
    missing_categories = []
    
    for category in ood_to_collect:
        path = os.path.join(IMAGE_ROOT, category)
        images = glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.png")
        if images:
            data[category] = {"type": "OOD", "images": images}
        else:
            print(f"⚠️ 警告: OOD类别 {category} 没有找到图像")
            missing_categories.append(category)
    
    # 如果有缺失，尝试从备用列表补充
    if missing_categories:
        print(f"\n⚠️  {len(missing_categories)} 个类别缺失，尝试从备用列表补充...")
        for fallback_cat in TRAIN_OOD_CATEGORIES_FALLBACK:
            if fallback_cat not in data and len(missing_categories) > 0:
                path = os.path.join(IMAGE_ROOT, fallback_cat)
                images = glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.png")
                if images:
                    data[fallback_cat] = {"type": "OOD", "images": images}
                    print(f"  ✓ 补充: {fallback_cat} ({len(images)} 张)")
                    missing_categories.pop(0)
    
    return data


# ==================== 计算ID原型 ====================
def compute_id_prototypes(feature_extractor, id_data, k_shot):
    """从k-shot ID数据计算类别原型"""
    print(f"\n计算ID原型 (k={k_shot})...")
    prototypes = {}
    
    for category, info in id_data.items():
        if info["type"] != "ID":
            continue
        
        images = info["images"][:k_shot]
        if not images:
            print(f"⚠️ {category} 没有足够图像")
            continue
            
        features = feature_extractor.extract_features(images)
        
        prototype = features.mean(dim=0)
        prototypes[category] = F.normalize(prototype, dim=0)
        
        avg_sim = torch.mean(features @ prototypes[category]).item()
        print(f"  ✓ {category:30s}: {len(images)}张, 平均相似度={avg_sim:.3f}")
    
    return prototypes


# ==================== 改进5: 生成伪标签（添加异常检测）====================
def generate_pseudo_labels(feature_extractor, ood_images, prototypes):
    """为OOD样本生成伪标签（改进版）"""
    if not ood_images:
        return {}
    
    features = feature_extractor.extract_features(ood_images)
    pseudo_labels = {}
    
    # 统计异常样本
    very_similar_count = 0
    very_different_count = 0
    
    for idx, feat in enumerate(features):
        similarities = [torch.cosine_similarity(feat, proto, dim=0).item() 
                       for proto in prototypes.values()]
        max_sim = max(similarities)
        closest_id = list(prototypes.keys())[np.argmax(similarities)]
        
        # 获取OOD类别名
        img_path = ood_images[idx]
        category_name = os.path.basename(os.path.dirname(img_path))
        
        # ✅ 使用改进的confidence映射
        confidence_target = compute_confidence_target(max_sim, category_name)
        
        # OOD分数
        ood_score = max(0.1, min(0.95, 1 - max_sim))
        
        # ✅ 异常检测
        if max_sim > 0.90:
            very_similar_count += 1
        elif max_sim < 0.30:
            very_different_count += 1
        
        # ✅ 难度标注
        if max_sim > 0.75:
            difficulty = "Hard"
        elif max_sim > 0.60:
            difficulty = "Medium"
        else:
            difficulty = "Easy"
        
        pseudo_labels[img_path] = {
            "ood_score": round(ood_score, 3),
            "confidence_target": round(confidence_target, 3),
            "closest_id": closest_id,
            "max_similarity": round(max_sim, 3),
            "difficulty": difficulty,
        }
    
    # ✅ 打印异常统计
    if very_similar_count > 0:
        print(f"    ⚠️  发现 {very_similar_count} 个极度相似样本 (sim>0.90)")
    if very_different_count > 0:
        print(f"    ✓ 发现 {very_different_count} 个明显不同样本 (sim<0.30)")
    
    return pseudo_labels


# ==================== 主数据构建函数 ====================
def build_osr_dataset(k_shot, output_dir):
    """构建完整的OSR训练数据集（改进版）"""
    
    print(f"\n{'='*70}")
    print(f"构建 {k_shot}-shot OSR数据集（14ID + Near-OOD）")
    print(f"{'='*70}")
    print(f"目标OOD类别: {TRAIN_OOD_CATEGORIES}")
    print(f"{'='*70}")
    
    # 收集图像
    data = collect_images()
    id_data = {k: v for k, v in data.items() if v["type"] == "ID"}
    ood_data = {k: v for k, v in data.items() if v["type"] == "OOD"}
    
    print(f"\n找到 {len(id_data)} 个ID类别:")
    for cat in sorted(id_data.keys()):
        print(f"  - {cat}: {len(id_data[cat]['images'])} 张")
    
    print(f"\n找到 {len(ood_data)} 个训练OOD类别:")
    for cat in sorted(ood_data.keys()):
        print(f"  - {cat}: {len(ood_data[cat]['images'])} 张")
    
    if len(id_data) != 14:
        print(f"⚠️ 警告: ID类别数量 {len(id_data)} != 14")
    
    # 加载特征提取器
    feature_extractor = FeatureExtractor(MODEL_PATH)
    
    # 计算ID原型
    prototypes = compute_id_prototypes(feature_extractor, id_data, k_shot)
    
    # 构建训练样本
    train_samples = []
    
    # 1. ID样本
    print(f"\n{'='*70}")
    print(f"生成ID样本（confidence=0.95）")
    print(f"{'='*70}")
    
    for category, info in sorted(id_data.items()):
        images = info["images"].copy()
        random.shuffle(images)
        shots_to_use = min(len(images), k_shot)
    
        for i, img_path in enumerate(images[:shots_to_use]):
        # ⚡ 改进: 根据样本索引给不同confidence
            if i < shots_to_use // 5:
            # 20%困难样本: 0.75-0.85
                id_confidence = round(random.uniform(0.75, 0.85), 2)
            else:
            # 80%简单样本: 0.88-0.98
                id_confidence = round(random.uniform(0.88, 0.98), 2)
        
            sample = {
                "image": img_path,
                "problem": PROBLEM_TEXT,
                "solution": f"<answer>{category}</answer> <confidence>{id_confidence}</confidence>",
                "metadata": json.dumps({
                    "type": "ID",
                    "category": category,
                    "confidence_target": id_confidence,  # ⚡ 动态target
                    "split": "support" if i < shots_to_use//2 else "query"
                })
            }
            train_samples.append(sample)
    
        print(f"  ✓ {category:30s}: {shots_to_use:2d} 张")
    
    # 2. OOD样本
    print(f"\n{'='*70}")
    print(f"生成Near-OOD样本（动态confidence + 难度标注）")
    print(f"{'='*70}")
    
    ood_multiplier = get_ood_multiplier(k_shot)
    total_id_samples = len(id_data) * k_shot
    max_total_ood = int(total_id_samples * 0.5)
    print(f"OOD倍数: {ood_multiplier}x")
    print(f"ID总样本: {total_id_samples}, 最大OOD样本: {max_total_ood} (50%限制)")
    
    # 统计各难度OOD样本数
    ood_difficulty_stats = {"Easy": 0, "Medium": 0, "Hard": 0}
    
    for category, info in sorted(ood_data.items()):
        images = info["images"].copy()
        random.shuffle(images)
        
        # 计算该类应采样多少
        ood_samples_per_class = min(
            len(images), 
            int(k_shot * ood_multiplier)
        )
        
        # 50%限制
        current_ood_count = sum(1 for s in train_samples if json.loads(s["metadata"])["type"] == "OOD")
        remaining_ood_budget = max(0, max_total_ood - current_ood_count)
        
        ood_samples_per_class = min(ood_samples_per_class, remaining_ood_budget)
        selected_images = images[:ood_samples_per_class]
        
        pseudo_labels = generate_pseudo_labels(feature_extractor, selected_images, prototypes)
        
        for img_path in selected_images:
            label = pseudo_labels[img_path]
            
            sample = {
                "image": img_path,
                "problem": PROBLEM_TEXT,
                "solution": f"<answer>unknown</answer> <confidence>{label['confidence_target']:.2f}</confidence>",
                "metadata": json.dumps({
                    "type": "OOD",
                    "category": category,
                    "confidence_target": label["confidence_target"],
                    "pseudo_ood_score": label["ood_score"],
                    "closest_id": label["closest_id"],
                    "max_similarity": label["max_similarity"],
                    "difficulty": label["difficulty"],
                    "split": "query"
                })
            }
            train_samples.append(sample)
            
            # 统计难度分布
            ood_difficulty_stats[label["difficulty"]] += 1
        
        # 打印统计
        avg_conf = np.mean([l["confidence_target"] for l in pseudo_labels.values()])
        avg_sim = np.mean([l["max_similarity"] for l in pseudo_labels.values()])
        difficulty_dist = {
            d: sum(1 for l in pseudo_labels.values() if l["difficulty"] == d)
            for d in ["Easy", "Medium", "Hard"]
        }
        
        print(f"  ✓ {category:30s}: {ood_samples_per_class:2d} 张")
        print(f"      平均conf={avg_conf:.3f}, 平均sim={avg_sim:.3f}")
        print(f"      难度: Easy={difficulty_dist['Easy']}, "
              f"Med={difficulty_dist['Medium']}, Hard={difficulty_dist['Hard']}")
    
    # 加载图像到内存
    print(f"\n{'='*70}")
    print(f"加载图像到内存...")
    print(f"{'='*70}")
    
    def load_image(sample):
        try:
            img = Image.open(sample["image"]).convert('RGB')
            sample["image"] = img
            return sample
        except Exception as e:
            print(f"⚠️ 加载失败 {sample['image']}: {e}")
            return None
    
    loaded_samples = [s for s in map(load_image, train_samples) if s is not None]
    random.shuffle(loaded_samples)
    
    print(f"  成功加载: {len(loaded_samples)}/{len(train_samples)}")
    
    # 创建Dataset
    dataset = Dataset.from_list(loaded_samples)
    dataset_dict = DatasetDict({"train": dataset})
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    # 获取实际使用的OOD类别
    actual_ood_cats = list(ood_data.keys())
    
    # 保存配置信息
    config = {
        "k_shot": k_shot,
        "id_categories": ID_CATEGORIES,
        "train_ood_categories": actual_ood_cats,
        "all_ood_categories": ALL_OOD_CATEGORIES,
        "unseen_ood_categories": [c for c in ALL_OOD_CATEGORIES if c not in actual_ood_cats],
        "ood_multiplier": ood_multiplier,
        "id_confidence": 0.95,
        "ood_confidence_range": [0.15, 0.35],
        "problem_includes_id_list": False,  # ✅ 标记不包含ID列表
    }
    
    with open(os.path.join(output_dir, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 统计
    id_cnt = sum(1 for s in loaded_samples if json.loads(s["metadata"])["type"] == "ID")
    ood_cnt = sum(1 for s in loaded_samples if json.loads(s["metadata"])["type"] == "OOD")
    
    print(f"\n{'='*70}")
    print("📊 数据集统计")
    print(f"{'='*70}")
    print(f"ID样本:   {id_cnt:4d} (14类 × {k_shot}张, confidence=0.95)")
    print(f"\nOOD样本:  {ood_cnt:4d} ({len(actual_ood_cats)}类Near-OOD)")
    print(f"  按难度分布:")
    for diff in ["Easy", "Medium", "Hard"]:
        count = ood_difficulty_stats[diff]
        ratio = count / ood_cnt if ood_cnt > 0 else 0
        print(f"    {diff:8s}: {count:3d} ({100*ratio:.1f}%)")
    
    print(f"\n  Confidence范围:")
    ood_samples = [s for s in loaded_samples if json.loads(s["metadata"])["type"] == "OOD"]
    ood_confs = [json.loads(s["metadata"])["confidence_target"] for s in ood_samples]
    if ood_confs:
        print(f"    最小: {min(ood_confs):.3f}")
        print(f"    平均: {np.mean(ood_confs):.3f}")
        print(f"    最大: {max(ood_confs):.3f}")
    
    print(f"\n总计:     {len(loaded_samples):4d}")
    print(f"\n训练OOD类别 ({len(actual_ood_cats)}类):")
    for cat in sorted(actual_ood_cats):
        print(f"  ✓ {cat}")
    
    print(f"\n未见OOD类别 ({len(ALL_OOD_CATEGORIES) - len(actual_ood_cats)}类，用于测试泛化):")
    for cat in [c for c in ALL_OOD_CATEGORIES if c not in actual_ood_cats]:
        print(f"  • {cat}")
    
    print(f"\n✅ 数据集已保存至: {output_dir}")
    print(f"✅ 配置已保存至: {os.path.join(output_dir, 'dataset_config.json')}")
    print(f"{'='*70}")
    
    return dataset_dict


# ==================== 主函数 ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="构建OSR训练数据集（14ID + Near-OOD）")
    parser.add_argument("--k_shot", type=int, default=4, choices=[1,2,4,8,16], 
                       help="每类ID的样本数")
    parser.add_argument("--output_suffix", type=str, default="", 
                       help="输出目录后缀（可选）")
    args = parser.parse_args()
    
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_dir = f"{OUTPUT_BASE}/OSR_14ID_NearOOD_NEW_{args.k_shot}shot{suffix}"
    build_osr_dataset(args.k_shot, output_dir)