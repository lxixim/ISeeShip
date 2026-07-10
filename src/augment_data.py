#!/usr/bin/env python3
"""
数据增强脚本 - 策略1: 按类别分层分配
============================================

原理:
1. ID类别 → 高confidence，但有分布 (0.70-0.95)
   - 70%: 0.78-0.88 (典型ID)
   - 20%: 0.70-0.78 (不太典型)
   - 10%: 0.88-0.95 (非常典型)

2. 保持原数据的answer不变，只改confidence
"""

import re
import random
import numpy as np
from datasets import DatasetDict
from collections import Counter

# 配置
INPUT_PATH = "/data/ljx/visualRft/share_data/Ship30_Zeroshot_OOD_4shot"
OUTPUT_PATH = "/data/ljx/visualRft/share_data/Ship30_Zeroshot_OOD_4shot_augmented"

ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier", "destroyer"
]

def extract_answer(solution):
    """从solution中提取answer"""
    match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().replace(' ', '').replace('_', '').lower()
    return ""

def augment_confidence_smart(example):
    """
    智能增强: 根据答案是否是ID类别分配confidence
    """
    solution = example["solution"]
    
    # 检查是否有confidence标签
    conf_match = re.search(r'<confidence>([\d.]+)</confidence>', solution, re.IGNORECASE)
    if not conf_match:
        return example
    
    # 提取答案
    answer = extract_answer(solution)
    
    # 判断是否是ID类别
    is_id = any(cat in answer for cat in ID_CATEGORIES)
    
    if is_id:
        # ID类别: 应该有较高confidence，但也要有分布
        rand = random.random()
        if rand < 0.70:
            # 70%: 典型情况，高confidence
            new_conf = random.uniform(0.78, 0.88)
        elif rand < 0.90:
            # 20%: 不太典型，中等confidence
            new_conf = random.uniform(0.70, 0.78)
        else:
            # 10%: 非常典型，很高confidence
            new_conf = random.uniform(0.88, 0.95)
    else:
        # 非ID类别或可能错误: 应该有较低confidence
        # (实际上您的数据集可能都是ID，这里作为保险)
        new_conf = random.uniform(0.65, 0.85)
    
    # 保留两位小数
    new_conf = round(new_conf, 2)
    
    # 替换confidence
    new_solution = re.sub(
        r'<confidence>[\d.]+</confidence>',
        f'<confidence>{new_conf}</confidence>',
        solution,
        flags=re.IGNORECASE
    )
    
    example["solution"] = new_solution
    return example

def main():
    print("=" * 70)
    print("🔧 数据增强 - 智能分配confidence")
    print("=" * 70)
    
    # 加载数据
    print(f"\n📂 加载数据: {INPUT_PATH}")
    dataset = DatasetDict.load_from_disk(INPUT_PATH)
    print(f"✅ 训练样本: {len(dataset['train'])}")
    
    # 检查原始分布
    print("\n📊 原始Confidence分布:")
    original_confs = []
    for example in dataset["train"]:
        sol = example["solution"]
        conf_match = re.search(r'<confidence>([\d.]+)</confidence>', sol, re.IGNORECASE)
        if conf_match:
            original_confs.append(float(conf_match.group(1)))
    
    if original_confs:
        counter = Counter(original_confs)
        for conf, count in sorted(counter.items()):
            pct = count / len(original_confs) * 100
            print(f"  {conf:.2f}: {count}个 ({pct:.1f}%)")
    
    # 设置随机种子（可复现）
    random.seed(42)
    np.random.seed(42)
    
    # 应用增强
    print("\n🚀 应用增强...")
    augmented_dataset = dataset.map(
        augment_confidence_smart,
        desc="增强confidence"
    )
    
    # 检查增强后分布
    print("\n✅ 增强后Confidence分布:")
    augmented_confs = []
    for example in augmented_dataset["train"]:
        sol = example["solution"]
        conf_match = re.search(r'<confidence>([\d.]+)</confidence>', sol, re.IGNORECASE)
        if conf_match:
            augmented_confs.append(float(conf_match.group(1)))
    
    if augmented_confs:
        conf_array = np.array(augmented_confs)
        print(f"\n统计指标:")
        print(f"  均值:    {conf_array.mean():.3f}")
        print(f"  标准差:  {conf_array.std():.3f}")
        print(f"  最小值:  {conf_array.min():.2f}")
        print(f"  最大值:  {conf_array.max():.2f}")
        print(f"  唯一值:  {len(set(augmented_confs))}种")
        
        # 分箱统计
        bins = [0.0, 0.60, 0.70, 0.78, 0.88, 0.95, 1.0]
        hist, _ = np.histogram(conf_array, bins=bins)
        print(f"\n分布直方图:")
        for i in range(len(bins)-1):
            pct = hist[i] / len(conf_array) * 100 if len(conf_array) > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:2d}个 ({pct:5.1f}%) {bar}")
    
    # 示例对比
    print(f"\n📋 示例对比 (前3个):")
    for i in range(min(3, len(dataset["train"]))):
        orig = dataset["train"][i]["solution"]
        aug = augmented_dataset["train"][i]["solution"]
        
        orig_conf = re.search(r'<confidence>([\d.]+)</confidence>', orig, re.IGNORECASE)
        aug_conf = re.search(r'<confidence>([\d.]+)</confidence>', aug, re.IGNORECASE)
        
        print(f"\n  样本{i+1}:")
        print(f"    原始: {orig_conf.group(1) if orig_conf else 'N/A'}")
        print(f"    增强: {aug_conf.group(1) if aug_conf else 'N/A'}")
    
    # 保存
    print(f"\n💾 保存到: {OUTPUT_PATH}")
    augmented_dataset.save_to_disk(OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("✅ 数据增强完成!")
    print("=" * 70)
    print("\n📋 下一步:")
    print(f"  1. 使用增强后的数据训练:")
    print(f"     DATA_PATH=\"{OUTPUT_PATH}\"")
    print(f"  2. 运行训练脚本:")
    print(f"     bash train_grpo_diversity.sh")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
