#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查OpenSet数据集的具体问题"""
from datasets import load_from_disk
from collections import defaultdict

OPENSET_CATEGORIES_14 = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier",
    "tugboat"
]

datasets = {
    '1shot': ('/data/ljx/visualRft/share_data/Ship30_OpenSet_1shot_cot_new/train', 1),
    '2shot': ('/data/ljx/visualRft/share_data/Ship30_OpenSet_2shot_cot_new/train', 2),
    '4shot': ('/data/ljx/visualRft/share_data/Ship30_OpenSet_4shot_cot_new/train', 4),
    '8shot': ('/data/ljx/visualRft/share_data/Ship30_OpenSet_4shot_cot_new/train', 8),
    '16shot': ('/data/ljx/visualRft/share_data/Ship30_OpenSet_16shot_cot_new/train', 16),
}

for name, (path, expected) in datasets.items():
    print(f"\n{'='*70}")
    print(f"检查 {name} 数据集")
    print(f"{'='*70}")
    
    ds = load_from_disk(path)
    category_indices = defaultdict(list)
    
    for i, sample in enumerate(ds):
        solution = sample['solution']
        if '<answer>' in solution and '</answer>' in solution:
            start = solution.find('<answer>') + len('<answer>')
            end = solution.find('</answer>')
            category = solution[start:end].strip()
            category_indices[category].append(i)
    
    print(f"\n所有类别统计:")
    for cat in OPENSET_CATEGORIES_14:
        count = len(category_indices.get(cat, []))
        if count == 0:
            print(f"  {cat:30s}: {count:3d} ❌ 缺失")
        elif count != expected:
            print(f"  {cat:30s}: {count:3d} ⚠️  不足 (期望 {expected})")
            print(f"    索引: {category_indices.get(cat, [])}")
        else:
            print(f"  {cat:30s}: {count:3d} ✓")
    
    # 检查unknown
    unknown_count = len(category_indices.get('unknown', []))
    print(f"\n  unknown: {unknown_count} 个样本")
    
    # 检查是否有其他无效类别
    all_categories = set(category_indices.keys())
    invalid = all_categories - set(OPENSET_CATEGORIES_14) - {'unknown'}
    if invalid:
        print(f"\n⚠️  其他类别: {invalid}")
        for cat in invalid:
            print(f"  {cat}: {len(category_indices[cat])} 个样本")



