#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查数据集的类别样本数和索引顺序
"""
from datasets import load_from_disk
from collections import defaultdict
import os
import re
# 读取标准类别列表（30个类别，用于ship30_coco数据集）
with open('/data/ljx/visualRft/classification/val_data/ships.txt', 'r') as f:
    VALID_CATEGORIES_30 = [line.strip() for line in f if line.strip()]

# OpenSet数据集的14个类别（来自grpo_classification_ood_openset.py）
OPENSET_CATEGORIES_14 = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier",
    "tugboat"
]

# unknown是OpenSet数据集的有效类别
OPENSET_CATEGORIES_WITH_UNKNOWN = OPENSET_CATEGORIES_14 + ['unknown']
VALID_CATEGORIES_WITH_UNKNOWN = VALID_CATEGORIES_30 + ['unknown']

# 需要检查的数据集
DATASETS = {
    'Ship30_OpenSet_1shot': '/data/ljx/visualRft/share_data/Ship_OpenSet_1shot/train',
    'Ship30_OpenSet_2shot': '/data/ljx/visualRft/share_data/Ship_OpenSet_2shot/train',
    'Ship30_OpenSet_4shot': '/data/ljx/visualRft/share_data/Ship_OpenSet_4shot/train',
    'Ship30_OpenSet_8shot': '/data/ljx/visualRft/share_data/Ship_OpenSet_8shot/train',
    'Ship30_OpenSet_16shot': '/data/ljx/visualRft/share_data/Ship_OpenSet_16shot/train',
    'shot_1': '/data/ljx/visualRft/share_data/ship30_coco/shot_1_cot_new/train',
    'shot_2': '/data/ljx/visualRft/share_data/ship30_coco/shot_2_cot_new/train',
    'shot_4': '/data/ljx/visualRft/share_data/ship30_coco/shot_4_cot_new/train',
    'shot_5': '/data/ljx/visualRft/share_data/ship30_coco/shot_5_cot_new/train',
    'shot_8': '/data/ljx/visualRft/share_data/ship30_coco/shot_8_cot_new/train',
    'shot_10': '/data/ljx/visualRft/share_data/ship30_coco/shot_10_cot_new/train',
    'shot_16': '/data/ljx/visualRft/share_data/ship30_coco/shot_16_cot_new/train',
}

def check_dataset(name, path):
    """检查单个数据集"""
    if not os.path.exists(path):
        print(f"\n{name:30s}: ❌ 不存在")
        return
    
    try:
        ds = load_from_disk(path)
        
        # 判断是OpenSet数据集（14类）还是ship30_coco数据集（30类）
        is_openset = 'OpenSet' in name
        is_detection = 'shot_' in name or 'coco' in path.lower()
        
        if is_openset:
            valid_categories = OPENSET_CATEGORIES_14
            valid_with_unknown = OPENSET_CATEGORIES_WITH_UNKNOWN
            num_categories = 14
        elif is_detection:
            # 检测数据集使用30个类别
            valid_categories = VALID_CATEGORIES_30
            valid_with_unknown = VALID_CATEGORIES_30  # 检测数据集没有unknown
            num_categories = 30
        else:
            valid_categories = VALID_CATEGORIES_30
            valid_with_unknown = VALID_CATEGORIES_WITH_UNKNOWN
            num_categories = 30
        
        # 从名称推断期望的shot数
        if '1shot' in name or 'shot_1' in name:
            expected = 1
        elif '2shot' in name or 'shot_2' in name:
            expected = 2
        elif '4shot' in name or 'shot_4' in name:
            expected = 4
        elif '5shot' in name or 'shot_5' in name:
            expected = 5
        elif '8shot' in name or 'shot_8' in name:
            expected = 8
        elif '10shot' in name or 'shot_10' in name:
            expected = 10
        elif '16shot' in name or 'shot_16' in name:
            expected = 16
        else:
            expected = None
        
        category_indices = defaultdict(list)
        invalid_categories = []
        
        # 检查是否是检测数据集（从problem字段提取类别）
        is_detection_dataset = 'shot_' in name or 'coco' in path.lower()
        
        for i, sample in enumerate(ds):
            if is_detection_dataset:
                # 检测数据集：从problem字段提取类别
                problem = sample.get('problem', '')
                # 查找 "the category 'xxx' in the image" 格式
                # 匹配 "the category 'xxx'" 或 "category 'xxx'"
                match = re.search(r"(?:the\s+)?category\s+['\"]([^'\"]+)['\"]", problem, re.IGNORECASE)
                if match:
                    category = match.group(1)
                    category_indices[category].append(i)
                else:
                    # 如果找不到，记录问题
                    if i < 3:  # 只记录前3个失败的样本用于调试
                        pass  # 不打印，避免干扰输出
            else:
                # 分类数据集：从solution字段提取类别
                solution = sample['solution']
                if '<answer>' in solution and '</answer>' in solution:
                    start = solution.find('<answer>') + len('<answer>')
                    end = solution.find('</answer>')
                    category = solution[start:end].strip()
                    category_indices[category].append(i)
                    
                    # 检查是否是无效类别（排除unknown，因为OpenSet数据集需要它）
                    if category not in valid_with_unknown:
                        category_lower = category.lower()
                        if category_lower not in [c.lower() for c in valid_with_unknown]:
                            if category not in invalid_categories:
                                invalid_categories.append(category)
        
        # 检查结果
        issues = []
        
        # 检查无效类别
        if invalid_categories:
            issues.append(f"无效类别: {len(invalid_categories)} 个")
        
        # 检查样本数（只检查标准类别，不包括unknown）
        if expected:
            missing_categories = []  # 缺失的类别
            wrong_count = []  # 样本数不对的类别
            
            for cat in valid_categories:
                count = len(category_indices.get(cat, []))
                if count == 0:
                    missing_categories.append(cat)
                elif count != expected:
                    wrong_count.append((cat, count, expected))
            
            if missing_categories:
                issues.append(f"缺失类别: {len(missing_categories)} 个")
            if wrong_count:
                issues.append(f"样本数异常: {len(wrong_count)} 个类别")
        
        # 检查索引顺序（每个类别的索引应该是连续的）
        # 注意：OpenSet数据集可能不是按类别顺序排列的，所以不检查索引连续性
        if not is_openset:
            order_issues = []
            for cat in valid_categories:
                indices = sorted(category_indices.get(cat, []))
                if len(indices) > 1:
                    # 检查是否连续
                    for i in range(len(indices) - 1):
                        if indices[i+1] - indices[i] != 1:
                            order_issues.append(cat)
                            break
            
            if order_issues:
                issues.append(f"索引不连续: {len(order_issues)} 个类别")
        
        # 输出结果
        if issues:
            print(f"{name:30s}: ⚠️  {'; '.join(issues)}")
            if invalid_categories:
                print(f"{'':30s}   无效类别: {invalid_categories[:3]}{'...' if len(invalid_categories) > 3 else ''}")
            if expected and missing_categories:
                print(f"{'':30s}   缺失类别: {missing_categories[:5]}{'...' if len(missing_categories) > 5 else ''}")
            if expected and wrong_count:
                print(f"{'':30s}   样本数异常示例: {wrong_count[0][0]} ({wrong_count[0][1]}/{wrong_count[0][2]})")
        else:
            unknown_count = len(category_indices.get('unknown', []))
            expected_total = expected * num_categories if expected else '未知'
            print(f"{name:30s}: ✓ 正常 (总样本: {len(ds)}, {num_categories}类×{expected}shot={expected_total}, unknown: {unknown_count})")
            
    except Exception as e:
        print(f"{name:30s}: ❌ 错误 - {str(e)[:50]}")

def main():
    print("="*70)
    print("快速检查数据集")
    print("="*70)
    
    for name, path in DATASETS.items():
        print(f"检查 {name}...", end='', flush=True)
        check_dataset(name, path)
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

