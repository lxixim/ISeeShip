#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从现有 Ship_OpenSet_*shot 数据集中抽取 5shot 和 10shot 数据集

- ID 类别：14 个（与 grpo_classification_ood_openset.py 中一致）
- OOD / unknown：全部保留，不做下采样

来源：
- 5shot 从 8shot 数据集抽取
- 10shot 从 16shot 数据集抽取
"""

import os
import random
from collections import defaultdict

from datasets import load_from_disk, Dataset, DatasetDict

# 固定随机种子，保证可复现
random.seed(42)

# 14 个 ID 类别（OpenSet，需与 grpo_classification_ood_openset.py 中 ID_CATEGORIES 一致）
OPENSET_CATEGORIES_14 = [
    "bulk_carrier",
    "container_ship",
    "general_cargo_ship",
    "passenger_cargo_ship",
    "fishing_boat",
    "sailing_trimaran",
    "chemical_tanker",
    "crude_oil_tanker",
    "oil_products_tanker",
    "LNG_tanker",
    "LPG_tanker",
    "kayak",
    "heavy_load_carrier",
    "tugboat",
]

UNKNOWN_LABEL = "unknown"

# 源数据集路径（支持带 _cot 和不带 _cot 的数据集）
SOURCE_DATASETS_COT = {
    8: "/data/ljx/visualRft/share_data/Ship_OpenSet_8shot_cot/train",   # 用来抽 5shot (COT)
    16: "/data/ljx/visualRft/share_data/Ship_OpenSet_16shot_cot/train",  # 用来抽 10shot (COT)
}

SOURCE_DATASETS_NO_COT = {
    8: "/data/ljx/visualRft/share_data/Ship_OpenSet_8shot/train",   # 用来抽 5shot (非COT)
    16: "/data/ljx/visualRft/share_data/Ship_OpenSet_16shot/train",  # 用来抽 10shot (非COT)
}

# 输出路径
OUTPUT_BASE = "/data/ljx/visualRft/share_data"


# OOD 配置（参考 create_openset_dataset_with_ood.get_ood_config，并为 5/10 补充）
OOD_CONFIGS = {
    1:  {"ood_ratio": 0.25, "min_ood_samples": 5,  "min_ood_coverage": 0.5},
    2:  {"ood_ratio": 0.25, "min_ood_samples": 9,  "min_ood_coverage": 0.7},
    4:  {"ood_ratio": 0.40, "min_ood_samples": 30, "min_ood_coverage": 1.0},
    # 5/10 为我们额外设计的配置，使曲线在 4/8/16 之间平滑过渡
    5:  {"ood_ratio": 0.38, "min_ood_samples": 35, "min_ood_coverage": 1.0},
    8:  {"ood_ratio": 0.35, "min_ood_samples": 40, "min_ood_coverage": 1.0},
    10: {"ood_ratio": 0.32, "min_ood_samples": 50, "min_ood_coverage": 1.0},
    16: {"ood_ratio": 0.30, "min_ood_samples": 68, "min_ood_coverage": 1.0},
}


def get_ood_config(k_shot: int):
    """获取 OOD 配置，默认给一个相对保守的值"""
    return OOD_CONFIGS.get(
        k_shot,
        {"ood_ratio": 0.20, "min_ood_samples": 20, "min_ood_coverage": 1.0},
    )


def extract_category_from_solution(solution: str):
    """从 solution 中提取 <answer>category</answer> 里的类别名"""
    if not isinstance(solution, str):
        return None
    start_tag = "<answer>"
    end_tag = "</answer>"
    if start_tag not in solution or end_tag not in solution:
        return None
    start = solution.find(start_tag) + len(start_tag)
    end = solution.find(end_tag)
    return solution[start:end].strip()


def group_samples(dataset):
    """按类别分组样本，并单独收集 unknown 样本"""
    id_samples = defaultdict(list)
    unknown_samples = []
    invalid_samples = []

    for sample in dataset:
        cat = extract_category_from_solution(sample.get("solution", ""))
        if cat is None:
            invalid_samples.append(sample)
            continue

        if cat == UNKNOWN_LABEL:
            unknown_samples.append(sample)
        elif cat in OPENSET_CATEGORIES_14:
            id_samples[cat].append(sample)
        else:
            # 既不是 14 个 ID，也不是 unknown，视为无效
            invalid_samples.append(sample)

    return id_samples, unknown_samples, invalid_samples


def sample_ood_samples(unknown_samples, id_total: int, k_shot: int):
    """根据 OOD 配置，从 unknown 中抽取合适数量的 OOD 样本

    - 目标：保持与 get_ood_config(k_shot) 中的 ood_ratio 近似
    - 同时满足最小 OOD 数量约束
    """
    if not unknown_samples or id_total <= 0:
        return []

    cfg = get_ood_config(k_shot)
    ood_ratio = cfg.get("ood_ratio", 0.3)
    min_ood_samples = cfg.get("min_ood_samples", 20)

    # 期望的 OOD 数量：  ood_ratio = N_ood / (N_id + N_ood)
    # => N_ood = ood_ratio / (1 - ood_ratio) * N_id
    target_ood = int(ood_ratio * id_total / max(1e-6, (1 - ood_ratio)))

    # 应用最小 / 最大约束
    target_ood = max(target_ood, min_ood_samples)
    target_ood = min(target_ood, len(unknown_samples))

    if target_ood <= 0:
        return []

    # 随机下采样 OOD
    return random.sample(unknown_samples, target_ood)


def create_openset_dataset(src_path: str, target_shot: int, output_name: str):
    """从指定路径的 OpenSet 数据集中抽取新的 shot 数据集

    - 只对 14 个 ID 类别做下采样
    - unknown/OOD 样本按比例采样
    """
    print("=" * 70)
    print(f"从 {src_path} 抽取 {target_shot}-shot，输出为 {output_name}")
    print("=" * 70)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"源数据集不存在: {src_path}")

    ds = load_from_disk(src_path)
    print(f"源数据集样本数: {len(ds)}")

    id_samples, unknown_samples, invalid_samples = group_samples(ds)

    print("\nID 类别样本统计 (源数据):")
    for cat in OPENSET_CATEGORIES_14:
        cnt = len(id_samples.get(cat, []))
        print(f"  {cat:25s}: {cnt:4d} 个样本")
    print(f"\nunknown 样本数 : {len(unknown_samples)}")
    print(f"无效样本数   : {len(invalid_samples)}\n")

    # 下采样 14 个 ID 类别
    selected_samples = []
    for cat in OPENSET_CATEGORIES_14:
        samples = id_samples.get(cat, [])
        if not samples:
            print(f"⚠️  类别 {cat} 在源数据集中不存在，跳过")
            continue
        if len(samples) < target_shot:
            print(f"⚠️  类别 {cat} 只有 {len(samples)} 个样本 < {target_shot}，全部保留")
            chosen = samples  # 不足的类别全部用上
        else:
            chosen = random.sample(samples, target_shot)
        selected_samples.extend(chosen)

    id_total = len(selected_samples)

    # 按比例从 unknown 中采样 OOD
    sampled_ood = sample_ood_samples(unknown_samples, id_total=id_total, k_shot=target_shot)
    selected_samples.extend(sampled_ood)

    print("\n抽取结果:")
    print(f"  ID 类别总样本: {id_total}")
    print(f"  unknown 样本 : {len(sampled_ood)} / 原始 {len(unknown_samples)}")
    print(f"  合计样本数   : {len(selected_samples)}")

    # 构建新的 Dataset
    new_ds = Dataset.from_list(selected_samples)
    new_dd = DatasetDict({"train": new_ds})

    out_path = os.path.join(OUTPUT_BASE, output_name)
    os.makedirs(out_path, exist_ok=True)

    print(f"\n保存到 {out_path} ...")
    new_dd.save_to_disk(out_path)
    print("保存完成！\n")


def main():
    # 5shot: 从 8shot 抽取（带 _cot 版本）
    create_openset_dataset(
        src_path=SOURCE_DATASETS_COT[8],
        target_shot=5,
        output_name="Ship_OpenSet_5shot_cot",
    )

    # 5shot: 从 8shot 抽取（不带 _cot 版本）
    create_openset_dataset(
        src_path=SOURCE_DATASETS_NO_COT[8],
        target_shot=5,
        output_name="Ship_OpenSet_5shot",
    )

    # 10shot: 从 16shot 抽取（带 _cot 版本）
    create_openset_dataset(
        src_path=SOURCE_DATASETS_COT[16],
        target_shot=10,
        output_name="Ship_OpenSet_10shot_cot",
    )

    # 10shot: 从 16shot 抽取（不带 _cot 版本）
    create_openset_dataset(
        src_path=SOURCE_DATASETS_NO_COT[16],
        target_shot=10,
        output_name="Ship_OpenSet_10shot",
    )


if __name__ == "__main__":
    main()