#!/usr/bin/env python3
"""
precompute_distance_nearest_neighbor.py - 最近邻距离版
"""

import os
import json
import re
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import DatasetDict
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier",
    "tugboat"
]

class DistanceComputerNearest:
    """最近邻距离版"""
    
    def __init__(self, model_path, device="cuda", max_pixels=401408, min_pixels=3136):
        print(f"�� 加载模型: {model_path}")
        self.device = device
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = min_pixels
        
        self.id_prototypes = {}
        self.feature_dim = None
        print("✅ 模型加载完成")
    
    def _safe_get_image(self, image_data):
        """安全获取图像对象"""
        if isinstance(image_data, str):
            return Image.open(image_data).convert('RGB')
        elif isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, dict) and 'path' in image_data:
            return Image.open(image_data['path']).convert('RGB')
        else:
            raise ValueError(f"不支持的图像格式: {type(image_data)}")
    
    def _get_nested_value(self, data, key_path):
        """支持嵌套字典访问"""
        if 'solution' in key_path.lower() or key_path == 'solution':
            if isinstance(data, dict) and 'solution' in data:
                solution = data['solution']
                match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
                if match:
                    return match.group(1).strip()
            return None
        
        if '.' not in key_path:
            value = data.get(key_path) if isinstance(data, dict) else getattr(data, key_path, None)
            return value.strip() if isinstance(value, str) else value
        
        keys = key_path.split('.')
        value = data
        for key in keys:
            if isinstance(value, str) and (value.strip().startswith('{') or value.strip().startswith('[')):
                try:
                    value = json.loads(value)
                except:
                    return None
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
            if value is None:
                return None
        
        return value.strip() if isinstance(value, str) else value
    
    @torch.no_grad()
    def extract_pure_visual_feature(self, image, verbose=False):
        """提取纯视觉特征"""
        image = self._safe_get_image(image)
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What ship?"}
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            if HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(
                    text=[text], images=[image],
                    padding=True, return_tensors="pt"
                ).to(self.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-2]
            
            visual_features = []
            if hasattr(inputs, 'pixel_values') and hasattr(inputs, 'image_grid_thw'):
                start_idx = 1
                for i, (t, h, w) in enumerate(inputs.image_grid_thw):
                    num_visual = t * h * w
                    visual_tokens = hidden_state[:, start_idx:start_idx + num_visual]
                    visual_features.append(visual_tokens)
                    start_idx += num_visual
                
                visual_features = torch.cat(visual_features, dim=1)
                features = visual_features.mean(dim=1)
            else:
                features = hidden_state[:, 0]
            
            return F.normalize(features, dim=-1)
            
        except Exception as e:
            if verbose:
                print(f"    ❌ 特征提取失败: {e}")
            raise
    
    def build_prototypes(self, train_data, label_key="metadata.category", max_samples_per_class=10):
        """构建ID原型"""
        print("\n�� 构建ID原型...")
        
        # 自动检测label_key
        if len(train_data) > 0:
            example = train_data[0]
            if 'metadata' not in example and 'solution' in example:
                print(f"   ⚠️  自动切换到 'solution'")
                label_key = "solution"
        
        print(f"   数据集keys: {train_data[0].keys()}")
        print(f"   使用标签字段: '{label_key}'")
        
        # 收集样本
        class_to_indices = defaultdict(list)
        unknown_count = 0
        
        for idx, example in enumerate(tqdm(train_data, desc="   Collecting")):
            category = self._get_nested_value(example, label_key)
            
            if idx < 5:
                print(f"   样本 {idx}: category='{category}'")
            
            if category == "unknown" or category is None:
                unknown_count += 1
                continue
            
            if category in ID_CATEGORIES:
                class_to_indices[category].append(idx)
        
        print(f"   ✓ 排除 {unknown_count} 个OOD样本")
        print(f"   发现类别: {[(cls, len(indices)) for cls, indices in class_to_indices.items()]}")
        
        # 提取特征
        class_to_features = defaultdict(list)
        
        for idx, example in enumerate(tqdm(train_data, desc="   Extracting")):
            category = self._get_nested_value(example, label_key)
            
            if category not in ID_CATEGORIES:
                continue
            
            try:
                image = example['image']
                feature = self.extract_pure_visual_feature(image)
                class_to_features[category].append(feature.squeeze(0).cpu())
            except Exception as e:
                continue
        
        print(f"   类别统计: {[(cls, len(feats)) for cls, feats in class_to_features.items()]}")
        
        # 构建原型
        for cls_name in tqdm(ID_CATEGORIES, desc="   Building prototypes"):
            features = class_to_features.get(cls_name, [])
            
            if not features:
                continue
            
            features = torch.stack(features)
            
            # K-means聚类
            if len(features) >= 3:
                n_clusters = min(2, len(features) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                features_np = features.numpy()
                kmeans.fit(features_np)
                main_cluster_idx = np.argmax(np.bincount(kmeans.labels_))
                prototype = torch.tensor(kmeans.cluster_centers_[main_cluster_idx]).float()
            else:
                prototype = features.mean(dim=0)
            
            # 异常值过滤
            distances = torch.norm(features - prototype, dim=1)
            mean_dist, std_dist = distances.mean(), distances.std()
            inliers = features[distances < mean_dist + 2 * std_dist]
            
            if len(inliers) > 0:
                prototype = inliers.mean(dim=0)
            
            self.id_prototypes[cls_name] = F.normalize(prototype.unsqueeze(0), dim=-1)
        
        print(f"   ✓ 成功构建 {len(self.id_prototypes)} 个原型")
        return label_key
    
    def compute_distance(self, image_feature):
        """计算到最近ID原型的距离"""
        if not self.id_prototypes:
            return 1.0, "unknown", -1.0
        
        # 计算到所有ID原型的距离
        distances = []
        nearest_id = None
        min_distance = float('inf')
        
        for cls_name, proto in self.id_prototypes.items():
            # 欧氏距离
            dist = torch.norm(image_feature - proto.to(image_feature.device)).item()
            distances.append(dist)
            
            if dist < min_distance:
                min_distance = dist
                nearest_id = cls_name
        
        # 返回最小距离
        return min_distance, nearest_id, 0.0
    
    def process_dataset(self, dataset_path, output_path, label_key="metadata.category"):
        """处理数据集"""
        print(f"�� 加载数据集: {dataset_path}")
        dataset = DatasetDict.load_from_disk(dataset_path)
        
        if 'train' not in dataset:
            raise ValueError(f"数据集缺少 'train' 分片")
        
        train_data = dataset['train']
        print(f"   训练集样本数: {len(train_data)}")
        
        # 构建原型
        actual_label_key = self.build_prototypes(train_data, label_key=label_key)
        label_key = actual_label_key
        
        # 计算距离
        print("\n�� 计算最近邻距离...")
        results = {}
        
        for idx in tqdm(range(len(train_data)), desc="Processing images"):
            try:
                image = train_data[idx]['image']
                feature = self.extract_pure_visual_feature(image)
                distance, nearest_id, _ = self.compute_distance(feature)
                
                true_category = self._get_nested_value(train_data[idx], label_key)
                
                results[str(idx)] = {
                    "distance": round(distance, 4),
                    "nearest_id": nearest_id,
                    "true_category": true_category,
                }
            except Exception as e:
                results[str(idx)] = {
                    "distance": 0.5,
                    "nearest_id": "unknown",
                    "true_category": "unknown",
                    "error": str(e)
                }
        
        # 保存结果
        print(f"\n�� 保存结果: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 即时分析
        valid = [v for v in results.values() if "error" not in v]
        id_dist = [v["distance"] for v in valid if v["true_category"] in ID_CATEGORIES]
        ood_dist = [v["distance"] for v in valid if v["true_category"] == "unknown"]
        
        if id_dist and ood_dist:
            print(f"\n�� 分离度分析:")
            print(f"   ID样本: {len(id_dist)}, OOD样本: {len(ood_dist)}")
            print(f"   ID范围: [{min(id_dist):.2f}, {max(id_dist):.2f}], 均值: {np.mean(id_dist):.2f}")
            print(f"   OOD范围: [{min(ood_dist):.2f}, {max(ood_dist):.2f}], 均值: {np.mean(ood_dist):.2f}")
            
            gap = min(ood_dist) - max(id_dist)
            print(f"   �� ID-OOD间隙: {gap:.2f}")
            
            if gap > 0:
                print("   ✅ 分离成功！")
                print(f"   建议GRPO阈值:")
                print(f"     id_unknown_low: 0.05")
                print(f"     id_unknown_mid: {max(id_dist) + gap*0.3:.2f}")
                print(f"     id_unknown_high: {min(ood_dist) - gap*0.2:.2f}")
                print(f"     dist_threshold_mid: {max(id_dist) + gap*0.2:.2f}")
                print(f"     dist_threshold_high: {min(ood_dist) - gap*0.2:.2f}")
            else:
                print("   ⚠️  有重叠")
        
        print(f"\n✅ 完成!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="计算最近邻距离")
    parser.add_argument("--dataset_path", type=str, default="/data/ljx/visualRft/share_data/Ship_OpenSet_10shot")
    parser.add_argument("--model_path", type=str, default="/data/ljx/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--label_key", type=str, default="metadata.category")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = args.dataset_path.rstrip('/') + "_distance_nearest.json"
    
    computer = DistanceComputerNearest(args.model_path)
    computer.process_dataset(
        args.dataset_path,
        args.output_path,
        label_key=args.label_key
    )


if __name__ == "__main__":
    main()
