import json
import os

# 你的distance文件路径
base_path = "/data/ljx/visualRft/share_data"
shot_configs = [1, 2, 4, 5, 8, 10, 16]

for shot in shot_configs:
    file_path = f"{base_path}/Ship_OpenSet_{shot}shot_distance_nearest.json"
    
    if not os.path.exists(file_path):
        print(f"❌ {shot}-shot: 文件不存在")
        continue
    
    with open(file_path) as f:
        data = json.load(f)
    
    # 分离ID和OOD
    id_distances = []
    ood_distances = []
    
    for idx, info in data.items():
        dist = info["distance"]
        # 假设你的数据里有label或者可以从ground_truth判断
        # 这里需要你提供如何区分ID/OOD的逻辑
        # 暂时先全部统计
        id_distances.append(dist)
    
    print(f"\n{'='*50}")
    print(f"📊 {shot}-shot 分析:")
    print(f"{'='*50}")
    print(f"总样本数: {len(id_distances)}")
    print(f"Distance范围: [{min(id_distances):.3f}, {max(id_distances):.3f}]")
    print(f"Distance均值: {sum(id_distances)/len(id_distances):.6f}")
    print(f"Distance中位数: {sorted(id_distances)[len(id_distances)//2]:.6f}")
    
    # 分档统计
    thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]
    print(f"\n分档统计:")
    prev = 0
    for th in thresholds:
        count = sum(1 for d in id_distances if prev <= d < th)
        pct = count / len(id_distances) * 100
        print(f"  [{prev:.2f}, {th:.2f}): {count:4d} ({pct:5.1f}%)")
        prev = th
    count = sum(1 for d in id_distances if d >= prev)
    pct = count / len(id_distances) * 100
    print(f"  >={prev:.2f}:      {count:4d} ({pct:5.1f}%)")
