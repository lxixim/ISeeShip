#!/usr/bin/env python3
"""
诊断训练数据：检查OOD样本
"""

from datasets import DatasetDict
import json

DATA_PATH = '/data/ljx/visualRft/share_data/Ship30_OpenSet_4shot_fixed'

print("="*80)
print("🔍 数据诊断")
print("="*80)

dataset = DatasetDict.load_from_disk(DATA_PATH)
train = dataset['train']

print(f"\n总样本: {len(train)}")

# 分离ID和OOD
id_samples = [s for s in train if json.loads(s['metadata'])['type'] == 'ID']
ood_samples = [s for s in train if json.loads(s['metadata'])['type'] == 'OOD']

print(f"ID样本: {len(id_samples)}")
print(f"OOD样本: {len(ood_samples)} ({100*len(ood_samples)/len(train):.1f}%)")

if len(ood_samples) == 0:
    print("\n❌ 致命错误：没有OOD样本！")
    print("这就是为什么OOD拒识率为0%")
    print("\n解决方案：")
    print("  python /mnt/user-data/outputs/create_data_4shot_enhanced.py")
    exit(1)

# 检查OOD标注
print(f"\n{'='*80}")
print("📊 OOD样本检查")
print("="*80)

ood_with_unknown = 0
ood_without_unknown = []

for s in ood_samples:
    if '<answer>unknown</answer>' in s['solution']:
        ood_with_unknown += 1
    else:
        meta = json.loads(s['metadata'])
        ood_without_unknown.append(meta.get('original_category', 'N/A'))

print(f"\n✅ 标注为unknown: {ood_with_unknown}/{len(ood_samples)}")

if ood_without_unknown:
    print(f"❌ 未标注为unknown: {len(ood_without_unknown)}")
    print(f"   类别: {', '.join(ood_without_unknown[:5])}")

# 显示示例
print(f"\n{'='*80}")
print("📝 OOD样本示例")
print("="*80)

for i in range(min(3, len(ood_samples))):
    meta = json.loads(ood_samples[i]['metadata'])
    orig_cat = meta.get('original_category', 'N/A')
    solution = ood_samples[i]['solution']
    
    print(f"\n[样本 {i+1}]")
    print(f"  原始类别: {orig_cat}")
    print(f"  Solution: {solution[:120]}...")
    print(f"  长度: {len(solution)} 字符")

# 统计长度
print(f"\n{'='*80}")
print("📏 Solution长度统计")
print("="*80)

id_lengths = [len(s['solution']) for s in id_samples]
ood_lengths = [len(s['solution']) for s in ood_samples]

print(f"\nID样本:")
print(f"  平均: {sum(id_lengths)/len(id_lengths):.0f} 字符")
print(f"  最大: {max(id_lengths)} 字符")

print(f"\nOOD样本:")
print(f"  平均: {sum(ood_lengths)/len(ood_lengths):.0f} 字符")
print(f"  最大: {max(ood_lengths)} 字符")

# 诊断结论
print(f"\n{'='*80}")
print("💡 诊断结论")
print("="*80)

if len(ood_samples) < 10:
    print("\n❌ 问题：OOD样本太少")
    print(f"   当前: {len(ood_samples)} 个")
    print(f"   推荐: 15-25 个")
    print("\n解决方案：")
    print("   python /mnt/user-data/outputs/create_data_4shot_enhanced.py")

elif ood_with_unknown < len(ood_samples):
    print("\n❌ 问题：OOD样本标注错误")
    print(f"   {len(ood_samples) - ood_with_unknown} 个样本没有标注为unknown")
    print("\n解决方案：")
    print("   重新生成数据")

elif max(ood_lengths) > 150:
    print("\n⚠️  问题：OOD reasoning太长")
    print(f"   最大: {max(ood_lengths)} 字符")
    print(f"   推荐: <100 字符")
    print("\n解决方案：")
    print("   python /mnt/user-data/outputs/create_data_4shot_enhanced.py")

else:
    print("\n✅ 数据质量良好")
    print(f"   OOD样本: {len(ood_samples)} 个")
    print(f"   标注正确: {ood_with_unknown}/{len(ood_samples)}")
    print(f"   长度合理: {max(ood_lengths)} 字符")
    
    print("\n但SFT后OOD拒识率为0%，可能原因：")
    print("   1. OOD样本比例偏低（当前{:.1f}%，推荐25-30%）".format(100*len(ood_samples)/len(train)))
    print("   2. 训练epochs不够（当前5，可尝试8-10）")
    print("   3. 直接进行GRPO训练（GRPO会强化OOD拒识）")
    
    print("\n推荐方案：")
    print("   方案A：增加OOD样本到25个（30%）")
    print("     python /mnt/user-data/outputs/create_data_4shot_enhanced.py")
    print("   方案B：直接进行GRPO训练")
    print("     cd /data/ljx/visualRft/src/virft")
    print("     bash train_grpo.sh")

print("="*80)