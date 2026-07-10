#!/usr/bin/env python3
# subsample_ship.py
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from sklearn.model_selection import train_test_split

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="船舶 XML 数据集：先扫全类，再分层抽小份")
    parser.add_argument("--xml_dir", required=True, help="Pascal VOC XML 文件夹")
    parser.add_argument("--img_dir", required=True, help="JPEG 图片文件夹")
    parser.add_argument("--out_dir", default="ship_mini", help="输出小份目录")
    parser.add_argument("--max_imgs", type=int, default=1500, help="想保留多少张图")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()

def scan_classes(xml_dir):
    """扫全 XML 得到所有类别"""
    cls_set = set()
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        for obj in tree.getroot().iter('object'):
            cls_set.add(obj.find('name').text.strip())
    return sorted(cls_set)

def get_img2cls(xml_dir):
    """返回 图片base -> 主类别（用最多框的类代表）"""
    img2cls = {}
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        base = xml_file[:-4]
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        cls_list = [obj.find('name').text.strip() for obj in tree.getroot().iter('object')]
        # 用出现次数最多的类代表这张图片
        img2cls[base] = max(set(cls_list), key=cls_list.count) if cls_list else 'unknown'
    return img2cls

def main():
    args = parse_args()
    os.makedirs(f"{args.out_dir}/Annotations", exist_ok=True)
    os.makedirs(f"{args.out_dir}/JPEGImages", exist_ok=True)

    # 1. 扫描全量类别
    class_names = scan_classes(args.xml_dir)
    print(f"发现 {len(class_names)} 类：{class_names}")
    if not class_names:
        raise ValueError("未在 XML 中找到任何类别，请检查 Annotations 目录。")

    # 2. 建立 图片base -> 主类别
    img2cls = get_img2cls(args.xml_dir)
    xml_list = list(img2cls.keys())          # 所有图片 base 名

    # 3. 统计每类出现次数
    cls_cnt = Counter(img2cls[base] for base in xml_list)

    # 4. 拆出“独苗类”（只有 1 张）和“多图类”
    single_imgs = [base for base in xml_list if cls_cnt[img2cls[base]] == 1]
    multi_imgs  = [base for base in xml_list if cls_cnt[img2cls[base]] > 1]

    # 5. 分层采样：独苗全部进训练，其余再抽
    remain = args.max_imgs - len(single_imgs)
    if remain <= 0:
        # 如果独苗就已经超过想保留的数量，随机降采样
        chosen = random.sample(xml_list, args.max_imgs)
    else:
        # 多图部分再分层抽
        if len(multi_imgs) == 0:
            chosen = random.sample(xml_list, args.max_imgs)
        else:
            multi_chosen, _ = train_test_split(
                multi_imgs, train_size=remain, random_state=args.random_seed,
                stratify=[img2cls[base] for base in multi_imgs]
            )
            chosen = single_imgs + multi_chosen

    # 6. 复制小份
    for base in chosen:
        # XML
        shutil.copy(os.path.join(args.xml_dir, base + '.xml'),
                    os.path.join(args.out_dir, 'Annotations', base + '.xml'))
        # JPG
        shutil.copy(os.path.join(args.img_dir, base + '.jpg'),
                    os.path.join(args.out_dir, 'JPEGImages', base + '.jpg'))

    print(f"✅ 已复制 {len(chosen)} 张到 {args.out_dir}")
    # 打印类别分布
    out_cls_cnt = Counter(img2cls[base] for base in chosen)
    print("小份类别分布:", dict(out_cls_cnt))

if __name__ == '__main__':
    main()