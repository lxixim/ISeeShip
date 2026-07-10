#!/usr/bin/env python3
# split_shot_ship.py
import os
import json
import shutil
import random
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description='船舶 XML → 1/2/4/8/16 shot 分层子集')
    parser.add_argument("--xml_dir", required=True, help="Pascal VOC XML 文件夹")
    parser.add_argument("--img_dir", required=True, help="JPEG 图片文件夹")
    parser.add_argument("--out_dir", default="ship_shot", help="输出根目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--shots", nargs='+', type=int, default=[1, 2, 4, 8, 16],
                        help="要生成的 shot 档，例：1 2 4 8 16")
    return parser.parse_args()


def scan_cls(xml_path):
    tree = ET.parse(xml_path)
    return [obj.find('name').text.strip() for obj in tree.getroot().iter('object')]


def find_real_img_path(img_dir, base):
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        candidate = os.path.join(img_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"找不到图片：{base}（.jpg/.jpeg/.png/.webp 都不存在）")


def build_coco_subset(base_list, xml_dir, img_dir, out_dir, cls_filter=None):
    """把 base_list 里的图片拷出来并生成 COCO 格式 instances.json"""
    os.makedirs(os.path.join(out_dir, "Annotations"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "JPEGImages"), exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": []}
    cat2id, ann_id = {}, 0

    for base in base_list:
        cls_list = scan_cls(os.path.join(xml_dir, base + '.xml'))
        main_cls = max(set(cls_list), key=cls_list.count) if cls_list else 'unknown'
        if cls_filter is not None and main_cls != cls_filter:
            continue   # 仅用于验证集只保留主类

        # 拷贝文件
        xml_src = os.path.join(xml_dir, base + '.xml')
        img_src = find_real_img_path(img_dir, base)
        dst_xml = os.path.join(out_dir, 'Annotations', base + '.xml')
        dst_img = os.path.join(out_dir, 'JPEGImages', base + os.path.splitext(img_src)[1])
        shutil.copy(xml_src, dst_xml)
        shutil.copy(img_src, dst_img)

        # 写 COCO
        tree = ET.parse(xml_src)
        root = tree.getroot()
        img_name = root.find('filename').text
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)
        img_id = len(coco["images"])
        coco["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})

        if main_cls not in cat2id:
            cat2id[main_cls] = len(cat2id)
            coco["categories"].append({"id": cat2id[main_cls], "name": main_cls})

        for obj in root.iter('object'):
            obj_name = obj.find('name').text.strip()
            if obj_name != main_cls:  # 训练/验证都只保留主类框
                continue
            x1, y1, x2, y2 = [int(float(obj.find('bndbox').find(p).text)) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cat2id[main_cls],
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0
            })
            ann_id += 1

    json.dump(coco, open(os.path.join(out_dir, 'instances.json'), 'w'), indent=2)
    return coco


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.random_seed)

    # 1. 扫全量 & 按类分组
    cls2imgs = defaultdict(list)
    for xml_file in os.listdir(args.xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        base = xml_file[:-4]
        cls_list = scan_cls(os.path.join(args.xml_dir, xml_file))
        main_cls = Counter(cls_list).most_common(1)[0][0] if cls_list else 'unknown'
        cls2imgs[main_cls].append(base)
    exclude_cls = {'bulk cargo carrier', 'kayak', 'general cargo', 'submarine', 'buoy'}
    for ec in exclude_cls:
        if ec in cls2imgs:
            print(f"[排除] 类别 {ec} 共 {len(cls2imgs[ec])} 张，已剔除")
            cls2imgs.pop(ec)

    # 2. 每类划分训练池 / 验证集
    train_pool, val_pool = defaultdict(list), defaultdict(list)
    for cls, imgs in cls2imgs.items():
        if len(imgs) < 2:
            train_pool[cls] = imgs
        else:
            tr, va = train_test_split(imgs, test_size=args.val_ratio,
                                      random_state=args.random_seed, shuffle=True)
            train_pool[cls], val_pool[cls] = tr, va

    # 3. 生成统一验证集
    val_dir = os.path.join(args.out_dir, 'val')
    build_coco_subset([base for lst in val_pool.values() for base in lst],
                      args.xml_dir, args.img_dir, val_dir)
    print(f'✅ 验证集完成，路径：{val_dir}')

    # 4. 对每个 shot 档从训练池采样
    for shot in args.shots:
        shot_dir = os.path.join(args.out_dir, f"shot_{shot}")
        shot_imgs = []
        for cls, imgs in train_pool.items():
            n_pick = min(shot, len(imgs))
            # ↓↓↓ 新增：直接告诉你够不够 ↓↓↓
            if len(imgs) < shot:
                print(f"[shot={shot}] 类别 {cls} 只有 {len(imgs)} 张，不足 {shot} 张，全拿！")
            picked = random.sample(imgs, n_pick) if len(imgs) >= n_pick else imgs
            shot_imgs.extend(picked)

        build_coco_subset(shot_imgs, args.xml_dir, args.img_dir, shot_dir)
        print(f'✅ shot_{shot} 完成：{len(shot_imgs)} 张图')

    # 5. 输出类别-图片-XML 列表（基于训练池）
    cls_list_file = os.path.join(args.out_dir, "class_image_list.txt")
    with open(cls_list_file, "w", encoding="utf-8") as f_out:
        for cls, imgs in train_pool.items():
            f_out.write(f"【{cls}】共 {len(imgs)} 张\n")
            for base in imgs:
                jpg_path = find_real_img_path(args.img_dir, base)
                xml_path = os.path.join(args.xml_dir, base + '.xml')
                f_out.write(f"{jpg_path}\t{xml_path}\n")
            f_out.write("\n")
    print(f'✅ 类别-图片-XML 列表已写入：{cls_list_file}')


if __name__ == '__main__':
    main()