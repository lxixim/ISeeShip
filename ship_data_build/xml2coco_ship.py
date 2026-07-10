#!/usr/bin/env python3
# xml2coco_ship.py
import xml.etree.ElementTree as ET
import json, os, shutil, argparse, random
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Ship XML -> COCO train/val')
    parser.add_argument('--xml_dir', required=True, help='Pascal VOC XML folder')
    parser.add_argument('--img_dir', required=True, help='JPEG images folder')
    parser.add_argument('--out_dir', default='ship_coco', help='output COCO dir')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio')
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()

def scan_classes(xml_dir):
    """扫全 XML -> 得到所有类别"""
    cls_set = set()
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'): continue
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        for obj in tree.getroot().iter('object'):
            cls_set.add(obj.find('name').text.strip())
    return sorted(cls_set)

def xml2coco(xml_dir, img_dir, class_names, img_list):
    """把 img_list 里的图片+XML 转成 COCO 字典"""
    imgs, anns, img_id, ann_id = [], [], 0, 0
    cls2id = {c: i for i, c in enumerate(class_names)}
    for xml_base in img_list:
        xml_path = os.path.join(xml_dir, xml_base + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find('filename').text
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)
        imgs.append({"id": img_id, "file_name": img_name, "width": w, "height": h})
        for obj in root.iter('object'):
            cls = obj.find('name').text.strip()
            if cls not in cls2id: continue   # 防御
            cls_id = cls2id[cls]
            xmlbox = obj.find('bndbox')
            x1, y1, x2, y2 = [int(float(xmlbox.find(p).text)) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
            anns.append({"id": ann_id, "image_id": img_id, "category_id": cls_id,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            ann_id += 1
        img_id += 1
    cats = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    return {"images": imgs, "annotations": anns, "categories": cats}

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(f"{args.out_dir}/annotations", exist_ok=True)
    os.makedirs(f"{args.out_dir}/images/train", exist_ok=True)
    os.makedirs(f"{args.out_dir}/images/val", exist_ok=True)

    # 1. 扫描全部类别
    class_names = scan_classes(args.xml_dir)
    print(f"发现 {len(class_names)} 类：{class_names}")
    if not class_names:
        raise ValueError("未在 XML 中找到任何类别，请检查 Annotations 目录。")

    # 2. 建立 图片base -> 主类别（用最多框的类代表）
    img2cats = defaultdict(set)
    for xml_file in os.listdir(args.xml_dir):
        if not xml_file.endswith('.xml'): continue
        base = xml_file[:-4]
        tree = ET.parse(os.path.join(args.xml_dir, xml_file))
        for obj in tree.getroot().iter('object'):
            img2cats[base].add(obj.find('name').text.strip())
    img_list = list(img2cats.keys())

    # 3. 修复：独苗类全进训练，其余再分层
    def max_label(img):
        return max(img2cats[img], key=lambda c: class_names.index(c))
    cls_cnt = Counter([max_label(img) for img in img_list])
    single_imgs = [img for img in img_list if cls_cnt[max_label(img)] == 1]
    multi_imgs  = [img for img in img_list if cls_cnt[max_label(img)] > 1]

    if len(multi_imgs) < 2:
        # 全是独苗 → 随机拆
        train_imgs, val_imgs = train_test_split(img_list, test_size=args.val_ratio, random_state=args.random_seed)
    else:
        # 独苗全进训练；其余分层抽
        multi_train, multi_val = train_test_split(
            multi_imgs, test_size=args.val_ratio, random_state=args.random_seed,
            stratify=[max_label(img) for img in multi_imgs]
        )
        train_imgs = single_imgs + multi_train
        val_imgs   = multi_val

    # 4. 生成 COCO JSON
    train_coco = xml2coco(args.xml_dir, args.img_dir, class_names, train_imgs)
    val_coco = xml2coco(args.xml_dir, args.img_dir, class_names, val_imgs)
    json.dump(train_coco, open(f"{args.out_dir}/annotations/instances_train.json", "w"), indent=2)
    json.dump(val_coco, open(f"{args.out_dir}/annotations/instances_val.json", "w"), indent=2)

    # 5. 复制图片到对应文件夹
    for img_base in train_imgs:
        shutil.copy(os.path.join(args.img_dir, img_base + '.jpg'),
                    f"{args.out_dir}/images/train/")
    for img_base in val_imgs:
        shutil.copy(os.path.join(args.img_dir, img_base + '.jpg'),
                    f"{args.out_dir}/images/val/")

    print("✅ 完成！目录结构：")
    print(f"  {args.out_dir}/annotations/instances_train.json  # 训练标注")
    print(f"  {args.out_dir}/annotations/instances_val.json    # 验证标注")
    print(f"  {args.out_dir}/images/train/                     # 训练图片")
    print(f"  {args.out_dir}/images/val/                       # 验证图片")

if __name__ == '__main__':
    main()