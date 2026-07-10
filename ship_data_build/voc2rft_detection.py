#!/usr/bin/env python3
# voc2rft_all_detection.py
import json
import argparse
import os
from collections import defaultdict

TMPL = (
    "Detect all objects belonging to the category '{cls}' in the image, and provide the bounding boxes "
    "(between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\n"
    "If no object belonging to the category '{cls}' in the image, return 'No Objects'.\n"
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "The output answer format should be as follows:\n"
    "<think> ... </think> <answer>[{{'Position': [x1, y1, x2, y2], 'Confidence': number}}, ...]</answer>\n"
    "Please strictly follow the format."
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()

def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def scale2k(bbox, W, H):
    x1, y1, x2, y2 = bbox
    return [int(round(x1 / W * 1000)),
            int(round(y1 / H * 1000)),
            int(round(x2 / W * 1000)),
            int(round(y2 / H * 1000))]

def main():
    args = parse_args()
    coco = json.load(open(args.coco_json))
    id2cat = {c["id"]: c["name"] for c in coco["categories"]}
    id2img = {img["id"]: img for img in coco["images"]}

    # 1. 先聚合标注，再确定每张图的“主类”（出现次数最多的类）
    annos = defaultdict(lambda: defaultdict(list))   # img_id -> cat_id -> bbox_list
    # 统计各类出现次数
    img_cls_cnt = defaultdict(lambda: defaultdict(int))
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        xyxy = xywh2xyxy(ann["bbox"])
        xyxy_k = scale2k(xyxy, id2img[img_id]["width"], id2img[img_id]["height"])
        annos[img_id][cat_id].append({"Position": xyxy_k, "Confidence": 1.00})
        img_cls_cnt[img_id][cat_id] += 1

    # 2. 只生成主类样本
    records = []
    for img_id, img_info in id2img.items():
        if img_id not in img_cls_cnt:          # 无标注图跳过
            continue
        # 取出现次数最多的类作为主类
        main_cat_id = max(img_cls_cnt[img_id], key=img_cls_cnt[img_id].get)
        main_name = id2cat[main_cat_id]

        filename = img_info["file_name"]
        full_path = os.path.join(args.img_dir, filename)

        # 只收集主类的框
        bbox_list = annos[img_id].get(main_cat_id, [])
        answer = json.dumps(bbox_list, ensure_ascii=False) if bbox_list else "No Objects"

        records.append({
            "image": f"[IMAGE_SAVED_AS:{full_path}]",
            "problem": TMPL.format(cls=main_name),
            "solution": f"<answer>{answer}</answer>"
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ 完成！共 {len(records)} 条，已写入 {args.out_json}")

if __name__ == "__main__":
    main()