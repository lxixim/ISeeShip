#!/usr/bin/env python3
"""
tsne_from_imagefolder.py

直接从 /data/ljx/dataset/boat-29/images/ 这种目录结构提取特征
每个子文件夹对应一个类别，按 ID/OOD-Train/OOD-Reserved 分组

使用:
  # 提取特征 + 画图（第一次）
  python tsne_from_imagefolder.py \
    --image_dir  /data/ljx/dataset/boat-29/images \
    --base_model /data/ljx/Qwen2-VL-2B-Instruct \
    --grpo_model /data/ljx/.../checkpoint-xxx \
    --output_dir ./tsne_out \
    --n_per_class 80 \
    --save_feats

  # 特征已缓存，只重画（快）
  python tsne_from_imagefolder.py \
    --image_dir  /data/ljx/dataset/boat-29/images \
    --base_model /data/ljx/Qwen2-VL-2B-Instruct \
    --output_dir ./tsne_out \
    --load_base_feats ./tsne_out/feats_base.npz \
    --load_grpo_feats ./tsne_out/feats_grpo.npz
"""

import os
import re
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

# ===== 类别定义 =====
ID_CATEGORIES = [
    "bulk_carrier", "container_ship", "general_cargo_ship",
    "passenger_cargo_ship", "fishing_boat", "sailing_trimaran",
    "chemical_tanker", "crude_oil_tanker", "oil_products_tanker",
    "LNG_tanker", "LPG_tanker", "kayak", "heavy_load_carrier",
    "tugboat"
]
OOD_TRAIN_CATEGORIES = [
    "destroyer", "aircraft_carrier", "frigate", "submarine",
    "firefighting", "reefer", "vehicles_carrier", "passenger_ro-ro_ship",
    "fso", "cruise"
]
OOD_RESERVED_CATEGORIES = [
    "bitumen", "passenger_ship", "catamaran_yacht",
    "monohull_yacht", "monohull_sailboat", "sailing_catamaran"
]

# norm映射（处理大小写/下划线/连字符差异）
def _norm(s):
    return s.replace(' ','').replace('_','').replace('-','').lower()

ID_NORM      = {_norm(c): c for c in ID_CATEGORIES}
OOD_TR_NORM  = {_norm(c): c for c in OOD_TRAIN_CATEGORIES}
OOD_RES_NORM = {_norm(c): c for c in OOD_RESERVED_CATEGORIES}

SHORT = [
    "bulk","container","cargo","pass_cargo",
    "fishing","trimaran","chem_tnk","crude_tnk",
    "oil_tnk","LNG","LPG","kayak","heavy","tug"
]
SHORT_OOD_TR  = ["destroyer","carrier","frigate","submarine",
                 "fire","reefer","vehicles","ro-ro","fso","cruise"]
SHORT_OOD_RES = ["bitumen","pass_ship","cat_yacht",
                 "mono_yacht","mono_sail","sail_cat"]

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tiff'}


# ==========================================
# 1. 扫描图片目录
# ==========================================
def scan_image_dir(image_dir, n_per_class=50, seed=42, select_classes=None):
    """
    扫描 image_dir/class_name/*.jpg 结构
    select_classes: list of str，只保留这些类（None=全部）
    返回 list of dict: {path, category, label_type, short}
    """
    random.seed(seed)
    all_items = []
    unknown_folders = []

    # 构建 select 的 norm 集合
    select_norms = None
    if select_classes:
        select_norms = {_norm(c) for c in select_classes}

    for folder in sorted(os.listdir(image_dir)):
        folder_path = os.path.join(image_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        n = _norm(folder)

        # 过滤：只处理选定类
        if select_norms and n not in select_norms:
            continue

        if n in ID_NORM:
            label_type = 'id'
            canon      = ID_NORM[n]
            ci         = ID_CATEGORIES.index(canon)
            short      = SHORT[ci]
        elif n in OOD_TR_NORM:
            label_type = 'ood_train'
            canon      = OOD_TR_NORM[n]
            ci         = OOD_TRAIN_CATEGORIES.index(canon)
            short      = SHORT_OOD_TR[ci]
        elif n in OOD_RES_NORM:
            label_type = 'ood_reserved'
            canon      = OOD_RES_NORM[n]
            ci         = OOD_RESERVED_CATEGORIES.index(canon)
            short      = SHORT_OOD_RES[ci]
        else:
            unknown_folders.append(folder)
            continue

        imgs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        if not imgs:
            continue

        sampled = random.sample(imgs, min(n_per_class, len(imgs)))
        for p in sampled:
            all_items.append({
                'path':       p,
                'category':   canon,
                'label_type': label_type,
                'short':      short,
            })

    if unknown_folders:
        print(f"  ⚠️  未识别文件夹（跳过）: {unknown_folders}")

    cnt = defaultdict(int)
    for d in all_items:
        cnt[d['label_type']] += 1

    actual_classes = sorted({d['category'] for d in all_items})
    print(f"  扫描完成: {len(actual_classes)} 个类，共 {len(all_items)} 张图片")
    print(f"    ID         : {cnt['id']} 张  ({len([c for c in actual_classes if c in ID_CATEGORIES])} 类)")
    print(f"    OOD-Train  : {cnt['ood_train']} 张  ({len([c for c in actual_classes if c in OOD_TRAIN_CATEGORIES])} 类)")
    print(f"    OOD-Reserv : {cnt['ood_reserved']} 张  ({len([c for c in actual_classes if c in OOD_RESERVED_CATEGORIES])} 类)")
    print(f"    具体类别: {actual_classes}")

    return all_items


# ==========================================
# 2. 特征提取（完全复用 precompute 逻辑）
# ==========================================
@torch.no_grad()
def extract_one(model, processor, image_path, device):
    image = Image.open(image_path).convert('RGB')
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "What ship?"}
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    if HAS_QWEN:
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt"
        ).to(device)
    else:
        inputs = processor(
            text=[text], images=[image],
            padding=True, return_tensors="pt"
        ).to(device)

    out    = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-2]

    if hasattr(inputs, 'pixel_values') and hasattr(inputs, 'image_grid_thw'):
        start, vfeats = 1, []
        for t, h, w in inputs.image_grid_thw:
            n = t * h * w
            vfeats.append(hidden[:, start:start+n])
            start += n
        feat = torch.cat(vfeats, dim=1).mean(dim=1)
    else:
        feat = hidden[:, 0]

    return F.normalize(feat, dim=-1).squeeze(0).cpu().float().numpy()


def extract_features(model, processor, items):
    device = next(model.parameters()).device
    feats  = []
    failed = 0
    for item in tqdm(items, desc="  提取特征"):
        try:
            f = extract_one(model, processor, item['path'], device)
            feats.append(f)
        except Exception as e:
            feats.append(np.zeros(1536))
            failed += 1
    if failed:
        print(f"  ⚠️  失败 {failed} 张（已用零向量填充）")
    return np.array(feats)


def load_model(base_path, ckpt_path=None):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"  加载 base: {base_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True).eval()

    if ckpt_path:
        adapter_cfg = os.path.join(ckpt_path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            print(f"  加载 LoRA: {ckpt_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=False)
        else:
            print(f"  加载 merged: {ckpt_path}")
            del model; torch.cuda.empty_cache()
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                ckpt_path, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True).eval()

    proc_path = ckpt_path if ckpt_path else base_path
    proc = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    if hasattr(proc, "image_processor"):
        proc.image_processor.max_pixels = 401408
        proc.image_processor.min_pixels = 3136
    return model, proc


# ==========================================
# 3. T-SNE
# ==========================================
def run_tsne(feats, perplexity=40):
    from sklearn.manifold import TSNE
    print(f"  T-SNE: {feats.shape} → 2D  perplexity={perplexity}")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                max_iter=1500, init='pca',
                learning_rate='auto', random_state=42, verbose=1)
    return tsne.fit_transform(feats)


# ==========================================
# 4. 白底绘图
# ==========================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 14色 ID palette
ID_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728",
    "#9467bd","#8c564b","#e377c2","#7f7f7f",
    "#bcbd22","#17becf","#393b79","#637939",
    "#8c6d31","#843c39",
]
OOD_TR_COL  = "#E53935"
OOD_RES_COL = "#FF6D00"


def draw_panel(ax, coords, items, title, pt_size=18, alpha=0.70):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8, color='#222')
    ax.tick_params(labelsize=8, colors='#666')
    ax.grid(True, color='#F0F0F0', lw=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC'); sp.set_linewidth(0.8)

    # ---- ID（只画实际存在的类）----
    present_id = sorted({d['category'] for d in items if d['label_type']=='id'})
    for cat in present_id:
        ci  = ID_CATEGORIES.index(cat) if cat in ID_CATEGORIES else 0
        col = ID_PALETTE[ci % len(ID_PALETTE)]
        mask = np.array([d['label_type']=='id' and d['category']==cat for d in items])
        pts  = coords[mask]
        ax.scatter(pts[:,0], pts[:,1], c=col, s=pt_size,
                   alpha=alpha, edgecolors='none', zorder=3)
        cx, cy = np.median(pts[:,0]), np.median(pts[:,1])
        ax.text(cx, cy, SHORT[ci], fontsize=7.5, color=col,
                ha='center', va='center', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=col, linewidth=0.8, alpha=0.88))

    # ---- OOD-Train ▲ ----
    mask_ot = np.array([d['label_type']=='ood_train' for d in items])
    if mask_ot.sum() > 0:
        ax.scatter(coords[mask_ot,0], coords[mask_ot,1],
                   c=OOD_TR_COL, s=pt_size+6, marker='^',
                   alpha=0.82, edgecolors='none', zorder=4)
        for cat in sorted({d['category'] for d in items if d['label_type']=='ood_train'}):
            ci = OOD_TRAIN_CATEGORIES.index(cat) if cat in OOD_TRAIN_CATEGORIES else 0
            m  = np.array([d['label_type']=='ood_train' and d['category']==cat for d in items])
            if m.sum() < 3: continue
            cx, cy = np.median(coords[m,0]), np.median(coords[m,1])
            ax.text(cx, cy, SHORT_OOD_TR[ci], fontsize=7, color=OOD_TR_COL,
                    ha='center', va='center', fontweight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              edgecolor=OOD_TR_COL, linewidth=0.7, alpha=0.85))

    # ---- OOD-Reserved ◆ ----
    mask_or = np.array([d['label_type']=='ood_reserved' for d in items])
    if mask_or.sum() > 0:
        ax.scatter(coords[mask_or,0], coords[mask_or,1],
                   c='none', s=pt_size+10, marker='D',
                   edgecolors=OOD_RES_COL, lw=1.2, alpha=0.88, zorder=4)
        for cat in sorted({d['category'] for d in items if d['label_type']=='ood_reserved'}):
            ci = OOD_RESERVED_CATEGORIES.index(cat) if cat in OOD_RESERVED_CATEGORIES else 0
            m  = np.array([d['label_type']=='ood_reserved' and d['category']==cat for d in items])
            if m.sum() < 3: continue
            cx, cy = np.median(coords[m,0]), np.median(coords[m,1])
            ax.text(cx, cy, SHORT_OOD_RES[ci], fontsize=7, color=OOD_RES_COL,
                    ha='center', va='center', fontweight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              edgecolor=OOD_RES_COL, linewidth=0.7, alpha=0.85))

    # 左下计数
    n_id = sum(d['label_type'] == 'id' for d in items)
    ax.text(
        0.01,
        0.01,
        f"ID: {n_id}  OOD-Train: {mask_ot.sum()}  OOD-Res: {mask_or.sum()}",
        transform=ax.transAxes,
        fontsize=7.5,
        color='#888',
        va='bottom'
    )


def build_legend(items):
    """根据实际出现的类动态生成图例"""
    h = []
    for cat in sorted({d['category'] for d in items if d['label_type']=='id'}):
        ci  = ID_CATEGORIES.index(cat) if cat in ID_CATEGORIES else 0
        col = ID_PALETTE[ci % len(ID_PALETTE)]
        h.append(mpatches.Patch(facecolor=col, edgecolor='none',
                                label=SHORT[ci], alpha=0.85))
    n_ot = sum(d['label_type']=='ood_train'    for d in items)
    n_or = sum(d['label_type']=='ood_reserved' for d in items)
    if n_ot > 0:
        h.append(Line2D([0],[0], marker='^', color='w',
                        markerfacecolor=OOD_TR_COL, markeredgecolor='none',
                        markersize=8, label=f"OOD-Train ({len({d['category'] for d in items if d['label_type']=='ood_train'})} cls)"))
    if n_or > 0:
        h.append(Line2D([0],[0], marker='D', color='w',
                        markerfacecolor='none', markeredgecolor=OOD_RES_COL,
                        markeredgewidth=1.5, markersize=8,
                        label=f"OOD-Reserved ({len({d['category'] for d in items if d['label_type']=='ood_reserved'})} cls)"))
    return h


def plot_compare(coords_base, coords_grpo, items, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    draw_panel(ax1, coords_base, items, "Base Model (Before Training)")
    draw_panel(ax2, coords_grpo, items, "After Distance-Guided GRPO")
    fig.suptitle(
        "T-SNE Feature Distribution: Base Model  vs.  Distance-Guided GRPO",
        fontsize=13, fontweight='bold', y=0.99, color='#222')
    n_cols = min(len(build_legend(items)), 10)
    fig.legend(handles=build_legend(items), loc='lower center', ncol=n_cols,
               fontsize=8, frameon=True, edgecolor='#CCC',
               facecolor='white', bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(out_dir, "tsne_compare.png")
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out.replace('.png','.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ 对比图: {out}")


def plot_single(coords, items, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('white')
    draw_panel(ax, coords, items, title)
    n_cols = min(len(build_legend(items)), 10)
    fig.legend(handles=build_legend(items), loc='lower center', ncol=n_cols,
               fontsize=7.5, frameon=True, edgecolor='#CCC',
               facecolor='white', bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_path.replace('.png','.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ 单图: {out_path}")


# ==========================================
# 5. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",       required=True,
                        help="图片根目录，如 /data/ljx/dataset/boat-29/images")
    parser.add_argument("--base_model",      default=None)
    parser.add_argument("--grpo_model",      default=None,
                        help="GRPO checkpoint（LoRA 或 merged）")
    parser.add_argument("--output_dir",      default="./tsne_out")
    parser.add_argument("--n_per_class",     type=int, default=50,
                        help="每类最多取多少张图（默认50）")
    parser.add_argument("--select_classes",  default=None,
                        help="只可视化这些类，逗号分隔，如:\n"
                             "  bulk_carrier,container_ship,kayak,LNG_tanker,tugboat,"
                             "fishing_boat,destroyer,aircraft_carrier,submarine,cruise,"
                             "passenger_ship,catamaran_yacht,monohull_sailboat\n"
                             "不传则使用全部类")
    parser.add_argument("--perplexity",      type=float, default=40)
    parser.add_argument("--save_feats",      action="store_true")
    parser.add_argument("--load_base_feats", default=None,
                        help="Base 特征 .npz，跳过提取")
    parser.add_argument("--load_grpo_feats", default=None,
                        help="GRPO 特征 .npz，跳过提取")
    parser.add_argument("--mode",            default="compare",
                        choices=["compare","single_base","single_grpo"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 解析 select_classes
    select_classes = None
    if args.select_classes:
        select_classes = [c.strip() for c in args.select_classes.split(',') if c.strip()]
        print(f"\n  只提取 {len(select_classes)} 个类: {select_classes}")

    # 扫描图片
    print(f"\n[1] 扫描图片目录: {args.image_dir}")
    items = scan_image_dir(args.image_dir,
                           n_per_class=args.n_per_class,
                           select_classes=select_classes)

    # 特征获取
    def get_feats(npz_path, ckpt_path, role):
        if npz_path and os.path.exists(npz_path):
            print(f"\n  读取 {role} 特征缓存: {npz_path}")
            return np.load(npz_path, allow_pickle=True)['feats']
        assert args.base_model, "需要 --base_model"
        print(f"\n  加载模型提取 {role} 特征...")
        model, proc = load_model(args.base_model, ckpt_path)
        feats = extract_features(model, proc, items)
        del model; torch.cuda.empty_cache()
        if args.save_feats:
            p = os.path.join(args.output_dir, f"feats_{role}.npz")
            np.savez(p, feats=feats)
            print(f"  特征已保存: {p}")
        return feats

    if args.mode == "compare":
        feats_base = get_feats(args.load_base_feats, None,            "base")
        feats_grpo = get_feats(args.load_grpo_feats, args.grpo_model, "grpo")
        print("\n[3] 联合 T-SNE（坐标系一致）...")
        coords_all = run_tsne(np.concatenate([feats_base, feats_grpo]), args.perplexity)
        n = len(feats_base)
        plot_compare(coords_all[:n], coords_all[n:], items, args.output_dir)

    elif args.mode == "single_base":
        feats  = get_feats(args.load_base_feats, None, "base")
        coords = run_tsne(feats, args.perplexity)
        plot_single(coords, items, "Base Model Feature Space (T-SNE)",
                    os.path.join(args.output_dir, "tsne_base.png"))

    elif args.mode == "single_grpo":
        feats  = get_feats(args.load_grpo_feats, args.grpo_model, "grpo")
        coords = run_tsne(feats, args.perplexity)
        plot_single(coords, items, "Distance-Guided GRPO Feature Space (T-SNE)",
                    os.path.join(args.output_dir, "tsne_grpo.png"))


if __name__ == "__main__":
    main()