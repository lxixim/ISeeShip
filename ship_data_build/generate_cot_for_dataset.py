import re
import base64
import argparse
import time
import os
from io import BytesIO
from typing import Optional

from datasets import load_from_disk, Dataset, DatasetDict
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ============ 1. 类别定义（严格对应 IAD-R1 实验设置） ============
ID_CLASSES = [
    "LNG_tanker", "LPG_tanker", "bulk_carrier", "chemical_tanker",
    "container_ship", "crude_oil_tanker", "fishing_boat", "general_cargo_ship",
    "heavy_load_carrier", "kayak", "oil_products_tanker", "passenger_cargo_ship",
    "sailing_trimaran", "tugboat"
]

# ============ 2. 配置与工具函数 ============
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_BASE_URL = "https://xiaoai.plus/v1"
DEFAULT_API_KEY = os.getenv("XIAOAI_API_KEY")

def encode_pil_image(pil_image: Image.Image, format: str = "JPEG") -> str:
    """将 PIL Image 编码为 base64"""
    buffer = BytesIO()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_answer_from_solution(solution: str) -> Optional[str]:
    """从原始 solution 标签中提取答案"""
    match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# ============ 3. 高精度专家流 Prompt 生成器（只生成 think） ============
def generate_classification_prompt(ship_class: str) -> str:
    """分类任务：只生成推理过程"""
    return f"""Analyze this maritime vessel. 
Ground Truth: {ship_class}

Guidelines:
- Write ONE short paragraph (50-80 words).
- Identify the core hull structure and deck equipment.
- Explicitly mention why it is {ship_class} and NOT a similar vessel type.
- Do not use bullets or numbers.

**IMPORTANT**: Output ONLY the <think> tag with your reasoning. Do NOT include <answer> tag.

Format:
<think>[Your concise expert logic]</think>"""

def generate_detection_prompt(ship_class: str, original_answer: str) -> str:
    """检测任务：只生成空间推理过程"""
    return f"""Task: Expert Ship Detection & Localization.
**Target Category**: {ship_class}
**Ground Truth Localization**: {original_answer}

**Expert Analysis Guidelines**:
Write ONE professional paragraph (50-80 words) justifying the bounding box position.
1. **Spatial Anchoring**: Describe the vessel's position relative to the frame.
2. **Visual Evidence**: Describe the hull and superstructure features at these coordinates.
3. **Extent Justification**: Explain why the box covers the ship from bow to stern.
4. **Constraint**: No numbering, no line breaks, single paragraph only.

**IMPORTANT**: Output ONLY the <think> tag with your reasoning. Do NOT include <answer> tag.

Format:
<think>[Your spatial and visual analysis paragraph]</think>"""

def generate_openset_prompt(ship_class: str) -> str:
    """开集识别：只生成推理过程"""
    is_known = ship_class in ID_CLASSES
    reasoning_hint = "is a known class" if is_known else "is NOT in the known classes"
    
    return f"""Perform open-set ship recognition. 
**Known ID Classes**: {", ".join(ID_CLASSES)}
**Ground Truth**: The vessel is a {ship_class}, which {reasoning_hint}.

**Expert Analysis Guidelines**:
Write ONE coherent paragraph (50-80 words) following a "bottom-up" reasoning flow.
1. **Identification**: Identify the ship's actual type based on visual clues.
2. **Matching**: Compare its features against the 14 known ID classes. 
3. **Rejection (if unknown)**: If not a known class, explain which target features are missing.
4. **Constraint**: No numbering, no line breaks, single paragraph only.

**IMPORTANT**: Output ONLY the <think> tag with your reasoning. Do NOT include <answer> tag.

Format:
<think>[Your reasoning proving the known/unknown identity]</think>"""

def get_system_instruction() -> str:
    """系统级约束：强调精炼专家流"""
    return """You are a senior maritime expert. 
Follow the COMPACT Expert Flow protocol:
1. BREVITY IS KEY: Keep the reasoning process within 50-80 words.
2. ESSENTIAL EVIDENCE ONLY: Describe ONLY the 2-3 most critical visual features that define this vessel.
3. LOGICAL BRIDGE: Briefly explain WHY these features lead to the answer.
4. NO FILLERS: Remove introductory phrases like "The image shows...". Start directly with the evidence.
5. OUTPUT ONLY <think> TAG: Do not generate <answer> tag."""

# ============ 4. 核心 API 生成函数 ============

def generate_cot_solution(
    client: OpenAI,
    pil_image: Image.Image,
    ship_class: str,
    task_type: str,
    original_solution: str,
    model: str = DEFAULT_MODEL
) -> Optional[str]:
    """生成 think 标签，然后拼接原始 answer"""
    
    # 1. 提取原始 answer
    original_answer = extract_answer_from_solution(original_solution)
    if not original_answer:
        print(f" ⚠️ 警告：无法从原始 solution 中提取 answer")
        return None
    
    # 2. 编码图片
    try:
        img_b64 = encode_pil_image(pil_image)
    except Exception as e:
        print(f" ❌ 图片编码失败: {e}")
        return None

    # 3. 根据任务类型生成 prompt（只要求生成 think）
    if task_type == "classification":
        prompt_text = generate_classification_prompt(ship_class)
    elif task_type == "detection":
        prompt_text = generate_detection_prompt(ship_class, original_answer)
    elif task_type == "openset":
        prompt_text = generate_openset_prompt(ship_class)
    else:
        return None

    # 4. 调用 API，最多重试 3 次
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_system_instruction()},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            think_content = response.choices[0].message.content.strip()
            
            # 5. 验证返回内容包含 think 标签
            if "<think>" not in think_content:
                print(f" ⚠️ 警告：API 未返回 <think> 标签，重试...")
                time.sleep(2 ** attempt)
                continue
            
            # 6. 提取 think 内容（去除可能的多余内容）
            think_match = re.search(r'<think>(.*?)</think>', think_content, re.DOTALL)
            if not think_match:
                print(f" ⚠️ 警告：无法解析 <think> 标签，重试...")
                time.sleep(2 ** attempt)
                continue
                
            think_text = think_match.group(0)  # 保留完整的 <think>...</think>
            
            # 7. ✅ 手动拼接：API生成的think + 原始answer
            final_solution = f"<solution>\n{think_text}\n<answer>{original_answer}</answer>\n</solution>"
            
            return final_solution
            
        except Exception as e:
            print(f" ❌ API 调用失败 (尝试 {attempt + 1}/3): {e}")
            time.sleep(2 ** attempt)
    
    return None

# ============ 5. 主处理循环 ============

def process_dataset(
    input_path: str,
    output_path: str,
    api_key: str,
    task_type: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    batch_size: int = 10,
    delay: float = 1.0,
    max_samples: int = None
) -> None:
    print(f"📂 加载数据集: {input_path} | 任务: {task_type}")
    ds = load_from_disk(input_path)
    train_ds = ds['train'] if isinstance(ds, DatasetDict) else ds
    
    total_samples = len(train_ds) if max_samples is None else min(max_samples, len(train_ds))
    print(f"📊 数据集大小: {len(train_ds)} 样本 (处理 {total_samples} 个)")
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    new_data = {'image': [], 'problem': [], 'solution': []}
    success_count = 0
    
    for i in tqdm(range(total_samples), desc=f"生成 {task_type} CoT"):
        sample = train_ds[i]
        
        # 自动提取原始类别
        raw_sol_content = extract_answer_from_solution(sample['solution'])
        
        # 兼容性逻辑：如果是检测任务，类别通常在 problem 文本中
        if task_type == "detection":
            match = re.search(r"category '(.*?)'", sample['problem'])
            ship_class = match.group(1) if match else "unknown"
        else:
            ship_class = raw_sol_content if raw_sol_content and "[" not in raw_sol_content else "unknown"

        # 调用 API 生成新 solution（think + 原始answer）
        res = generate_cot_solution(
            client, 
            sample['image'], 
            ship_class, 
            task_type, 
            sample['solution'],  # 传入原始 solution
            model
        )
        
        if res:
            new_data['image'].append(sample['image'])
            new_data['problem'].append(sample['problem'])
            new_data['solution'].append(res)
            success_count += 1
            print(f"  ✅ 样本 {i+1}/{total_samples}: {ship_class}")
        else:
            print(f"  ❌ 样本 {i+1}/{total_samples}: 生成失败，跳过")
        
        # 批次延迟
        if (i + 1) % batch_size == 0:
            time.sleep(delay)

    print(f"\n📈 成功生成: {success_count}/{total_samples} 样本")
    
    # 保存数据集（使用 DatasetDict 格式，包含 'train' 分割）
    if success_count > 0:
        new_ds = Dataset.from_dict(new_data)
        ds_dict = DatasetDict({'train': new_ds})
        ds_dict.save_to_disk(output_path)
        print(f"✅ 处理完成！数据集保存至: {output_path}")
    else:
        print(f"❌ 没有成功生成任何样本，不保存数据集")

# ============ 6. CLI 入口 ============

def main():
    parser = argparse.ArgumentParser(description="船舶多任务高精度 CoT 生成脚本（只插入think标签）")
    parser.add_argument("--input_dataset", type=str, required=True, help="输入数据集路径")
    parser.add_argument("--output_dataset", type=str, required=True, help="输出数据集路径")
    parser.add_argument("--task_type", choices=["classification", "detection", "openset"], required=True, help="任务类型")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="API密钥")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--max_samples", type=int, default=None, help="最大处理样本数")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="API基础URL")
    parser.add_argument("--batch_size", type=int, default=10, help="批次大小")
    parser.add_argument("--delay", type=float, default=1.0, help="批次间延迟(秒)")
    
    args = parser.parse_args()
    
    process_dataset(
        input_path=args.input_dataset,
        output_path=args.output_dataset,
        api_key=args.api_key,
        task_type=args.task_type,
        base_url=args.base_url,
        model=args.model,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        delay=args.delay
    )

if __name__ == "__main__":
    main()
