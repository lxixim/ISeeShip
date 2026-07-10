#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型推理脚本 - 支持 Qwen2-VL API, LLaVA, InternVL, gemini, GPT-4 (并发版本)
已将Qwen改为百炼平台API调用
"""
import sys
import os

# 🔥 步骤3：正常的imports
import io
import re
import json
import asyncio
import base64
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

import logging
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings

# 屏蔽 Python 的标准警告
warnings.filterwarnings("ignore")

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_model_inference.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==================== 模型配置 ====================
# 可选: "qwen2_7b", "qwen2.5_7b", "qwen2.5_3b", "gemini", "gpt4"
MODEL_TYPE = "gemini"

# API配置
API_CONFIGS = {
     "qwen2_7b_hf": {
        "base_url": "https://router.huggingface.co/v1",
        "api_key": os.getenv("HF_TOKEN"),
        "model": "Qwen/Qwen2-VL-7B-Instruct:hyperbolic"
    },
    "qwen2.5_7b_hf": {
        "base_url": "https://router.huggingface.co/v1", 
        "api_key": os.getenv("HF_TOKEN"),
        "model": "Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic"
    },
    # ===== 百炼平台 Qwen 系列 =====
    "qwen2_7b": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "model": "qwen2-vl-7b-instruct"
    },
    "qwen2.5_7b": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "model": "qwen2.5-vl-7b-instruct"
    },
    "qwen2.5_3b": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "model": "qwen2.5-vl-3b-instruct"
    },
    # ===== 其他商业模型 =====
    "gemini": {
        "base_url": "https://xiaoai.plus/v1",
        "api_key": os.getenv("XIAOAI_API_KEY"),
        "model": "claude-haiku-4-5-20251001-thinking"
    },
    "gpt4": {
        "base_url": "https://xiaoai.plus/v1",
        "api_key": os.getenv("XIAOAI_API_KEY"),
        "model": "gpt-4.1-nano"
    }
}

# 获取当前模型配置
MODEL_CONFIG = API_CONFIGS.get(MODEL_TYPE)
MODEL_NAME = MODEL_CONFIG["model"] if MODEL_CONFIG else MODEL_TYPE

# ==================== 优化配置 ====================
MAX_NEW_TOKENS = 512

# API并发配置
API_CONCURRENCY = 5   # 并发请求数，百炼平台建议不要太高
API_BATCH_SIZE = 10   # API每批处理数量
API_RETRY_DELAY = 1   # 重试延迟基数（秒）

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_text(text):
    """文本标准化"""
    return text.replace(' ', '').replace('_', '').lower()


# ==================== 异步API调用 ====================
async def call_qwen_api_async(client, item, semaphore, model_type):
    """异步调用百炼平台Qwen VL API"""
    async with semaphore:
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(API_RETRY_DELAY * (2 ** attempt))
                
                response = await client.chat.completions.create(
                    model=API_CONFIGS[model_type]["model"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{item['image_base64']}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": item["question"]
                                }
                            ],
                        }
                    ],
                    max_tokens=MAX_NEW_TOKENS,
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower() or "Throttling" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = API_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"API 限流，等待 {wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API调用失败（已重试{max_retries}次）: {e}")
                        return ""
                else:
                    logger.error(f"API调用失败: {e}")
                    return ""
        return ""


async def call_gpt4_async(client, item, semaphore):
    """异步调用GPT-4"""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=API_CONFIGS["gpt4"]["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": item["question"]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{item['image_base64']}"
                                }
                            }
                        ],
                    }
                ],
                max_tokens=MAX_NEW_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 API调用失败: {e}")
            return ""


async def call_gemini_async(client, item, semaphore):
    """异步调用gemini"""
    async with semaphore:
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(API_RETRY_DELAY * (2 ** attempt))
                
                response = await client.chat.completions.create(
                    model=API_CONFIGS["gemini"]["model"],
                    max_tokens=MAX_NEW_TOKENS,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": item["question"]
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{item['image_base64']}"
                                    }
                                }
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = API_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"gemini API 429错误，等待 {wait_time:.1f}秒后重试")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"gemini API调用失败（已重试{max_retries}次）: {e}")
                        return ""
                else:
                    logger.error(f"gemini API调用失败: {e}")
                    return ""
        return ""


async def batch_call_api_async(items, model_type):
    """批量异步调用API"""
    from openai import AsyncOpenAI
    
    semaphore = asyncio.Semaphore(API_CONCURRENCY)
    config = API_CONFIGS[model_type]
    
    client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    # 根据模型类型选择调用函数
    if model_type in ["qwen2_7b", "qwen2.5_7b", "qwen2.5_3b", "qwen_vl_max", "qwen2_7b_hf", "qwen2.5_7b_hf"]:
        tasks = [call_qwen_api_async(client, item, semaphore, model_type) for item in items]
    elif model_type == "gpt4":
        tasks = [call_gpt4_async(client, item, semaphore) for item in items]
    elif model_type == "gemini":
        tasks = [call_gemini_async(client, item, semaphore) for item in items]
    else:
        raise ValueError(f"Unsupported API model type: {model_type}")
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常
    processed_responses = []
    for r in responses:
        if isinstance(r, Exception):
            logger.error(f"API调用异常: {r}")
            processed_responses.append("")
        else:
            processed_responses.append(r)
    
    return processed_responses


# ==================== 数据处理器 ====================
class DataProcessor:
    """统一的数据处理接口"""
    
    @staticmethod
    def prepare_api_batch(batch_data, categories, question):
        """准备API调用批次数据"""
        batch_items = []
        batch_labels = []
        
        for image_data in batch_data:
            for image_path, image_label in image_data.items():
                if not os.path.exists(image_path):
                    logger.warning(f"文件不存在，跳过: {image_path}")
                    continue
                
                try:
                    img = Image.open(image_path)
                    img = img.convert('RGB')
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img.close()
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    batch_items.append({
                        "image_base64": img_base64,
                        "question": question
                    })
                    batch_labels.append((image_path, image_label, categories[image_label]))
                except Exception as e:
                    logger.error(f"加载图片失败: {image_path}, {e}")
                    continue
        
        if not batch_items:
            return None, None
        
        return batch_items, batch_labels


# ==================== 批量推理 ====================
def batch_process_images(batch_data, categories, model_type):
    """批量处理图片"""
    
    question = (
        "This is an image containing a ship. Please identify the model of the ship based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>species name</answer>\n"
        "Please strictly follow the format."
    )
    
    # 准备数据
    inputs, valid_labels = DataProcessor.prepare_api_batch(batch_data, categories, question)
    
    if inputs is None or valid_labels is None:
        return []
    
    # API调用
    responses = asyncio.run(batch_call_api_async(inputs, model_type))
    
    # 处理结果
    results = []
    for (image_path, image_label, image_cate), response in zip(valid_labels, responses):
        result_item = {
            "image_path": image_path,
            "true_label": image_cate,
            "true_label_id": image_label,
            "model_response": response,
            "extracted_answer": None,
            "is_correct": False,
            "error": None
        }
        
        try:
            match = ANSWER_PATTERN.search(response)
            if match:
                answer_content = match.group(1).strip()
                result_item["extracted_answer"] = answer_content
                
                image_cate_norm = normalize_text(image_cate)
                answer_content_norm = normalize_text(answer_content)
                
                is_correct = image_cate_norm in answer_content_norm or answer_content_norm in image_cate_norm
                result_item["is_correct"] = is_correct
                results.append((is_correct, False, result_item))
            else:
                result_item["error"] = "Failed to extract answer from response"
                results.append((False, True, result_item))
        except Exception as e:
            error_msg = f"Error processing response: {e}"
            logger.error(f"{error_msg}, response: {response[:200] if response else 'N/A'}")
            result_item["error"] = error_msg
            results.append((False, True, result_item))
    
    return results


def run_api_inference(val_set, categories):
    """API模型的推理函数"""
    proc_logger = logging.getLogger(__name__)
    proc_logger.info(f"开始 {MODEL_TYPE.upper()} API 推理...")
    proc_logger.info(f"模型: {MODEL_CONFIG['model']}")
    proc_logger.info(f"总样本数: {len(val_set)}")
    proc_logger.info(f"并发数: {API_CONCURRENCY}, 批大小: {API_BATCH_SIZE}")
    
    error_count = 0
    right_count = 0
    total_processed = 0
    detailed_results = []
    
    batches = [val_set[i:i + API_BATCH_SIZE] for i in range(0, len(val_set), API_BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"{MODEL_TYPE} API")):
        results = batch_process_images(batch, categories, MODEL_TYPE)
        
        for is_correct, is_error, result_item in results:
            total_processed += 1
            if is_correct:
                right_count += 1
            if is_error:
                error_count += 1
            detailed_results.append(result_item)
        
        if (batch_idx + 1) % 5 == 0:
            current_acc = right_count / total_processed if total_processed > 0 else 0
            proc_logger.info(f'已处理: {total_processed}, 正确: {right_count}, 当前准确率: {current_acc:.2%}')
    
    proc_logger.info(f'最终正确数: {right_count}, 错误数: {error_count}')
    return [error_count, right_count, len(val_set), detailed_results]


def main():
    import torch
    
    logger.info('=' * 60)
    logger.info(f'模型类型: {MODEL_TYPE}')
    logger.info(f'API Base URL: {MODEL_CONFIG["base_url"]}')
    logger.info(f'Model: {MODEL_CONFIG["model"]}')
    logger.info(f'并发数: {API_CONCURRENCY}')
    logger.info(f'批大小: {API_BATCH_SIZE}')
    
    # 读取数据
    with open('./val_data/ships.txt', 'r') as file:
        categories = [line.strip() for line in file.readlines()]
    
    pth_file_path = './val_data/ship29.pth'
    predictions = torch.load(pth_file_path, weights_only=False)
    val_set = [{k: int(v['label'])} for item in predictions for k, v in item.items()]
    
    # 运行推理
    result = run_api_inference(val_set, categories)
    
    # 汇总结果
    global_count_error = int(result[0])
    global_count_right = result[1]
    global_results = int(result[2])
    all_detailed_results = result[3] if len(result) > 3 else []
    
    accuracy = global_count_right / global_results if global_results > 0 else 0
    logger.info('=' * 60)
    logger.info(f'模型: {MODEL_TYPE} ({MODEL_CONFIG["model"]})')
    logger.info('总样本数: %d', global_results)
    logger.info('正确数: %d', global_count_right)
    logger.info('错误数: %d', global_count_error)
    logger.info('准确率: %.2f%%', accuracy * 100)
    logger.info('=' * 60)
    
    # 保存结果
    output_file = f'results_{MODEL_TYPE}_{MODEL_NAME}.txt'
    with open(output_file, 'w') as f:
        f.write(f'Model Type: {MODEL_TYPE}\n')
        f.write(f'API Model: {MODEL_CONFIG["model"]}\n')
        f.write(f'Base URL: {MODEL_CONFIG["base_url"]}\n')
        f.write(f'Concurrency: {API_CONCURRENCY}\n')
        f.write(f'Batch Size: {API_BATCH_SIZE}\n')
        f.write(f'Error number: {global_count_error}\n')
        f.write(f'Total Right Number: {global_count_right}\n')
        f.write(f'Total Number: {global_results}\n')
        f.write(f'Accuracy: {accuracy * 100:.2f}%\n')
    
    detailed_output_file = f'{MODEL_TYPE}_{MODEL_NAME}_detailed.json'
    final_data = {
        "summary": {
            'model_type': MODEL_TYPE,
            'model_name': MODEL_NAME,
            'api_model': MODEL_CONFIG["model"],
            'base_url': MODEL_CONFIG["base_url"],
            'batch_size': API_BATCH_SIZE,
            'concurrency': API_CONCURRENCY,
            'total_samples': global_results,
            'correct_count': global_count_right,
            'error_count': global_count_error,
            'accuracy': accuracy
        },
        "detailed_results": all_detailed_results
    }
    
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f'统计结果已保存到: {output_file}')
    logger.info(f'详细推理过程已保存到: {detailed_output_file}')


if __name__ == "__main__":
    main()
