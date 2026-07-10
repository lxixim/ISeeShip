#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge SFT LoRA adapter into base model
用于GRPO训练前的准备
"""

import os
import shutil
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def cp_files(file_names, src_dir, dst_dir):
    """
    Copy files from src_dir to dst_dir.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    copied = []
    missing = []
    
    for file_name in file_names:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            copied.append(file_name)
            print(f"  ✓ Copied {file_name}")
        else:
            missing.append(file_name)
            print(f"  ⚠️  {file_name} not found (may not be needed)")
    
    return copied, missing


def main():
    # ==================== 配置路径 ====================
    BASE_MODEL_PATH = "/data/ljx/Qwen2-VL-2B-Instruct"
    SFT_ADAPTER_PATH = "/data/ljx/visualRft/share_models/classification/Qwen2-VL-2B-Instruct_SFT_ship_cls_10shot/checkpoint-200"
    SAVE_PATH = "/data/ljx/visualRft/share_models/classification/Qwen2-VL-2B-Instruct_SFT_ship_cls_10shot-merge"
    
    print("=" * 70)
    print("🔧 Merge SFT LoRA Adapter into Base Model")
    print("=" * 70)
    print(f"Base Model:   {BASE_MODEL_PATH}")
    print(f"SFT Adapter:  {SFT_ADAPTER_PATH}")
    print(f"Output Path:  {SAVE_PATH}")
    print("=" * 70 + "\n")
    
    # ==================== 验证路径 ====================
    print("1️⃣ Validating paths...")
    
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"❌ Base model not found: {BASE_MODEL_PATH}")
    print(f"  ✓ Base model exists")
    
    if not os.path.exists(SFT_ADAPTER_PATH):
        raise FileNotFoundError(f"❌ SFT adapter not found: {SFT_ADAPTER_PATH}")
    print(f"  ✓ SFT adapter exists")
    
    # 检查adapter文件
    adapter_file = os.path.join(SFT_ADAPTER_PATH, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"❌ Adapter file not found: {adapter_file}")
    print(f"  ✓ Adapter file exists: {os.path.getsize(adapter_file) / 1024 / 1024:.2f} MB\n")
    
    # ==================== 加载Base Model ====================
    print("2️⃣ Loading base model...")
    print("  (This may take a few minutes...)")
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",  # 自动分配到GPU
        trust_remote_code=True,
        torch_dtype="auto",
    )
    print("  ✓ Base model loaded\n")
    
    # ==================== 加载SFT Adapter ====================
    print("3️⃣ Loading SFT LoRA adapter...")
    
    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_PATH,
        is_trainable=False  # 推理模式
    )
    print("  ✓ SFT adapter loaded")
    print(f"  ✓ Adapter type: {model.peft_config}\n")
    
    # ==================== Merge ====================
    print("4️⃣ Merging adapter into base model...")
    print("  (This will take a few minutes...)")
    
    model = model.merge_and_unload()
    print("  ✓ Merge completed\n")
    
    # ==================== 保存Merged Model ====================
    print("5️⃣ Saving merged model...")
    
    if os.path.exists(SAVE_PATH):
        print(f"  ⚠️  Output directory exists, will overwrite: {SAVE_PATH}")
    
    model.save_pretrained(SAVE_PATH)
    print(f"  ✓ Model saved to {SAVE_PATH}\n")
    
    # ==================== 复制配置文件 ====================
    print("6️⃣ Copying configuration files...")
    print("  (优先使用 adapter 目录中的文件，因为训练时可能已修改)")
    
    # 模型推理必需的文件列表
    # 注意：优先从 adapter 复制（如果存在），因为训练时可能修改了这些文件
    # 如果 adapter 中没有，再从 base model 复制
    essential_files = [
        "preprocessor_config.json",
        "config.json", 
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "generation_config.json",
        "added_tokens.json",
        "merges.txt",  # tokenizer BPE 合并规则
        "chat_template.jinja",  # 聊天模板
        "video_preprocessor_config.json",  # 视频预处理器配置
    ]
    
    copied_from_adapter = []
    copied_from_base = []
    missing_files = []
    
    for file_name in essential_files:
        adapter_file = os.path.join(SFT_ADAPTER_PATH, file_name)
        base_file = os.path.join(BASE_MODEL_PATH, file_name)
        dst_file = os.path.join(SAVE_PATH, file_name)
        
        # 优先从 adapter 复制（如果存在）
        if os.path.exists(adapter_file):
            shutil.copy(adapter_file, dst_file)
            copied_from_adapter.append(file_name)
            print(f"  ✓ Copied {file_name} from adapter")
        # 如果 adapter 中没有，从 base model 复制
        elif os.path.exists(base_file):
            shutil.copy(base_file, dst_file)
            copied_from_base.append(file_name)
            print(f"  ✓ Copied {file_name} from base model")
        else:
            missing_files.append(file_name)
            print(f"  ⚠️  {file_name} not found (may not be needed)")
    
    print(f"\n  📊 Summary:")
    print(f"     - Copied {len(copied_from_adapter)} files from adapter (priority)")
    print(f"     - Copied {len(copied_from_base)} files from base model (fallback)")
    if missing_files:
        print(f"     - {len(missing_files)} files not found (may be optional)")
    
    print("\n  📝 Note: 以下文件不需要复制（训练相关，merge 后不再需要）：")
    print("     - adapter_config.json (LoRA 配置，已 merge 到模型)")
    print("     - adapter_model.safetensors (adapter 权重，已 merge 到模型)")
    print("     - training_args.bin (训练参数)")
    print("     - trainer_state.json (训练状态)")
    print("     - all_results.json, train_results.json (训练结果)")
    print("     - README.md (文档，可选)")
    print("     - checkpoint-*/ (检查点目录)")
    print()
    
    # ==================== 验证Merged Model ====================
    print("7️⃣ Verifying merged model...")
    
    try:
        test_model = Qwen2VLForConditionalGeneration.from_pretrained(
            SAVE_PATH,
            device_map="cpu",  # 验证时用CPU节省显存
            trust_remote_code=True,
            torch_dtype="auto",
        )
        print("  ✓ Merged model can be loaded successfully")
        
        # 检查模型参数数量
        total_params = sum(p.numel() for p in test_model.parameters())
        print(f"  ✓ Total parameters: {total_params:,}")
        
        del test_model  # 释放内存
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return
    
    print()
    
    # ==================== 完成 ====================
    print("=" * 70)
    print("✅ Merge completed successfully!")
    print("=" * 70)
    print(f"Merged model saved to:")
    print(f"  {SAVE_PATH}")
    print()
    print("📋 Next steps:")
    print("  1. Use this merged model for GRPO training:")
    print(f"     export SFT_MERGED={SAVE_PATH}")
    print(f"     --model_name_or_path $SFT_MERGED")
    print()
    print("  2. This ensures GRPO builds on top of SFT")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()