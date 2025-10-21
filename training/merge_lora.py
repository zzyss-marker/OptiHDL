"""
LoRA 模型合并脚本
将 LoRA 适配器合并到基础模型，生成可独立推理的完整模型
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ========== 硬编码配置 ==========
BASE_MODEL_PATH = "models/qwen"
LORA_ADAPTER_PATH = "models/qwen_finetuned"
OUTPUT_PATH = "models/qwen_finetuned_merged"


def merge_lora_to_base(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device: str = "auto"
):
    """
    合并 LoRA 适配器到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA 适配器路径
        output_path: 输出路径
        device: 设备 (auto/cpu/cuda)
    """
    logger.info("=" * 60)
    logger.info("LoRA 模型合并工具")
    logger.info("=" * 60)
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"LoRA 适配器: {lora_adapter_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info("=" * 60)
    
    # 检查路径
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"基础模型不存在: {base_model_path}")
    if not Path(lora_adapter_path).exists():
        raise FileNotFoundError(f"LoRA 适配器不存在: {lora_adapter_path}")
    
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 1. 加载基础模型
    logger.info("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    logger.info(f"✓ 基础模型加载完成")
    
    # 2. 加载 LoRA 适配器
    logger.info("正在加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    logger.info(f"✓ LoRA 适配器加载完成")
    
    # 3. 合并权重
    logger.info("正在合并 LoRA 权重到基础模型...")
    merged_model = model.merge_and_unload()
    logger.info(f"✓ 权重合并完成")
    
    # 4. 保存合并后的模型
    logger.info(f"正在保存完整模型到: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # 使用 safetensors 格式
        max_shard_size="5GB"
    )
    logger.info(f"✓ 模型保存完成")
    
    # 5. 复制 tokenizer
    logger.info("正在复制 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    logger.info(f"✓ Tokenizer 保存完成")
    
    # 6. 显示文件大小
    total_size = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file())
    logger.info("=" * 60)
    logger.info(f"✓ 合并完成！")
    logger.info(f"输出目录: {output_path}")
    logger.info(f"总大小: {total_size / (1024**3):.2f} GB")
    logger.info("=" * 60)
    logger.info("现在可以直接使用合并后的模型进行推理，无需加载基础模型")


def main():
    """主函数"""
    try:
        merge_lora_to_base(
            base_model_path=BASE_MODEL_PATH,
            lora_adapter_path=LORA_ADAPTER_PATH,
            output_path=OUTPUT_PATH,
            device="auto"
        )
    except Exception as e:
        logger.error(f"合并失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
