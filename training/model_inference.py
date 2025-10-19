#!/usr/bin/env python3
"""
简易终端对话工具（一键运行）

默认配置：
    - 模型路径：models/qwen（硬编码）
    - 交互模式：终端输入对话
    
用法：
    直接运行即可开始对话
    python training/model_inference.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 硬编码配置 ==========
MODEL_PATH = "models/qwen"
SYSTEM_PROMPT = "你是一个乐于助人的AI助手。"

# ========== 生成参数（可在代码中调整） ==========
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05


def resolve_device(device_option: str) -> torch.device:
    if device_option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_option)


def load_model(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    return tokenizer, model


def build_chat_input(tokenizer, user_prompt: str) -> str:
    """构建对话输入"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 若模型没有chat模板，使用简单拼接
    return f"{SYSTEM_PROMPT}\n\n用户: {user_prompt}\n助手:"


def determine_context_size(model) -> int:
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 4096


def generate_response(model, tokenizer, device, prompt_text: str) -> str:
    """生成模型回复"""
    context_limit = determine_context_size(model)
    max_input_length = max(8, context_limit - MAX_NEW_TOKENS - 32)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    """主函数：交互式对话"""
    print("=" * 60)
    print("简易终端对话工具")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"生成参数: max_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}, top_p={TOP_P}")
    print("提示: 输入 'exit' 或 'quit' 退出对话")
    print("=" * 60)
    print()

    # 加载模型
    print("正在加载模型...")
    device = resolve_device("auto")
    tokenizer, model = load_model(MODEL_PATH, device)
    print(f"✓ 模型已加载到设备: {device}")
    print()

    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("再见！")
                break
            
            # 构建输入并生成回复
            prompt_text = build_chat_input(tokenizer, user_input)
            print("AI: ", end="", flush=True)
            answer = generate_response(model, tokenizer, device, prompt_text)
            print(answer)
            print()
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            print()


if __name__ == "__main__":
    main()
