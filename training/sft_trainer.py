"""
SFT模型微调器
专门用于训练Verilog专家模型

默认配置：
- 基础模型：models/qwen（硬编码）
- 训练数据：data/verilog.json（硬编码）
- 输出目录：models/qwen_finetuned

用法：
    直接运行即可开始微调
    python training/sft_trainer.py
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 硬编码配置（一键运行） ==========
BASE_MODEL_PATH = "models/qwen"
DATASET_PATH = "data/verilog.json"
OUTPUT_DIR = "models/qwen_finetuned"


@dataclass
class SFTConfig:
    """SFT训练配置（8GB显存优化版）"""
    model_name: str = field(default=BASE_MODEL_PATH)
    dataset_path: str = field(default=DATASET_PATH)
    output_dir: str = field(default=OUTPUT_DIR)
    max_seq_length: int = field(default=512)  # 降低到512以节省显存
    num_train_epochs: int = field(default=50)
    per_device_train_batch_size: int = field(default=1)  # 降低到1
    gradient_accumulation_steps: int = field(default=16)  # 增加到16保持有效批次
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.1)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=8)  # 降低LoRA秩以节省显存
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    train_split_ratio: float = field(default=0.9)
    use_gradient_checkpointing: bool = field(default=True)  # 启用梯度检查点


class VerilogDatasetBuilder:
    """Verilog数据集构建器（从JSON文件加载）"""
    
    def __init__(self, dataset_path: str):
        """初始化数据集构建器
        
        Args:
            dataset_path: JSON数据集文件路径
        """
        self.dataset_path = dataset_path
        logger.info(f"数据集路径: {dataset_path}")
    
    def load_and_prepare_dataset(self) -> Dataset:
        """从JSON文件加载并准备数据集"""
        logger.info(f"正在加载数据集: {self.dataset_path}")
        
        # 读取JSON文件
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"原始数据集大小: {len(raw_data)} 条")
        
        # 转换为训练格式
        formatted_data = []
        for item in raw_data:
            instruction = item.get('instruction', '')
            output = item.get('output', '')
            
            # 构建训练文本
            text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}\n### 结束"
            formatted_data.append({"text": text})
        
        logger.info(f"格式化后数据集大小: {len(formatted_data)} 条")
        return Dataset.from_list(formatted_data)


class SFTRunner:
    """SFT训练器"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.add(f"{self.output_dir}/sft_training.log", rotation="10 MB")
        logger.info("SFT训练器初始化")
    
    def train(self):
        """执行SFT训练"""
        logger.info("=" * 60)
        logger.info("开始 SFT 微调训练")
        logger.info("=" * 60)
        logger.info(f"基础模型: {self.config.model_name}")
        logger.info(f"数据集: {self.config.dataset_path}")
        logger.info(f"输出目录: {self.config.output_dir}")
        logger.info("=" * 60)
        
        # 加载数据集
        dataset_builder = VerilogDatasetBuilder(self.config.dataset_path)
        full_dataset = dataset_builder.load_and_prepare_dataset()
        
        # 分割训练集和验证集
        dataset_dict = full_dataset.train_test_split(
            test_size=1 - self.config.train_split_ratio,
            seed=42
        )
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['test']
        
        logger.info(f"训练集大小: {len(train_dataset)} 条")
        logger.info(f"验证集大小: {len(eval_dataset)} 条")
        
        # 加载模型和分词器
        model, tokenizer = self._setup_model()
        
        # 预处理数据集：为 SFTTrainer 添加必要的字段
        def preprocess_function(examples):
            # 确保数据格式正确
            return {
                "input_ids": tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding=False,
                )["input_ids"]
            }
        
        logger.info("正在预处理数据集...")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="预处理训练集"
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="预处理验证集"
        )
        
        # 训练参数（8GB显存优化版）
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=1,  # 评估时也使用小批次
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,  # 禁用以节省显存
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            bf16=False,
            optim="adamw_torch",
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            max_grad_norm=1.0,
            dataloader_pin_memory=False,  # 禁用以节省显存
        )
        
        # 创建数据整理器（用于语言建模）
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言建模（非掩码语言建模）
        )
        
        # 创建训练器（使用标准 Transformers Trainer）
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        logger.info("训练器创建成功，准备开始训练...")
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"SFT训练完成，模型保存在: {self.config.output_dir}")
        
        return str(self.output_dir)
    
    def _setup_model(self):
        """设置模型和分词器"""
        logger.info("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"模型设备: {model.device if hasattr(model, 'device') else '分布式'}")
        
        # 启用梯度检查点以节省显存
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("✓ 已启用梯度检查点（节省显存）")
        
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model, tokenizer


def main():
    """主函数：一键运行微调"""
    print("=" * 60)
    print("OptiHDL SFT 微调训练（一键运行版）")
    print("=" * 60)
    print(f"基础模型: {BASE_MODEL_PATH}")
    print(f"训练数据: {DATASET_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    print()
    
    # 使用默认配置
    config = SFTConfig()
    
    # 创建训练器并开始训练
    trainer = SFTRunner(config)
    model_path = trainer.train()
    
    print()
    print("=" * 60)
    print(f"✓ 训练完成！模型保存在: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
