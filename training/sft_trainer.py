"""
模块1: SFT模型微调器
专门用于训练Verilog专家模型
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, HfArgumentParser
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class SFTConfig:
    """SFT训练配置"""
    model_name: str = field(default="codellama/CodeLlama-7b-Python-hf")
    output_dir: str = field(default="./sft_model")
    max_seq_length: int = field(default=2048)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.1)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    dataset_size: int = field(default=2000)


class VerilogDatasetBuilder:
    """Verilog数据集构建器"""
    
    def create_dataset(self, num_samples: int = 2000) -> Dataset:
        """创建Verilog训练数据集"""
        logger.info(f"创建Verilog数据集，样本数: {num_samples}")
        
        templates = [
            self._counter_template,
            self._adder_template,
            self._mux_template,
            self._fsm_template,
            self._memory_template
        ]
        
        data = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            example = template(i)
            
            prompt = f"请实现以下Verilog模块：{example['description']}"
            completion = example['code']
            
            data.append({
                "text": f"### 指令:\n{prompt}\n\n### 回答:\n{completion}\n### 结束"
            })
        
        return Dataset.from_list(data)
    
    def _counter_template(self, idx: int) -> Dict[str, str]:
        """计数器模板"""
        width = 4 + (idx % 8)
        return {
            "description": f"{width}位计数器，带复位功能",
            "code": f"""module counter_{width}bit(
    input wire clk,
    input wire reset,
    output reg [{width-1}:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= {width}'b0;
        else
            count <= count + 1'b1;
    end
endmodule"""
        }
    
    def _adder_template(self, idx: int) -> Dict[str, str]:
        """加法器模板"""
        width = 4 + (idx % 4)
        return {
            "description": f"{width}位全加器",
            "code": f"""module adder_{width}bit(
    input wire [{width-1}:0] a,
    input wire [{width-1}:0] b,
    input wire cin,
    output wire [{width-1}:0] sum,
    output wire cout
);
    assign {{cout, sum}} = a + b + cin;
endmodule"""
        }
    
    def _mux_template(self, idx: int) -> Dict[str, str]:
        """多路选择器模板"""
        sel_bits = 2 + (idx % 2)
        inputs = 2 ** sel_bits
        return {
            "description": f"{inputs}选1多路选择器",
            "code": f"""module mux_{inputs}to1(
    input wire [{inputs-1}:0] data_in,
    input wire [{sel_bits-1}:0] sel,
    output reg data_out
);
    always @(*) begin
        data_out = data_in[sel];
    end
endmodule"""
        }
    
    def _fsm_template(self, idx: int) -> Dict[str, str]:
        """状态机模板"""
        return {
            "description": "简单的3状态FSM",
            "code": """module simple_fsm(
    input wire clk,
    input wire reset,
    input wire start,
    output reg [1:0] state,
    output reg done
);
    parameter IDLE = 2'b00;
    parameter WORK = 2'b01;
    parameter DONE = 2'b10;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            done <= 1'b0;
        end else begin
            case (state)
                IDLE: if (start) state <= WORK;
                WORK: state <= DONE;
                DONE: begin
                    state <= IDLE;
                    done <= 1'b1;
                end
            endcase
        end
    end
endmodule"""
        }
    
    def _memory_template(self, idx: int) -> Dict[str, str]:
        """存储器模板"""
        depth = 16 * (2 ** (idx % 3))
        addr_width = (idx % 3) + 4
        return {
            "description": f"{depth}x8位单端口RAM",
            "code": f"""module ram_{depth}x8(
    input wire clk,
    input wire we,
    input wire [{addr_width-1}:0] addr,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);
    reg [7:0] memory [{depth-1}:0];
    
    always @(posedge clk) begin
        if (we) begin
            memory[addr] <= data_in;
        end
        data_out <= memory[addr];
    end
endmodule"""
        }


class SFTTrainer:
    """SFT训练器"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.add(f"{self.output_dir}/sft_training.log", rotation="10 MB")
        logger.info("SFT训练器初始化")
    
    def train(self):
        """执行SFT训练"""
        logger.info("开始SFT训练")
        
        # 创建数据集
        dataset_builder = VerilogDatasetBuilder()
        train_dataset = dataset_builder.create_dataset(self.config.dataset_size)
        eval_dataset = dataset_builder.create_dataset(self.config.dataset_size // 5)
        
        # 加载模型和分词器
        model, tokenizer = self._setup_model()
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
        )
        
        # 创建训练器
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"SFT训练完成，模型保存在: {self.config.output_dir}")
        
        return str(self.output_dir)
    
    def _setup_model(self):
        """设置模型和分词器"""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
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
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT模型训练器")
    parser.add_argument("--model", default="codellama/CodeLlama-7b-Python-hf", help="基础模型")
    parser.add_argument("--output", default="./sft_model", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--dataset_size", type=int, default=2000, help="数据集大小")
    parser.add_argument("--use_lora", action="store_true", help="使用LoRA")
    
    args = parser.parse_args()
    
    config = SFTConfig(
        model_name=args.model,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        dataset_size=args.dataset_size,
        use_lora=args.use_lora
    )
    
    trainer = SFTTrainer(config)
    model_path = trainer.train()
    
    print(f"训练完成！模型保存在: {model_path}")


if __name__ == "__main__":
    main()
