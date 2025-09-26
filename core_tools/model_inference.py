#!/usr/bin/env python3
"""
模型推理工具 - 展示如何调用Verilog优化模型进行代码生成

这个文件展示了完整的模型推理流程，包括：
1. 模型加载和配置
2. 提示词构建
3. 生成参数设置
4. 推理执行
5. 结果处理

使用方法:
    python core_tools/model_inference.py --input models/test.v --model models/qwen8b
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import re


class VerilogModelInference:
    """Verilog代码优化模型推理器"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化模型推理器
        
        Args:
            model_path: 模型路径
            device: 设备选择 ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        
        # 加载模型和分词器
        logger.info(f"加载模型: {model_path}")
        self.tokenizer, self.model = self._load_model()
        
        # 推理配置
        self.max_new_tokens = 2048
        self.temperature = 0.8
        self.top_p = 0.9
        self.top_k = 50
        self.repetition_penalty = 1.05
        
        logger.info(f"模型加载完成，设备: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """获取推理设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """加载模型和分词器"""
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            model = model.to("cpu")
        
        model.eval()
        return tokenizer, model
    
    def _get_context_window(self) -> int:
        """获取模型上下文窗口大小"""
        if hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        elif hasattr(self.model.config, 'max_sequence_length'):
            return self.model.config.max_sequence_length
        else:
            # 默认值，支持常见模型
            return 8192
    
    def build_optimization_prompt(self, source_code: str, target_description: str = "") -> str:
        """
        构建优化提示词
        
        Args:
            source_code: 源Verilog代码
            target_description: 优化目标描述
            
        Returns:
            构建好的提示词
        """
        # 基础约束
        constraints = (
            "要求：1) 严格保留顶层模块的接口(模块名、端口列表、位宽、方向、参数)完全一致；"
            "2) 仅修改模块体内实现，不改变接口和时序语义；"
            "3) 只输出优化后的 Verilog 代码(包含完整 module...endmodule)，不要额外说明。"
        )
        
        # 优化提示
        optimization_hints = [
            "4) 优化目标：减少面积、触发器数量和逻辑深度；",
            "5) 可尝试：常量折叠、冗余消除、表达式简化、资源共享等优化技术；",
            "6) 保持代码可读性和功能正确性。"
        ]
        
        if target_description:
            optimization_hints.insert(0, f"4) 特定优化目标：{target_description}；")
        
        # 构建完整提示
        prompt = f"""请优化以下Verilog代码以提升PPA性能：

{constraints}
{"".join(optimization_hints)}

原始代码：
```verilog
{source_code.strip()}
```

优化后的代码：
```verilog
"""
        
        return prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 粗略估算：中文约3字符/token，英文约4字符/token
        return len(text) // 3
    
    def _adjust_generation_params(self, prompt_length: int) -> Dict[str, Any]:
        """根据输入长度动态调整生成参数"""
        context_max = self._get_context_window()
        estimated_prompt_tokens = self._estimate_tokens(prompt_length)
        
        # 动态调整生成长度
        if estimated_prompt_tokens < 2000:
            effective_new = self.max_new_tokens
        elif estimated_prompt_tokens < 4000:
            effective_new = max(1024, int(self.max_new_tokens * 0.8))
        else:
            effective_new = max(768, context_max - estimated_prompt_tokens - 200)
        
        return {
            "max_new_tokens": effective_new,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
    
    def generate_code(self, prompt: str, use_chat_template: bool = True) -> Optional[str]:
        """
        生成优化代码
        
        Args:
            prompt: 输入提示词
            use_chat_template: 是否使用聊天模板
            
        Returns:
            生成的代码或None
        """
        logger.info("开始生成优化代码...")
        
        # 动态调整生成参数
        gen_params = self._adjust_generation_params(len(prompt))
        logger.info(f"生成参数: {gen_params}")
        
        try:
            # 构建输入
            if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "你是精通数字电路和Verilog的资深芯片工程师，目标是优化面积/触发器数/逻辑深度。"},
                    {"role": "user", "content": prompt},
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = prompt
            
            # 分词
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._get_context_window() - gen_params["max_new_tokens"] - 50
            )
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # 生成
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_params
                )
            
            generation_time = time.time() - start_time
            
            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            logger.info(f"生成完成，耗时: {generation_time:.2f}s，长度: {len(generated_text)}字符")
            
            # 提取Verilog代码
            extracted_code = self._extract_verilog_code(generated_text)
            return extracted_code
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return None
    
    def _extract_verilog_code(self, text: str) -> Optional[str]:
        """从生成文本中提取Verilog代码"""
        # 方法1: 提取```verilog代码块
        verilog_pattern = r'```verilog\s*(.*?)```'
        match = re.search(verilog_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if self._has_complete_module(code):
                return code
        
        # 方法2: 提取module...endmodule块
        module_pattern = r'(module\s+\w+.*?endmodule)'
        matches = re.findall(module_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # 取最长的匹配
            longest_match = max(matches, key=len)
            return longest_match.strip()
        
        # 方法3: 如果包含module关键字，返回整个文本
        if 'module' in text.lower() and 'endmodule' in text.lower():
            return text.strip()
        
        return None
    
    def _has_complete_module(self, text: str) -> bool:
        """检查是否包含完整的module定义"""
        return ("module" in text.lower()) and ("endmodule" in text.lower())
    
    def optimize_verilog(self, source_code: str, target_description: str = "") -> Dict[str, Any]:
        """
        完整的Verilog优化流程
        
        Args:
            source_code: 源代码
            target_description: 优化目标
            
        Returns:
            优化结果字典
        """
        logger.info("开始Verilog代码优化...")
        
        # 构建提示词
        prompt = self.build_optimization_prompt(source_code, target_description)
        logger.info(f"提示词长度: {len(prompt)}字符")
        
        # 生成优化代码
        optimized_code = self.generate_code(prompt)
        
        if optimized_code:
            return {
                "success": True,
                "original_code": source_code,
                "optimized_code": optimized_code,
                "prompt": prompt,
                "model_path": self.model_path
            }
        else:
            return {
                "success": False,
                "error": "代码生成失败",
                "original_code": source_code,
                "prompt": prompt
            }
    
    def batch_optimize(self, code_list: List[str], target_descriptions: List[str] = None) -> List[Dict[str, Any]]:
        """
        批量优化多个Verilog代码
        
        Args:
            code_list: 代码列表
            target_descriptions: 对应的优化目标列表
            
        Returns:
            优化结果列表
        """
        if target_descriptions is None:
            target_descriptions = [""] * len(code_list)
        
        results = []
        for i, (code, target) in enumerate(zip(code_list, target_descriptions)):
            logger.info(f"优化第 {i+1}/{len(code_list)} 个代码...")
            result = self.optimize_verilog(code, target)
            results.append(result)
        
        return results


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Verilog模型推理工具")
    parser.add_argument('--input', '-i', required=True, help='输入Verilog文件')
    parser.add_argument('--model', '-m', required=True, help='模型路径')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--target', '-t', default='', help='优化目标描述')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='推理设备')
    parser.add_argument('--temperature', type=float, default=0.8, help='生成温度')
    parser.add_argument('--max-tokens', type=int, default=2048, help='最大生成tokens')
    parser.add_argument('--top-p', type=float, default=0.9, help='nucleus sampling参数')
    parser.add_argument('--top-k', type=int, default=50, help='top-k sampling参数')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    if not args.verbose:
        logger.remove()
        logger.add(lambda msg: None)  # 静默模式
    
    # 读取输入代码
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    source_code = input_file.read_text(encoding='utf-8')
    
    # 创建推理器
    try:
        inferencer = VerilogModelInference(args.model, args.device)
        
        # 设置生成参数
        inferencer.temperature = args.temperature
        inferencer.max_new_tokens = args.max_tokens
        inferencer.top_p = args.top_p
        inferencer.top_k = args.top_k
        
        # 执行优化
        result = inferencer.optimize_verilog(source_code, args.target)
        
        if result["success"]:
            print("=== 优化成功 ===")
            if args.output:
                # 保存到文件
                output_file = Path(args.output)
                output_file.write_text(result["optimized_code"], encoding='utf-8')
                print(f"优化代码已保存到: {args.output}")
            else:
                # 输出到控制台
                print("\n=== 优化后的代码 ===")
                print(result["optimized_code"])
        else:
            print(f"优化失败: {result['error']}")
            
    except Exception as e:
        print(f"推理失败: {e}")


if __name__ == "__main__":
    main()
