"""
模块3: RL推理优化器
使用训练好的模型进行推理时优化，不训练模型参数
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from loguru import logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_tools.eda_wrapper import EDAWrapper, analyze_verilog_api


class RLOptimizer:
    """RL推理优化器"""
    
    def __init__(self, model_path: str, max_iterations: int = 10, 
                 population_size: int = 5, temperature: float = 0.8):
        """
        初始化RL优化器
        
        Args:
            model_path: 训练好的SFT模型路径
            max_iterations: 最大优化迭代次数
            population_size: 每次生成的候选代码数量
            temperature: 生成温度，控制多样性
        """
        self.model_path = model_path
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.temperature = temperature
        
        # 加载模型
        self.tokenizer, self.model = self._load_model()
        
        # EDA工具
        self.eda = EDAWrapper()
        
        # 优化历史
        self.optimization_history = []
        
        logger.info(f"RL优化器初始化完成，模型: {model_path}")
    
    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """加载训练好的模型"""
        logger.info(f"加载模型: {self.model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 检查是否是PEFT模型
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            # 加载基础模型
            with open(os.path.join(self.model_path, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # 加载PEFT适配器
            model = PeftModel.from_pretrained(base_model, self.model_path)
            model = model.merge_and_unload()  # 合并权重以提高推理速度
        else:
            # 直接加载完整模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        model.eval()
        return tokenizer, model
    
    def optimize(self, input_code: str, target_description: str = "") -> Dict[str, Any]:
        """
        优化Verilog代码
        
        Args:
            input_code: 输入的Verilog代码
            target_description: 优化目标描述
        
        Returns:
            优化结果字典
        """
        logger.info("开始RL优化过程")
        
        # 获取初始基准
        initial_metrics = self._evaluate_code(input_code)
        if not initial_metrics["success"]:
            return {
                "success": False,
                "error": f"初始代码评估失败: {initial_metrics['error']}",
                "original_code": input_code
            }
        
        best_code = input_code
        best_metrics = initial_metrics["data"]
        best_score = self._calculate_score(best_metrics)
        
        self.optimization_history = [{
            "iteration": 0,
            "code": input_code,
            "metrics": best_metrics,
            "score": best_score,
            "is_best": True
        }]
        
        logger.info(f"初始分数: {best_score:.4f}")
        
        # 迭代优化
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"优化迭代 {iteration}/{self.max_iterations}")
            
            # 生成候选代码
            candidates = self._generate_candidates(best_code, target_description)
            
            # 评估候选代码
            iteration_best_score = best_score
            iteration_best_code = best_code
            iteration_best_metrics = best_metrics
            
            for i, candidate in enumerate(candidates):
                logger.info(f"评估候选代码 {i+1}/{len(candidates)}")
                
                metrics_result = self._evaluate_code(candidate)
                if not metrics_result["success"]:
                    logger.warning(f"候选代码 {i+1} 评估失败: {metrics_result['error']}")
                    continue
                
                metrics = metrics_result["data"]
                score = self._calculate_score(metrics)
                
                # 记录历史
                self.optimization_history.append({
                    "iteration": iteration,
                    "candidate": i + 1,
                    "code": candidate,
                    "metrics": metrics,
                    "score": score,
                    "is_best": False
                })
                
                # 更新最佳结果
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_code = candidate
                    iteration_best_metrics = metrics
                    
                    # 标记为最佳
                    self.optimization_history[-1]["is_best"] = True
                    logger.info(f"发现更好的代码，分数: {score:.4f}")
            
            # 更新全局最佳
            if iteration_best_score > best_score:
                best_score = iteration_best_score
                best_code = iteration_best_code
                best_metrics = iteration_best_metrics
                logger.info(f"全局最佳分数更新: {best_score:.4f}")
            else:
                logger.info(f"本轮无改进，当前最佳分数: {best_score:.4f}")
        
        # 计算改进情况
        init_score = self._calculate_score(initial_metrics["data"])
        improvement = {
            "area_improvement": (initial_metrics["data"]["area"] - best_metrics["area"]) / max(1e-9, initial_metrics["data"]["area"]) * 100,
            "ff_improvement": (initial_metrics["data"].get("num_ff", 0) - best_metrics.get("num_ff", 0)) / max(1e-9, initial_metrics["data"].get("num_ff", 0) or 1) * 100,
            "depth_improvement": (initial_metrics["data"].get("logic_depth", 0) - best_metrics.get("logic_depth", 0)) / max(1e-9, initial_metrics["data"].get("logic_depth", 0) or 1) * 100,
            "score_improvement": (best_score - init_score) / max(1e-9, init_score or 1) * 100
        }
        
        return {
            "success": True,
            "original_code": input_code,
            "optimized_code": best_code,
            "original_metrics": initial_metrics["data"],
            "optimized_metrics": best_metrics,
            "improvement": improvement,
            "optimization_history": self.optimization_history,
            "total_iterations": self.max_iterations,
            "total_candidates": len(self.optimization_history) - 1
        }
    
    def _generate_candidates(self, base_code: str, target_description: str) -> List[str]:
        """生成候选代码"""
        candidates = []
        
        # 构建提示
        if target_description:
            prompt = f"请优化以下Verilog代码以{target_description}：\n{base_code}\n\n优化后的代码："
        else:
            prompt = f"请优化以下Verilog代码以提高性能和减少面积：\n{base_code}\n\n优化后的代码："
        
        # 生成多个候选
        for i in range(self.population_size):
            try:
                candidate = self._generate_single_candidate(prompt)
                if candidate and candidate != base_code:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"生成候选代码 {i+1} 失败: {e}")
        
        logger.info(f"生成了 {len(candidates)} 个有效候选代码")
        return candidates
    
    def _generate_single_candidate(self, prompt: str) -> Optional[str]:
        """生成单个候选代码"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的代码部分
        if "优化后的代码：" in generated_text:
            code_part = generated_text.split("优化后的代码：")[-1].strip()
        else:
            code_part = generated_text[len(prompt):].strip()
        
        # 简单的代码清理
        if code_part.startswith("```verilog"):
            code_part = code_part[10:]
        if code_part.endswith("```"):
            code_part = code_part[:-3]
        
        return code_part.strip()
    
    def _evaluate_code(self, code: str) -> Dict[str, Any]:
        """评估代码质量"""
        return analyze_verilog_api(code)
    
    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合分数：面积/FF/深度与通过项"""
        # 工具通过项：语法/综合/等价
        pass_bonus = 0.0
        if metrics.get("syntax_ok"):
            pass_bonus += 1.0
        if metrics.get("synth_ok"):
            pass_bonus += 1.0
        if metrics.get("equiv_ok"):
            pass_bonus += 1.0

        # 面积（越小越好）
        area = max(0.0, float(metrics.get("area", 0)))
        area_score = 1.0 / (1.0 + area / 1000.0)

        # 触发器数量（越少越好）
        ff = max(0.0, float(metrics.get("num_ff", 0)))
        ff_score = 1.0 / (1.0 + ff / 1000.0)

        # 逻辑深度（越小越好）
        depth = max(0.0, float(metrics.get("logic_depth", 0)))
        depth_score = 1.0 / (1.0 + depth / 10.0)

        # 综合：强调面积与FF，其次深度，并加入通过项加分
        total_score = 0.45 * area_score + 0.35 * ff_score + 0.20 * depth_score + 0.10 * pass_bonus
        return total_score
    
    def save_optimization_report(self, result: Dict[str, Any], output_path: str):
        """保存优化报告"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "optimization_config": {
                "max_iterations": self.max_iterations,
                "population_size": self.population_size,
                "temperature": self.temperature
            },
            "result": result
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化报告已保存: {output_path}")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'eda'):
            self.eda.cleanup()


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL推理优化器")
    parser.add_argument("--model", required=True, help="SFT模型路径")
    parser.add_argument("--input", "-i", help="输入Verilog文件")
    parser.add_argument("--code", "-c", help="直接输入Verilog代码")
    parser.add_argument("--target", "-t", default="", help="优化目标描述")
    parser.add_argument("--iterations", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--population", type=int, default=5, help="候选代码数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成温度")
    parser.add_argument("--output", "-o", help="输出目录")
    
    args = parser.parse_args()
    
    # 获取输入代码
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_code = f.read()
    elif args.code:
        input_code = args.code
    else:
        print("请指定输入文件 (--input) 或直接输入代码 (--code)")
        return
    
    # 创建优化器
    optimizer = RLOptimizer(
        model_path=args.model,
        max_iterations=args.iterations,
        population_size=args.population,
        temperature=args.temperature
    )
    
    try:
        # 执行优化
        result = optimizer.optimize(input_code, args.target)
        
        if result["success"]:
            print("=== 优化完成 ===")
            print(f"面积改进: {result['improvement']['area_improvement']:.2f}%")
            print(f"触发器数改进: {result['improvement']['ff_improvement']:.2f}%")
            print(f"逻辑深度改进: {result['improvement']['depth_improvement']:.2f}%")
            print(f"总分改进: {result['improvement']['score_improvement']:.2f}%")
            
            # 保存结果
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(exist_ok=True)
                
                # 保存优化后的代码
                with open(output_dir / "optimized_code.v", 'w', encoding='utf-8') as f:
                    f.write(result["optimized_code"])
                
                # 保存优化报告
                optimizer.save_optimization_report(result, str(output_dir / "optimization_report.json"))
                
                print(f"结果已保存到: {args.output}")
            else:
                print("\n=== 优化后的代码 ===")
                print(result["optimized_code"])
        else:
            print(f"优化失败: {result['error']}")
    
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()
