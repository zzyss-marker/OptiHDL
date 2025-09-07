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
import re
from difflib import SequenceMatcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_tools.eda_wrapper import EDAWrapper, analyze_verilog_api


class RLOptimizer:
    """RL推理优化器"""
    
    def __init__(self, model_path: str, max_iterations: int = 10, 
                 population_size: int = 1, temperature: float = 0.8,
                 max_new_tokens: int = 1024,
                 debug_gen: bool = False,
                 debug_dir: Optional[str] = None,
                 rl_mode: bool = True,
                 base_top_p: float = 0.9,
                 base_top_k: int = 50,
                 base_rep_penalty: float = 1.05):
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
        self.max_new_tokens = max_new_tokens
        
        # 加载模型
        self.tokenizer, self.model = self._load_model()
        
        # EDA工具
        self.eda = EDAWrapper()
        
        # 优化历史
        self.optimization_history = []
        
        # RL 动态调整参数
        self.rl_mode = rl_mode
        self.base_temperature = temperature
        self.base_top_p = base_top_p
        self.base_top_k = base_top_k
        self.base_rep_penalty = base_rep_penalty
        self.score_history = []  # 每轮最佳分数历史
        self.improvement_streak = 0  # 连续改进轮数
        self.stagnation_count = 0   # 连续无改进轮数
        self.decline_count = 0      # 连续下降轮数
        self.exploration_factor = 0.3  # 探索因子
        self.last_best_score = 0.0  # 上一轮最佳分数
        self.score_trend = 0.0      # 分数趋势
        
        # 调试
        self.debug_gen = debug_gen
        # 默认调试目录
        if debug_dir is None and self.debug_gen:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_root = Path("outputs") / f"debug_{ts}"
            debug_root.mkdir(parents=True, exist_ok=True)
            self.debug_dir = str(debug_root)
        else:
            self.debug_dir = debug_dir
        
        # 模块名（将在optimize方法中从代码中提取）
        self.module_name = None
        
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
        # 提取模块名
        self.module_name = self._extract_module_name(input_code)
        if not self.module_name:
            return {
                "success": False,
                "error": "无法从代码中提取模块名"
            }
        
        # 初始评估和预检查
        logger.info("开始RL优化过程")
        try:
            initial_data = self.eda.synthesize(input_code, self.module_name, target_freq=100.0)
        except Exception as e:
            return {
                "success": False,
                "error": f"初始代码综合失败: {str(e)}"
            }
        
        # 检查初始代码语法和综合状态
        if not initial_data.get("syntax_ok", False) or not initial_data.get("synth_ok", False):
            logger.error("❌ 初始代码语法或综合检查失败，停止RL训练")
            logger.error(f"语法检查: {initial_data.get('syntax_ok', False)}")
            logger.error(f"综合检查: {initial_data.get('synth_ok', False)}")
            return {
                "success": False,
                "error": "初始代码存在语法错误或综合失败，无法进行优化训练。请先修复代码语法问题。"
            }
        
        initial_score = self._calculate_score(initial_data)
        self.best_score = initial_score
        self.best_code = input_code
        self.best_metrics = initial_data
        self.last_best_score = initial_score
        logger.info(f"✅ 初始代码检查通过，分数: {initial_score:.4f}")
        
        # 初始化优化历史和变量
        best_score = initial_score
        best_code = input_code
        best_metrics = initial_data
        
        self.optimization_history = [{
            "iteration": 0,
            "code": input_code,
            "metrics": initial_data,
            "score": initial_score,
            "is_best": True
        }]
        
        # 迭代优化
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"优化迭代 {iteration}/{self.max_iterations}")
            
            # RL动态调整策略
            if self.rl_mode and iteration > 1:
                self._update_rl_strategy(iteration, best_score)
            
            # 生成候选代码
            candidates = self._generate_candidates(best_code, target_description, iteration, best_score, best_metrics)
            
            # 评估候选代码
            iteration_best_score = best_score
            iteration_best_code = best_code
            iteration_best_metrics = best_metrics
            
            for i, candidate in enumerate(candidates):
                logger.info(f"评估候选代码 {i+1}/{len(candidates)}")
                # 预处理：确保候选包含完整模块并与基准模块名一致
                module_name = self._detect_top_module(best_code) or "top"
                if not self._has_complete_module(candidate):
                    logger.warning(f"候选代码 {i+1} 缺少完整 module/endmodule，丢弃")
                    self._maybe_dump_text(iteration, i+1, "extracted_incomplete.v", candidate)
                    continue
                candidate_named = self._try_fix_module_name(candidate, module_name)
                candidate_fixed = self._align_module_header(best_code, candidate_named, module_name)
                # 调试保存
                self._maybe_dump_text(iteration, i+1, "candidate_fixed.v", candidate_fixed)
                
                # 先评估候选代码获取分数
                metrics_result = self._evaluate_code(candidate_fixed)
                if not metrics_result["success"]:
                    logger.warning(f"候选代码 {i+1} 评估失败: {metrics_result['error']}")
                    continue
                
                metrics = metrics_result["data"]
                score = self._calculate_score(metrics)
                
                # 记录候选历史（先记录再检查等价）
                self.optimization_history.append({
                    "iteration": iteration,
                    "candidate": i+1,
                    "code": candidate_fixed,
                    "metrics": metrics,
                    "score": score,
                    "is_best": False,
                    "equiv_checked": False
                })
                
                logger.info(f"候选 {i+1} 分数: {score:.4f} (面积: {metrics['area']}, FF: {metrics.get('num_ff', 0)}, 深度: {metrics.get('logic_depth', 0)})")
                
                # 直接基于分数更新最佳候选（已移除等价检查）
                if score > iteration_best_score:
                    logger.info(f"✅ 候选 {i+1} 分数更高，直接采用: {score:.4f}")
                    iteration_best_score = score
                    iteration_best_code = candidate_fixed
                    iteration_best_metrics = metrics
                    
                    # 保存最佳候选
                    if self.debug_dir:
                        best_file = Path(self.debug_dir) / f"best_candidate_iter_{iteration}.v"
                        best_file.write_text(candidate_fixed, encoding='utf-8')
                else:
                    decline = (iteration_best_score - score) / max(iteration_best_score, 1e-6) * 100
                    logger.debug(f"候选 {i+1} 分数较低: {score:.4f} (低 {decline:.2f}%)")
            
            # 更新全局最佳并记录RL状态
            self.score_history.append(iteration_best_score)
            score_change = iteration_best_score - self.last_best_score if self.last_best_score > 0 else 0
            
            if iteration_best_score > best_score:
                best_score = iteration_best_score
                best_code = iteration_best_code
                best_metrics = iteration_best_metrics
                self.improvement_streak += 1
                self.stagnation_count = 0
                improvement = (iteration_best_score - self.last_best_score) / max(self.last_best_score, 1e-6) * 100
                logger.info(f"🎉 全局最佳更新: {best_score:.4f} (+{improvement:.2f}%)")
                if self.rl_mode:
                    logger.info(f"RL状态: 连续改进 {self.improvement_streak} 轮")
            elif abs(iteration_best_score - best_score) < 1e-6:
                # 分数相等，视为停滞
                self.improvement_streak = 0
                self.stagnation_count += 1
                logger.info(f"本轮无改进，维持最佳: {best_score:.4f}")
                if self.rl_mode:
                    logger.info(f"RL状态: 连续停滞 {self.stagnation_count} 轮")
            else:
                # 分数下降
                self.improvement_streak = 0
                self.stagnation_count += 1
                decline = (best_score - iteration_best_score) / max(best_score, 1e-6) * 100
                logger.warning(f"⚠️  本轮表现下降: {iteration_best_score:.4f} (较最佳低 {decline:.2f}%)")
                if self.rl_mode:
                    logger.warning(f"RL状态: 需要调整策略，停滞 {self.stagnation_count} 轮")
            
            self.last_best_score = iteration_best_score
        
        # 计算改进情况
        init_score = self._calculate_score(initial_data)
        improvement = {
            "area_improvement": (initial_data["area"] - best_metrics["area"]) / max(1e-9, initial_data["area"]) * 100,
            "ff_improvement": (initial_data.get("num_ff", 0) - best_metrics.get("num_ff", 0)) / max(1e-9, initial_data.get("num_ff", 0) or 1) * 100,
            "depth_improvement": (initial_data.get("logic_depth", 0) - best_metrics.get("logic_depth", 0)) / max(1e-9, initial_data.get("logic_depth", 0) or 1) * 100,
            "score_improvement": (best_score - init_score) / max(1e-9, init_score or 1) * 100
        }
        
        return {
            "success": True,
            "original_code": input_code,
            "optimized_code": best_code,
            "original_metrics": initial_data,
            "optimized_metrics": best_metrics,
            "improvement": improvement,
            "optimization_history": self.optimization_history,
            "total_iterations": self.max_iterations,
            "total_candidates": len(self.optimization_history) - 1
        }
    
    def _update_rl_strategy(self, iteration: int, current_best_score: float):
        """基于历史表现动态调整RL策略"""
        recent_scores = self.score_history[-5:] if len(self.score_history) >= 5 else self.score_history
        
        # 计算多层次趋势
        short_trend = 0.0  # 短期趋势（最近2轮）
        long_trend = 0.0   # 长期趋势（最近5轮）
        
        if len(recent_scores) >= 2:
            short_trend = (recent_scores[-1] - recent_scores[-2]) / max(abs(recent_scores[-2]), 1e-6)
        if len(recent_scores) >= 3:
            long_trend = (recent_scores[-1] - recent_scores[0]) / max(abs(recent_scores[0]), 1e-6)
        
        self.score_trend = short_trend
        
        # 检测得分下降
        if short_trend < -0.01:  # 得分下降超过1%
            self.decline_count += 1
            logger.warning(f"检测到得分下降: {short_trend:.4f}, 连续下降 {self.decline_count} 轮")
        else:
            self.decline_count = 0
        
        # 动态调整策略 - 更敏感的响应
        if self.decline_count >= 2:
            # 连续下降，立即大幅增加探索
            self.exploration_factor = min(0.9, self.exploration_factor + 0.15)
            self.temperature = min(1.3, self.base_temperature + 0.3)
            logger.warning(f"连续下降模式: 大幅增加探索，温度={self.temperature:.2f}")
        elif self.stagnation_count >= 2:
            # 停滞，适度增加探索
            self.exploration_factor = min(0.7, self.exploration_factor + 0.1)
            self.temperature = min(1.1, self.base_temperature + 0.2)
            logger.info(f"停滞模式: 增加探索，温度={self.temperature:.2f}")
        elif self.improvement_streak >= 3:
            # 连续改进，专注利用
            self.exploration_factor = max(0.1, self.exploration_factor - 0.05)
            self.temperature = max(0.6, self.base_temperature - 0.1)
            logger.info(f"利用模式: 专注当前方向，温度={self.temperature:.2f}")
        elif short_trend > 0.02:  # 单轮大幅改进
            # 发现好方向，适度利用
            self.exploration_factor = max(0.2, self.exploration_factor - 0.03)
            self.temperature = max(0.7, self.base_temperature - 0.05)
            logger.info(f"发现改进: 适度利用，温度={self.temperature:.2f}")
        else:
            # 平衡状态，逐渐回归基准
            target_exploration = 0.3
            self.exploration_factor += (target_exploration - self.exploration_factor) * 0.3
            self.temperature += (self.base_temperature - self.temperature) * 0.2
        
        logger.info(f"RL策略: 温度={self.temperature:.2f}, 探索={self.exploration_factor:.2f}, 短期趋势={short_trend:.4f}, 长期趋势={long_trend:.4f}")

    def _generate_candidates(self, base_code: str, target_description: str, iteration: int, 
                           current_score: float = 0, current_metrics: Dict = None) -> List[str]:
        """生成候选代码"""
        candidates = []
        
        # 构建动态提示
        constraints = self._build_dynamic_constraints(iteration, current_score, current_metrics)
        if target_description:
            prompt = f"请优化以下Verilog代码以{target_description}，{constraints}\n\n原始代码：\n{base_code}\n\n优化后的代码："
        else:
            prompt = f"请优化以下Verilog代码以提高性能和减少面积，{constraints}\n\n原始代码：\n{base_code}\n\n优化后的代码："
        
        # RL模式：单候选生成，非RL模式：多候选生成
        for i in range(self.population_size):
            try:
                candidate = self._generate_single_candidate(prompt, iteration, i+1, current_score, current_metrics)
                if not candidate:
                    continue
                if self._is_meaningfully_different(base_code, candidate):
                    logger.debug(f"候选 {i+1} 长度: {len(candidate)} 字符")
                    candidates.append(candidate)
                else:
                    logger.debug(f"候选 {i+1} 与基准过于相似，已丢弃")
                    if self.rl_mode and (self.stagnation_count >= 2 or self.decline_count >= 1):
                        # RL模式下停滞或下降时，即使相似也保留，增加探索
                        logger.debug(f"RL探索模式：保留相似候选以增加多样性（停滞:{self.stagnation_count}, 下降:{self.decline_count}）")
                        candidates.append(candidate)
            except Exception as e:
                logger.warning(f"生成候选代码 {i+1} 失败: {e}")
        
        logger.info(f"生成了 {len(candidates)} 个有效候选代码")
        return candidates

    def _build_dynamic_constraints(self, iteration: int, current_score: float, current_metrics: Dict) -> str:
        """基于当前状态和RL策略构建动态约束提示"""
        base_constraints = (
            "要求：1) 严格保留顶层模块的接口(模块名、端口列表、位宽、方向、参数)完全一致；"
            "2) 仅修改模块体内实现，不改变接口和时序语义；"
            "3) 只输出优化后的 Verilog 代码(包含完整 module...endmodule)，不要额外说明。"
        )
        
        if not self.rl_mode or not current_metrics:
            return base_constraints
        
        # 基于当前指标动态添加优化方向提示
        dynamic_hints = []
        
        if self.exploration_factor > 0.5:
            # 高探索模式：鼓励大胆优化
            dynamic_hints.append("4) 可尝试较大幅度的逻辑重构、资源共享、时序优化等；")
        else:
            # 利用模式：保守优化
            dynamic_hints.append("4) 进行保守的局部优化，如常量折叠、冗余消除、表达式简化等；")
        
        # 基于当前瓶颈和RL状态指导优化重点
        if current_metrics:
            area = current_metrics.get('area', 0)
            ff_count = current_metrics.get('num_ff', 0)
            depth = current_metrics.get('logic_depth', 0)
            
            # 根据RL状态调整优化激进程度
            if self.decline_count >= 2:
                dynamic_hints.append("5) 当前策略效果不佳，尝试完全不同的优化方向：重新组织逻辑结构、改变实现方式；")
            elif self.exploration_factor > 0.6:
                if area > ff_count * 100:
                    dynamic_hints.append("5) 激进面积优化：大幅重构逻辑、合并相似功能、使用更紧凑的编码方式；")
                elif ff_count > 10:
                    dynamic_hints.append("5) 激进寄存器优化：重新设计状态机、合并寄存器、使用移位寄存器；")
                elif depth > 20:
                    dynamic_hints.append("5) 激进时序优化：重新分层逻辑、引入流水线、并行化计算；")
                else:
                    dynamic_hints.append("5) 探索性优化：尝试不同的实现范式、编码风格、结构组织；")
            else:
                if area > ff_count * 100:
                    dynamic_hints.append("5) 保守面积优化：逻辑简化、常量折叠、冗余消除；")
                elif ff_count > 10:
                    dynamic_hints.append("5) 保守寄存器优化：局部状态合并、寄存器复用；")
                elif depth > 20:
                    dynamic_hints.append("5) 保守时序优化：表达式分解、关键路径缓解；")
                else:
                    dynamic_hints.append("5) 渐进式优化：小幅改进表达式、优化信号命名、代码整理；")
        
        return base_constraints + "".join(dynamic_hints)
    
    def _generate_single_candidate(self, prompt: str, iteration: int, cand_idx: int, 
                                 current_score: float = 0, current_metrics: Dict = None) -> Optional[str]:
        """生成单个候选代码"""
        # 1) 构建输入（支持 chat 模板，如 Qwen 系列；若未配置模板则回退纯文本）
        use_chat = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)
        # 智能token分配：支持更大输入，动态调整生成长度
        context_max = self._get_context_window()
        estimated_prompt_tokens = len(prompt) // 3  # 更准确的估算（中文字符密度更高）
        
        # 根据输入长度动态调整生成tokens
        if estimated_prompt_tokens < 2000:
            # 短输入：保证充足生成空间
            effective_new = self.max_new_tokens
            input_max = context_max - effective_new - 100
        elif estimated_prompt_tokens < 4000:
            # 中等输入：平衡输入和生成
            effective_new = max(1024, int(self.max_new_tokens * 0.8))
            input_max = context_max - effective_new - 100
        else:
            # 长输入：优先保证输入完整性，但保证最小生成长度
            min_generation = 768  # 最小生成长度
            effective_new = max(min_generation, context_max - estimated_prompt_tokens - 200)
            input_max = context_max - effective_new - 100
        
        if estimated_prompt_tokens > input_max:
            logger.warning(f"输入过长（估算{estimated_prompt_tokens}tokens），可能影响生成质量")
        if self.debug_gen:
            logger.debug(f"context_max={context_max}, input_max={input_max}, effective_new={effective_new}, prompt_est={estimated_prompt_tokens}")
        if use_chat:
            try:
                messages = [
                    {"role": "system", "content": "你是精通数字电路和Verilog的资深芯片工程师，目标是优化面积/触发器数/逻辑深度。"},
                    {"role": "user", "content": prompt},
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=input_max)
            except Exception:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_max)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_max)

        input_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 2) 生成，仅解码新增 tokens，避免用字符串切片错位
        # 动态调整生成参数
        dynamic_top_p = max(0.7, min(0.95, self.base_top_p + (self.exploration_factor - 0.3) * 0.2))
        dynamic_top_k = max(30, min(80, int(self.base_top_k + (self.exploration_factor - 0.3) * 50)))
        dynamic_rep_penalty = max(1.0, min(1.15, self.base_rep_penalty + (self.exploration_factor - 0.3) * 0.1))
        
        if self.debug_gen:
            logger.debug(f"生成参数: temp={self.temperature:.2f}, top_p={dynamic_top_p:.2f}, top_k={dynamic_top_k}, rep_penalty={dynamic_rep_penalty:.2f}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=effective_new,
                temperature=self.temperature,
                do_sample=True,
                top_p=dynamic_top_p,
                top_k=dynamic_top_k,
                repetition_penalty=dynamic_rep_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        if new_tokens.numel() == 0:
            return None
        generated_tail = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self._maybe_dump_text(iteration, cand_idx, "generated_tail.txt", generated_tail)
        # 也保存 prompt，便于对照
        self._maybe_dump_text(iteration, cand_idx, "prompt.txt", prompt)

        # 3) 提取生成的代码部分（优先代码块，其次 module..endmodule）
        method = "raw"
        if "```" in generated_tail:
            method = "fenced"
        elif ("module" in generated_tail) and ("endmodule" in generated_tail):
            method = "module"
        code_part = self._extract_code_block(generated_tail).strip()
        if self.debug_gen:
            logger.debug(f"提取方法={method}, 提取后长度={len(code_part)}")
        self._maybe_dump_text(iteration, cand_idx, f"extracted_{method}.v", code_part)
        if not code_part:
            # 兜底：若模型把“优化后的代码：”重复了，进一步切分
            if "优化后的代码：" in generated_tail:
                code_part = generated_tail.split("优化后的代码：")[-1].strip()

        return code_part.strip() or None
    
    def _evaluate_code(self, code: str) -> Dict[str, Any]:
        """评估代码质量"""
        module = self._detect_top_module(code) or "top"
        return analyze_verilog_api(code, module)
    
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

    # === 辅助方法 ===
    def _has_complete_module(self, text: str) -> bool:
        return ("module" in text) and ("endmodule" in text)

    def _try_fix_module_name(self, code: str, module_name: str) -> str:
        """若候选的顶层 module 名称与期望不同，则仅替换第一个 module 声明的名称为期望名。支持可选参数块 #(...)."""
        pattern = re.compile(r"(\bmodule\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*(?:#\s*\([\s\S]*?\))?\s*\()", flags=re.IGNORECASE)
        def _repl(m):
            return f"{m.group(1)}{module_name}{m.group(3)}"
        return pattern.sub(_repl, code, count=1)
    
    def _extract_module_name(self, code: str) -> Optional[str]:
        """从Verilog代码中提取模块名"""
        import re
        pattern = re.compile(r'\bmodule\s+(\w+)', re.IGNORECASE)
        match = pattern.search(code)
        if match:
            return match.group(1)
        return None
    
    def _extract_module_ports(self, code: str, module_name: str) -> Optional[str]:
        """提取 module 头部括号内的端口列表文本。兼容可选参数块。"""
        pat = re.compile(rf"\bmodule\s+{re.escape(module_name)}\s*(?:#\s*\([\s\S]*?\))?\s*\(([\s\S]*?)\)\s*;", flags=re.IGNORECASE)
        m = pat.search(code)
        if m:
            return m.group(1)
        return None

    def _extract_module_params(self, code: str, module_name: str) -> Optional[str]:
        """提取 module 名后的参数块 #( ... ) 文本（包含括号），若无返回 None。"""
        pat = re.compile(rf"\bmodule\s+{re.escape(module_name)}\s*(#\s*\([\s\S]*?\))\s*\(", flags=re.IGNORECASE)
        m = pat.search(code)
        if m:
            return m.group(1)
        return None

    def _align_module_header(self, base_code: str, cand_code: str, module_name: str) -> str:
        """将候选的顶层 module 头部的参数块与端口列表替换为基准代码对应部分，以最大化接口一致性。"""
        base_ports = self._extract_module_ports(base_code, module_name)
        base_params = self._extract_module_params(base_code, module_name)
        if not base_ports:
            return cand_code
        # 组装替换：module name [params] (ports);
        def _repl(m):
            pre = m.group(1)  # 'module name'
            # 若基准有参数块则使用之，否则保持候选的（m.group(2)包含候选的可选参数块）
            cand_params = m.group(2) or ""
            params_use = base_params if base_params is not None else cand_params
            return f"{pre}{params_use}({base_ports})" + m.group(4)
        pat = re.compile(rf"(\bmodule\s+{re.escape(module_name)}\s*)(#\s*\([\s\S]*?\))?\s*\([\s\S]*?\)(\s*;)", flags=re.IGNORECASE)
        # 由于分组调整，修正规则为：pre (group1), params? (group2), we need suffix ; (group3)
        # 使用更明确的替换重写：
        def _repl2(m):
            pre = m.group(1)
            params_use = base_params if base_params is not None else (m.group(2) or "")
            suffix = m.group(3)
            return f"{pre}{params_use}({base_ports}){suffix}"
        return pat.sub(_repl2, cand_code, count=1)

    def _get_context_window(self) -> int:
        """获取模型上下文窗口大小"""
        try:
            m = int(getattr(self.tokenizer, "model_max_length", 4096) or 4096)
        except Exception:
            m = 4096
        # 过滤异常大值
        if m >= 1_000_000:
            m = 8192  # 提高默认上限
        # 设定全局上限，支持更长上下文
        m = max(2048, min(m, 32768))  # 支持到32K上下文
        return m

    # === 调试输出 ===
    def _candidate_dir(self, iteration: int, cand_idx: int) -> Optional[str]:
        if not self.debug_gen or not self.debug_dir:
            return None
        d = Path(self.debug_dir) / f"iter_{iteration:02d}" / f"cand_{cand_idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def _maybe_dump_text(self, iteration: int, cand_idx: int, name: str, content: str):
        try:
            d = self._candidate_dir(iteration, cand_idx)
            if not d:
                return
            p = Path(d) / name
            with open(p, 'w', encoding='utf-8') as f:
                f.write(content or "")
        except Exception:
            pass
    def _detect_top_module(self, code: str) -> Optional[str]:
        """从 Verilog 源码中自动检测顶层模块名。
        策略：
        1) 去除注释后，匹配第一个 module 声明；
        2) 若存在名为 top 的模块，则优先返回 top；
        3) 否则返回第一个模块名；
        4) 若未找到返回 None。
        """
        try:
            text = re.sub(r"/\*[\s\S]*?\*/", "", code)  # 块注释
            text = re.sub(r"//.*", "", text)               # 行注释
            names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text)
            if not names:
                return None
            # 优先 top
            for n in names:
                if n == "top" or n.lower() == "top":
                    return n
            return names[0]
        except Exception:
            return None
    def _extract_code_block(self, text: str) -> str:
        """从生成文本中抽取 Verilog 代码块。
        优先匹配 ```verilog / ```systemverilog 代码块；否则回退匹配 module...endmodule。"""
        # 1) 三引号代码块
        fence_matches = list(re.finditer(r"```(?:verilog|systemverilog)?\s*\n([\s\S]*?)\n```", text, flags=re.IGNORECASE))
        if fence_matches:
            return fence_matches[-1].group(1).strip()

        # 2) module..endmodule（非贪婪）
        mod = re.search(r"\bmodule\b[\s\S]*?\bendmodule\b", text, flags=re.IGNORECASE)
        if mod:
            return mod.group(0).strip()

        # 3) 直接返回原文（可能模型已只输出代码）
        return text.strip()

    def _normalize_code(self, code: str) -> str:
        """对 Verilog 代码做轻度归一化：去注释、压缩多空白。"""
        # 去掉 // 行注释
        code = re.sub(r"//.*", "", code)
        # 去掉 /* */ 块注释
        code = re.sub(r"/\*[\s\S]*?\*/", "", code)
        # 压缩空白
        code = re.sub(r"\s+", " ", code).strip()
        return code

    def _is_meaningfully_different(self, base: str, cand: str, threshold: float = 0.98) -> bool:
        """判断候选是否与基准存在“有意义差异”。
        使用归一化后的相似度，默认阈值为 0.98（越接近1越相似）。"""
        base_n = self._normalize_code(base)
        cand_n = self._normalize_code(cand)
        if not cand_n:
            return False
        # 完全相等直接拒绝
        if base_n == cand_n:
            return False
        sim = SequenceMatcher(a=base_n, b=cand_n).ratio()
        return sim < threshold


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
