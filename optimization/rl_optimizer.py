"""
Module 3: Agent-driven optimization loop for Verilog.

The optimizer uses an LLM to generate candidate HDL, calls external EDA tools to
measure objective metrics, and then lets the model act as an agent that decides
whether the loop should continue based on the observed results.
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core_tools.eda_wrapper import EDAWrapper, analyze_verilog_api
from optimization.competition_engine import (
    build_target_profile,
    generate_ai_insights,
    generate_competition_package,
)
from optimization.llm_client import build_llm_client


class RLOptimizer:
    """LLM + EDA closed-loop optimizer."""

    def __init__(
        self,
        model_path: str,
        max_iterations: int = 10,
        population_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        debug_gen: bool = False,
        debug_dir: Optional[str] = None,
        rl_mode: bool = True,
        base_top_p: float = 0.9,
        base_top_k: int = 50,
        base_rep_penalty: float = 1.05,
    ):
        self.model_path = model_path
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.rl_mode = rl_mode
        self.base_temperature = temperature
        self.base_top_p = base_top_p
        self.base_top_k = base_top_k
        self.base_rep_penalty = base_rep_penalty
        self.debug_gen = debug_gen

        if debug_dir is None and debug_gen:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_root = Path("outputs") / f"debug_{ts}"
            debug_root.mkdir(parents=True, exist_ok=True)
            self.debug_dir = str(debug_root)
        else:
            self.debug_dir = debug_dir

        self.eda = EDAWrapper()
        self.llm_client = build_llm_client(model_path)

        self.optimization_history: List[Dict[str, Any]] = []
        self.score_history: List[float] = []
        self.agent_decisions: List[Dict[str, Any]] = []

        self.improvement_streak = 0
        self.stagnation_count = 0
        self.decline_count = 0
        self.exploration_factor = 0.3
        self.last_best_score = 0.0
        self.score_trend = 0.0
        self.module_name: Optional[str] = None
        self.target_profile = build_target_profile("")

        logger.info(f"Optimizer initialized with client mode: {getattr(self.llm_client, 'mode', 'unknown')}")

    def optimize(self, input_code: str, target_description: str = "", scenario: str = "") -> Dict[str, Any]:
        self.target_profile = build_target_profile(target_description)
        self.module_name = self._extract_module_name(input_code)
        if not self.module_name:
            return {"success": False, "error": "Unable to extract module name from the input code."}

        logger.info("Starting optimization loop")
        try:
            initial_data = self.eda.synthesize(input_code, self.module_name, target_freq=100.0)
        except Exception as exc:
            return {"success": False, "error": f"Initial synthesis failed: {exc}"}

        if not initial_data.get("syntax_ok", False) or not initial_data.get("synth_ok", False):
            return {
                "success": False,
                "error": "Input Verilog contains syntax or synthesis errors and cannot be optimized.",
            }

        initial_score = self._calculate_score(initial_data)
        best_score = initial_score
        best_code = input_code
        best_metrics = initial_data
        self.last_best_score = initial_score
        self.score_history = []
        self.agent_decisions = []
        self.optimization_history = [
            {
                "iteration": 0,
                "code": input_code,
                "metrics": initial_data,
                "score": initial_score,
                "is_best": True,
            }
        ]

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Optimization iteration {iteration}/{self.max_iterations}")

            if self.rl_mode and iteration > 1:
                self._update_rl_strategy(best_score)

            candidates = self._generate_candidates(
                base_code=best_code,
                target_description=target_description,
                iteration=iteration,
                current_score=best_score,
                current_metrics=best_metrics,
            )

            iteration_best_score = best_score
            iteration_best_code = best_code
            iteration_best_metrics = best_metrics

            for candidate_index, candidate in enumerate(candidates, start=1):
                module_name = self._detect_top_module(best_code) or "top"
                if not self._has_complete_module(candidate):
                    self._maybe_dump_text(iteration, candidate_index, "candidate_incomplete.v", candidate)
                    continue

                candidate_named = self._try_fix_module_name(candidate, module_name)
                candidate_fixed = self._align_module_header(best_code, candidate_named, module_name)
                self._maybe_dump_text(iteration, candidate_index, "candidate_fixed.v", candidate_fixed)

                metrics_result = self._evaluate_code(candidate_fixed)
                if not metrics_result.get("success"):
                    self.optimization_history.append(
                        {
                            "iteration": iteration,
                            "candidate": candidate_index,
                            "code": candidate_fixed,
                            "score": None,
                            "error": metrics_result.get("error", "EDA failed"),
                            "is_best": False,
                        }
                    )
                    continue

                metrics = metrics_result["data"]
                score = self._calculate_score(metrics)
                self.optimization_history.append(
                    {
                        "iteration": iteration,
                        "candidate": candidate_index,
                        "code": candidate_fixed,
                        "metrics": metrics,
                        "score": score,
                        "is_best": False,
                    }
                )

                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_code = candidate_fixed
                    iteration_best_metrics = metrics
                    if self.debug_dir:
                        best_file = Path(self.debug_dir) / f"best_candidate_iter_{iteration}.v"
                        best_file.write_text(candidate_fixed, encoding="utf-8")

            self.score_history.append(iteration_best_score)

            if iteration_best_score > best_score:
                best_score = iteration_best_score
                best_code = iteration_best_code
                best_metrics = iteration_best_metrics
                self.improvement_streak += 1
                self.stagnation_count = 0
            elif abs(iteration_best_score - best_score) < 1e-6:
                self.improvement_streak = 0
                self.stagnation_count += 1
            else:
                self.improvement_streak = 0
                self.stagnation_count += 1
                self.decline_count += 1

            decision = self._agent_decide_next_step(
                iteration=iteration,
                target_description=target_description,
                current_best_score=best_score,
                initial_metrics=initial_data,
                best_metrics=best_metrics,
                latest_iteration_score=iteration_best_score,
            )
            self.agent_decisions.append(decision)
            self.last_best_score = iteration_best_score

            if decision["action"] == "stop":
                logger.info(f"Agent decided to stop at iteration {iteration}: {decision['reason']}")
                break

        init_score = self._calculate_score(initial_data)
        improvement = {
            "area_improvement": (initial_data["area"] - best_metrics["area"]) / max(1e-9, initial_data["area"]) * 100,
            "ff_improvement": (
                (initial_data.get("num_ff", 0) - best_metrics.get("num_ff", 0))
                / max(1e-9, initial_data.get("num_ff", 0) or 1)
                * 100
            ),
            "depth_improvement": (
                (initial_data.get("logic_depth", 0) - best_metrics.get("logic_depth", 0))
                / max(1e-9, initial_data.get("logic_depth", 0) or 1)
                * 100
            ),
            "score_improvement": (best_score - init_score) / max(1e-9, init_score or 1) * 100,
        }

        result: Dict[str, Any] = {
            "success": True,
            "original_code": input_code,
            "optimized_code": best_code,
            "original_metrics": initial_data,
            "optimized_metrics": best_metrics,
            "improvement": improvement,
            "optimization_history": self.optimization_history,
            "agent_decisions": self.agent_decisions,
            "total_iterations": len(self.score_history),
            "total_candidates": max(0, len(self.optimization_history) - 1),
            "target_profile": self.target_profile,
            "llm_mode": getattr(self.llm_client, "mode", "unknown"),
        }

        result["ai_insights"] = generate_ai_insights(
            original_code=input_code,
            optimized_code=best_code,
            original_metrics=initial_data,
            optimized_metrics=best_metrics,
            profile=self.target_profile,
            target_description=target_description,
            scenario=scenario,
        )
        result["competition_package"] = generate_competition_package(
            result=result,
            profile=self.target_profile,
            target_description=target_description,
            scenario=scenario,
        )
        return result

    def _update_rl_strategy(self, current_best_score: float) -> None:
        recent_scores = self.score_history[-5:]
        short_trend = 0.0
        long_trend = 0.0
        if len(recent_scores) >= 2:
            short_trend = (recent_scores[-1] - recent_scores[-2]) / max(abs(recent_scores[-2]), 1e-6)
        if len(recent_scores) >= 3:
            long_trend = (recent_scores[-1] - recent_scores[0]) / max(abs(recent_scores[0]), 1e-6)
        self.score_trend = short_trend

        if short_trend < -0.01:
            self.decline_count += 1
        else:
            self.decline_count = 0

        if self.decline_count >= 2:
            self.exploration_factor = min(0.9, self.exploration_factor + 0.15)
            self.temperature = min(1.3, self.base_temperature + 0.3)
        elif self.stagnation_count >= 2:
            self.exploration_factor = min(0.7, self.exploration_factor + 0.1)
            self.temperature = min(1.1, self.base_temperature + 0.2)
        elif self.improvement_streak >= 3:
            self.exploration_factor = max(0.1, self.exploration_factor - 0.05)
            self.temperature = max(0.6, self.base_temperature - 0.1)
        elif short_trend > 0.02:
            self.exploration_factor = max(0.2, self.exploration_factor - 0.03)
            self.temperature = max(0.7, self.base_temperature - 0.05)
        else:
            self.exploration_factor += (0.3 - self.exploration_factor) * 0.3
            self.temperature += (self.base_temperature - self.temperature) * 0.2

        logger.info(
            f"Strategy updated: temperature={self.temperature:.2f}, exploration={self.exploration_factor:.2f}, "
            f"short_trend={short_trend:.4f}, long_trend={long_trend:.4f}, score={current_best_score:.4f}"
        )

    def _generate_candidates(
        self,
        base_code: str,
        target_description: str,
        iteration: int,
        current_score: float = 0,
        current_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        constraints = self._build_dynamic_constraints(iteration, current_score, current_metrics or {})
        profile_hint = self.target_profile.get("prompt_focus", "")
        if target_description:
            prompt = (
                f"请优化以下 Verilog 代码，目标是：{target_description}。"
                f"{profile_hint} {constraints}\n\n原始代码：\n{base_code}\n\n优化后的代码："
            )
        else:
            prompt = (
                f"请优化以下 Verilog 代码，使其在资源和时序上更优。"
                f"{profile_hint} {constraints}\n\n原始代码：\n{base_code}\n\n优化后的代码："
            )

        candidates: List[str] = []
        for i in range(self.population_size):
            try:
                candidate = self._generate_single_candidate(prompt, iteration, i + 1, current_score, current_metrics or {})
                if not candidate:
                    continue
                if self._is_meaningfully_different(base_code, candidate):
                    candidates.append(candidate)
                elif self.rl_mode and (self.stagnation_count >= 2 or self.decline_count >= 1):
                    candidates.append(candidate)
            except Exception as exc:
                logger.warning(f"Candidate generation failed for #{i + 1}: {exc}")
        logger.info(f"Generated {len(candidates)} valid candidates")
        return candidates

    def _build_dynamic_constraints(self, iteration: int, current_score: float, current_metrics: Dict[str, Any]) -> str:
        base_constraints = (
            "要求：1) 必须严格保持顶层模块名、端口列表、位宽和方向一致；"
            "2) 只修改模块内部实现，不改变功能语义；"
            "3) 仅输出完整 Verilog 代码，不要输出解释。"
        )
        if not self.rl_mode or not current_metrics:
            return base_constraints

        hints: List[str] = []
        if self.exploration_factor > 0.5:
            hints.append("4) 可以尝试更激进的逻辑重构、资源共享或关键路径重写；")
        else:
            hints.append("4) 优先做保守优化，例如常量折叠、冗余消除和表达式简化；")

        area = current_metrics.get("area", 0)
        ff_count = current_metrics.get("num_ff", 0)
        depth = current_metrics.get("logic_depth", 0)

        if self.decline_count >= 2:
            hints.append("5) 当前方向效果不佳，请尝试与前几轮显著不同的实现思路；")
        elif area > ff_count * 100:
            hints.append("5) 当前面积是主要瓶颈，请优先减少重复逻辑和中间信号。")
        elif ff_count > 10:
            hints.append("5) 当前寄存器数量偏多，请优先考虑寄存器复用和状态压缩。")
        elif depth > 20:
            hints.append("5) 当前逻辑深度偏高，请优先缩短关键路径。")
        else:
            hints.append(f"5) 当前得分为 {current_score:.4f}，请做小步但有效的结构优化。")

        hints.append(f"6) 当前为第 {iteration} 轮优化。")
        return base_constraints + "".join(hints)

    def _generate_single_candidate(
        self,
        prompt: str,
        iteration: int,
        cand_idx: int,
        current_score: float = 0,
        current_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        dynamic_top_p = max(0.7, min(0.95, self.base_top_p + (self.exploration_factor - 0.3) * 0.2))
        dynamic_top_k = max(30, min(80, int(self.base_top_k + (self.exploration_factor - 0.3) * 50)))
        dynamic_rep_penalty = max(1.0, min(1.15, self.base_rep_penalty + (self.exploration_factor - 0.3) * 0.1))

        generated_tail = self.llm_client.generate(
            prompt,
            system_prompt="你是面向 Verilog 代码优化任务的 AI Agent，需要结合外部 EDA 指标持续改进代码。",
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=dynamic_top_p,
            top_k=dynamic_top_k,
            repetition_penalty=dynamic_rep_penalty,
        )
        if not generated_tail:
            return None

        self._maybe_dump_text(iteration, cand_idx, "generated_tail.txt", generated_tail)
        self._maybe_dump_text(iteration, cand_idx, "prompt.txt", prompt)

        method = "raw"
        if "```" in generated_tail:
            method = "fenced"
        elif ("module" in generated_tail) and ("endmodule" in generated_tail):
            method = "module"

        code_part = self._extract_code_block(generated_tail).strip()
        self._maybe_dump_text(iteration, cand_idx, f"extracted_{method}.v", code_part)

        if not code_part and "优化后的代码" in generated_tail:
            code_part = generated_tail.split("优化后的代码")[-1].strip(":： \n")

        return code_part.strip() or None

    def _evaluate_code(self, code: str) -> Dict[str, Any]:
        module = self._detect_top_module(code) or "top"
        return analyze_verilog_api(code, module)

    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        weights = self.target_profile.get("weights", {})
        area_weight = float(weights.get("area", 0.45))
        ff_weight = float(weights.get("ff", 0.35))
        depth_weight = float(weights.get("depth", 0.20))
        pass_bonus_weight = float(weights.get("pass_bonus", 0.10))

        pass_bonus = 0.0
        if metrics.get("syntax_ok"):
            pass_bonus += 1.0
        if metrics.get("synth_ok"):
            pass_bonus += 1.0
        if metrics.get("equiv_ok"):
            pass_bonus += 1.0

        area = max(0.0, float(metrics.get("area", 0)))
        ff = max(0.0, float(metrics.get("num_ff", 0)))
        depth = max(0.0, float(metrics.get("logic_depth", 0)))

        area_score = 1.0 / (1.0 + area / 1000.0)
        ff_score = 1.0 / (1.0 + ff / 1000.0)
        depth_score = 1.0 / (1.0 + depth / 10.0)

        return (
            area_weight * area_score
            + ff_weight * ff_score
            + depth_weight * depth_score
            + pass_bonus_weight * pass_bonus
        )

    def _agent_decide_next_step(
        self,
        iteration: int,
        target_description: str,
        current_best_score: float,
        initial_metrics: Dict[str, Any],
        best_metrics: Dict[str, Any],
        latest_iteration_score: float,
    ) -> Dict[str, Any]:
        payload = {
            "iteration": iteration,
            "target": target_description or self.target_profile.get("label", "balanced"),
            "initial_metrics": initial_metrics,
            "best_metrics": best_metrics,
            "current_best_score": round(current_best_score, 6),
            "latest_iteration_score": round(latest_iteration_score, 6),
            "stagnation_count": self.stagnation_count,
            "decline_count": self.decline_count,
            "improvement_streak": self.improvement_streak,
        }

        prompt = (
            "你是优化控制智能体。请根据如下 EDA 结果判断是否继续优化。\n"
            "只输出 JSON，格式为 {\"action\":\"continue|stop\",\"reason\":\"...\",\"focus\":\"...\"}。\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        fallback = self._heuristic_agent_decision(payload)
        try:
            raw_text = self.llm_client.generate(
                prompt,
                system_prompt="你负责决定 Verilog 优化流程是否继续执行，必须基于客观 EDA 指标作出判断。",
                temperature=0.2,
                max_new_tokens=160,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.0,
            )
            parsed = self._parse_agent_json(raw_text)
            if parsed is None:
                parsed = fallback
                parsed["reason"] = f"{fallback['reason']} 模型原始输出解析失败。"
                parsed["raw_model_output"] = raw_text
            else:
                parsed["raw_model_output"] = raw_text
            return parsed
        except Exception as exc:
            fallback["reason"] = f"{fallback['reason']} Agent decision fallback due to error: {exc}"
            return fallback

    def _heuristic_agent_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        area_improved = payload["initial_metrics"].get("area", 0) - payload["best_metrics"].get("area", 0)
        depth_improved = payload["initial_metrics"].get("logic_depth", 0) - payload["best_metrics"].get("logic_depth", 0)

        if payload["stagnation_count"] >= 3:
            return {
                "action": "stop",
                "reason": "Multiple stagnant rounds detected.",
                "focus": "Summarize current best design and stop.",
            }
        if payload["iteration"] >= self.max_iterations:
            return {
                "action": "stop",
                "reason": "Maximum iteration budget reached.",
                "focus": "Output current best design.",
            }
        if area_improved <= 0 and depth_improved <= 0 and payload["iteration"] >= 2:
            return {
                "action": "continue",
                "reason": "No clear metric gain yet, continue searching with a different strategy.",
                "focus": "Increase exploration and change rewrite style.",
            }
        return {
            "action": "continue",
            "reason": "Current direction still has optimization potential.",
            "focus": "Preserve improved structure and continue refining.",
        }

    def _parse_agent_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        text = raw_text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("action") in {"continue", "stop"}:
                return data
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and data.get("action") in {"continue", "stop"}:
                return data
        except Exception:
            return None
        return None

    def save_optimization_report(self, result: Dict[str, Any], output_path: str) -> None:
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "optimization_config": {
                "max_iterations": self.max_iterations,
                "population_size": self.population_size,
                "temperature": self.temperature,
                "llm_mode": getattr(self.llm_client, "mode", "unknown"),
            },
            "result": result,
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        logger.info(f"Optimization report saved to {output_path}")

    def cleanup(self) -> None:
        if hasattr(self, "eda"):
            self.eda.cleanup()
        if hasattr(self, "llm_client"):
            self.llm_client.cleanup()

    def _has_complete_module(self, text: str) -> bool:
        return ("module" in text) and ("endmodule" in text)

    def _try_fix_module_name(self, code: str, module_name: str) -> str:
        pattern = re.compile(
            r"(\bmodule\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*(?:#\s*\([\s\S]*?\))?\s*\()",
            flags=re.IGNORECASE,
        )

        def repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{module_name}{match.group(3)}"

        return pattern.sub(repl, code, count=1)

    def _extract_module_name(self, code: str) -> Optional[str]:
        match = re.search(r"\bmodule\s+(\w+)", code, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_module_ports(self, code: str, module_name: str) -> Optional[str]:
        pattern = re.compile(
            rf"\bmodule\s+{re.escape(module_name)}\s*(?:#\s*\([\s\S]*?\))?\s*\(([\s\S]*?)\)\s*;",
            flags=re.IGNORECASE,
        )
        match = pattern.search(code)
        if match:
            return match.group(1)
        return None

    def _extract_module_params(self, code: str, module_name: str) -> Optional[str]:
        pattern = re.compile(
            rf"\bmodule\s+{re.escape(module_name)}\s*(#\s*\([\s\S]*?\))\s*\(",
            flags=re.IGNORECASE,
        )
        match = pattern.search(code)
        if match:
            return match.group(1)
        return None

    def _align_module_header(self, base_code: str, cand_code: str, module_name: str) -> str:
        base_ports = self._extract_module_ports(base_code, module_name)
        base_params = self._extract_module_params(base_code, module_name)
        if not base_ports:
            return cand_code

        pattern = re.compile(
            rf"(\bmodule\s+{re.escape(module_name)}\s*)(#\s*\([\s\S]*?\))?\s*\([\s\S]*?\)(\s*;)",
            flags=re.IGNORECASE,
        )

        def repl(match: re.Match[str]) -> str:
            prefix = match.group(1)
            params = base_params if base_params is not None else (match.group(2) or "")
            suffix = match.group(3)
            return f"{prefix}{params}({base_ports}){suffix}"

        return pattern.sub(repl, cand_code, count=1)

    def _candidate_dir(self, iteration: int, cand_idx: int) -> Optional[str]:
        if not self.debug_gen or not self.debug_dir:
            return None
        path = Path(self.debug_dir) / f"iter_{iteration:02d}" / f"cand_{cand_idx:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _maybe_dump_text(self, iteration: int, cand_idx: int, name: str, content: str) -> None:
        try:
            directory = self._candidate_dir(iteration, cand_idx)
            if not directory:
                return
            file_path = Path(directory) / name
            file_path.write_text(content or "", encoding="utf-8")
        except Exception:
            return

    def _detect_top_module(self, code: str) -> Optional[str]:
        try:
            text = re.sub(r"/\*[\s\S]*?\*/", "", code)
            text = re.sub(r"//.*", "", text)
            names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text)
            if not names:
                return None
            for name in names:
                if name.lower() == "top":
                    return name
            return names[0]
        except Exception:
            return None

    def _extract_code_block(self, text: str) -> str:
        fence_matches = list(
            re.finditer(r"```(?:verilog|systemverilog)?\s*\n([\s\S]*?)\n```", text, flags=re.IGNORECASE)
        )
        if fence_matches:
            return fence_matches[-1].group(1).strip()

        module_match = re.search(r"\bmodule\b[\s\S]*?\bendmodule\b", text, flags=re.IGNORECASE)
        if module_match:
            return module_match.group(0).strip()
        return text.strip()

    def _normalize_code(self, code: str) -> str:
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"/\*[\s\S]*?\*/", "", code)
        code = re.sub(r"\s+", " ", code).strip()
        return code

    def _is_meaningfully_different(self, base: str, cand: str, threshold: float = 0.98) -> bool:
        base_norm = self._normalize_code(base)
        cand_norm = self._normalize_code(cand)
        if not cand_norm:
            return False
        if base_norm == cand_norm:
            return False
        similarity = SequenceMatcher(a=base_norm, b=cand_norm).ratio()
        return similarity < threshold


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agent-driven Verilog optimizer")
    parser.add_argument("--model", required=True, help="Path to local model or adapter directory")
    parser.add_argument("--input", "-i", help="Input Verilog file")
    parser.add_argument("--code", "-c", help="Inline Verilog code")
    parser.add_argument("--target", "-t", default="", help="Optimization target description")
    parser.add_argument("--scenario", default="", help="Competition scenario description")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum optimization iterations")
    parser.add_argument("--population", type=int, default=5, help="Candidate count per iteration")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--output", "-o", help="Output directory")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            input_code = handle.read()
    elif args.code:
        input_code = args.code
    else:
        print("Please provide --input or --code")
        return

    optimizer = RLOptimizer(
        model_path=args.model,
        max_iterations=args.iterations,
        population_size=args.population,
        temperature=args.temperature,
    )

    try:
        result = optimizer.optimize(input_code, args.target, args.scenario)
        if result["success"]:
            print("=== Optimization Finished ===")
            print(f"Area improvement: {result['improvement']['area_improvement']:.2f}%")
            print(f"FF improvement: {result['improvement']['ff_improvement']:.2f}%")
            print(f"Depth improvement: {result['improvement']['depth_improvement']:.2f}%")
            print(f"Score improvement: {result['improvement']['score_improvement']:.2f}%")
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(exist_ok=True)
                (output_dir / "optimized_code.v").write_text(result["optimized_code"], encoding="utf-8")
                optimizer.save_optimization_report(result, str(output_dir / "optimization_report.json"))
                if result.get("competition_package", {}).get("markdown_report"):
                    (output_dir / "competition_summary.md").write_text(
                        result["competition_package"]["markdown_report"],
                        encoding="utf-8",
                    )
                print(f"Saved to: {args.output}")
        else:
            print(f"Optimization failed: {result['error']}")
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()
