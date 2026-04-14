"""
Module 3: Agent-driven optimization loop for Verilog.

Architecture — four explicit roles:
  Planner   → analyzes code structure, estimates complexity, sets strategy
  Coder     → generates optimized Verilog candidates via LLM
  Evaluator → calls Yosys for objective EDA metrics
  Judge     → decides continue / stop based on metrics & history

AgentOptimizer is the orchestrator that wires the four roles into a closed
loop and reports progress through an optional callback.
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from core_tools.eda_wrapper import EDAWrapper, analyze_verilog_api
from optimization.competition_engine import (
    build_target_profile,
    generate_ai_insights,
    generate_competition_package,
    summarize_code_features,
)
from optimization.llm_client import BaseLLMClient, build_llm_client

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Analyzes code structure, estimates task complexity, and sets the
    optimization strategy for the Coder."""

    def analyze(
        self,
        code: str,
        metrics: Dict[str, Any],
        target_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create the initial optimization plan."""
        features = summarize_code_features(code)
        complexity = self._estimate_complexity(features, metrics)
        bottlenecks = self._identify_bottlenecks(metrics, features, target_profile)
        strategy = self._initial_strategy(bottlenecks, target_profile)

        return {
            "complexity": complexity,
            "features": features,
            "bottlenecks": bottlenecks,
            "strategy": strategy,
            # Dynamic budgets — simple tasks finish fast, complex ones get room
            "min_iterations": {"simple": 2, "medium": 2, "complex": 3}[complexity],
            "stagnation_patience": {"simple": 2, "medium": 3, "complex": 4}[complexity],
            "suggested_max_iterations": {"simple": 5, "medium": 8, "complex": 10}[complexity],
        }

    def refine_strategy(
        self,
        plan: Dict[str, Any],
        score_history: List[float],
        stagnation_count: int,
        decline_count: int,
    ) -> str:
        """Return an updated strategy key after each iteration."""
        if decline_count >= 2:
            return "aggressive_restructure"
        if stagnation_count >= 2:
            return "explore_alternative"
        if len(score_history) >= 2 and score_history[-1] > score_history[-2]:
            return "exploit_current"
        return plan.get("strategy", "balanced")

    # -- internals ----------------------------------------------------------

    def _estimate_complexity(
        self, features: Dict[str, Any], metrics: Dict[str, Any],
    ) -> str:
        lines = features.get("line_count", 0)
        always = features.get("always_count", 0)
        modules = features.get("module_count", 1)
        if lines <= 30 and always <= 2 and modules <= 1:
            return "simple"
        if lines <= 80 and always <= 5:
            return "medium"
        return "complex"

    def _identify_bottlenecks(
        self,
        metrics: Dict[str, Any],
        features: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> List[str]:
        bottlenecks: List[str] = []
        area = float(metrics.get("area", 0) or 0)
        ff = float(metrics.get("num_ff", 0) or 0)
        depth = float(metrics.get("logic_depth", 0) or 0)
        weights = profile.get("weights", {})

        if weights.get("area", 0) >= 0.4 and area > 50:
            bottlenecks.append("area")
        if weights.get("ff", 0) >= 0.3 and ff > 8:
            bottlenecks.append("registers")
        if weights.get("depth", 0) >= 0.3 and depth > 8:
            bottlenecks.append("logic_depth")
        return bottlenecks or ["general"]

    def _initial_strategy(
        self, bottlenecks: List[str], profile: Dict[str, Any],
    ) -> str:
        if "area" in bottlenecks:
            return "reduce_area"
        if "registers" in bottlenecks:
            return "reduce_registers"
        if "logic_depth" in bottlenecks:
            return "reduce_depth"
        return "balanced"


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

class Coder:
    """Generates optimized Verilog candidates via the LLM."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        target_profile: Dict[str, Any],
        max_new_tokens: int = 1024,
        base_temperature: float = 0.8,
        base_top_p: float = 0.9,
        base_top_k: int = 50,
        base_rep_penalty: float = 1.05,
        debug_gen: bool = False,
        debug_dir: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.target_profile = target_profile
        self.max_new_tokens = max_new_tokens
        self.base_temperature = base_temperature
        self.temperature = base_temperature
        self.base_top_p = base_top_p
        self.base_top_k = base_top_k
        self.base_rep_penalty = base_rep_penalty
        self.exploration_factor = 0.3
        self.debug_gen = debug_gen
        self.debug_dir = debug_dir

    # -- public API ---------------------------------------------------------

    def generate_candidates(
        self,
        base_code: str,
        plan: Dict[str, Any],
        iteration: int,
        current_score: float,
        current_metrics: Dict[str, Any],
        target_description: str,
        population_size: int = 1,
    ) -> List[str]:
        prompt = self._build_prompt(
            base_code, plan, iteration, current_score, current_metrics, target_description,
        )

        candidates: List[str] = []
        for i in range(population_size):
            try:
                raw = self._generate_single(prompt, iteration, i + 1)
                if not raw:
                    continue
                if self._is_meaningfully_different(base_code, raw):
                    candidates.append(raw)
                elif plan.get("strategy") in ("aggressive_restructure", "explore_alternative"):
                    candidates.append(raw)
            except Exception as exc:
                logger.warning(f"Candidate generation failed #{i + 1}: {exc}")

        logger.info(f"Generated {len(candidates)} valid candidates")
        return candidates

    def update_sampling(self, strategy: str, exploration_factor: float) -> None:
        """Adjust sampling hyper-parameters based on the current strategy."""
        self.exploration_factor = exploration_factor
        if strategy == "aggressive_restructure":
            self.temperature = min(1.3, self.base_temperature + 0.3)
        elif strategy == "explore_alternative":
            self.temperature = min(1.1, self.base_temperature + 0.2)
        elif strategy == "exploit_current":
            self.temperature = max(0.6, self.base_temperature - 0.1)
        else:
            # Gently decay toward base
            self.temperature += (self.base_temperature - self.temperature) * 0.3

    # -- prompt construction ------------------------------------------------

    def _build_prompt(
        self,
        base_code: str,
        plan: Dict[str, Any],
        iteration: int,
        score: float,
        metrics: Dict[str, Any],
        target_description: str,
    ) -> str:
        constraints = self._build_constraints(plan, iteration, score, metrics)
        profile_hint = self.target_profile.get("prompt_focus", "")
        if target_description:
            return (
                f"请优化以下 Verilog 代码，目标是：{target_description}。"
                f"{profile_hint} {constraints}\n\n原始代码：\n{base_code}\n\n优化后的代码："
            )
        return (
            f"请优化以下 Verilog 代码，使其在资源和时序上更优。"
            f"{profile_hint} {constraints}\n\n原始代码：\n{base_code}\n\n优化后的代码："
        )

    def _build_constraints(
        self, plan: Dict[str, Any], iteration: int, score: float, metrics: Dict[str, Any],
    ) -> str:
        base = (
            "要求：1) 必须严格保持顶层模块名、端口列表、位宽和方向一致；"
            "2) 只修改模块内部实现，不改变功能语义；"
            "3) 仅输出完整 Verilog 代码，不要输出解释。"
        )
        hints: List[str] = []
        strategy = plan.get("strategy", "balanced")
        if strategy == "aggressive_restructure":
            hints.append("4) 请尝试完全不同的实现思路，如重构状态机、资源共享或关键路径重写；")
        elif strategy == "explore_alternative":
            hints.append("4) 前几轮优化效果有限，请换一个角度尝试优化；")
        elif strategy == "exploit_current":
            hints.append("4) 当前方向有效，继续做保守的细粒度优化；")
        else:
            hints.append("4) 优先做常量折叠、冗余消除和表达式简化等安全优化；")

        bottlenecks = plan.get("bottlenecks", [])
        if "area" in bottlenecks:
            hints.append("5) 面积是主要瓶颈，请优先减少重复逻辑和中间信号。")
        elif "registers" in bottlenecks:
            hints.append("5) 寄存器数量偏多，请优先考虑寄存器复用和状态压缩。")
        elif "logic_depth" in bottlenecks:
            hints.append("5) 逻辑深度偏高，请优先缩短关键路径。")
        else:
            hints.append(f"5) 当前得分 {score:.4f}，请做小步有效的结构优化。")

        hints.append(f"6) 当前第 {iteration} 轮优化。")
        return base + "".join(hints)

    # -- single generation --------------------------------------------------

    def _generate_single(
        self, prompt: str, iteration: int, cand_idx: int,
    ) -> Optional[str]:
        ef = self.exploration_factor
        dynamic_top_p = max(0.7, min(0.95, self.base_top_p + (ef - 0.3) * 0.2))
        dynamic_top_k = max(30, min(80, int(self.base_top_k + (ef - 0.3) * 50)))
        dynamic_rep = max(1.0, min(1.15, self.base_rep_penalty + (ef - 0.3) * 0.1))

        generated = self.llm_client.generate(
            prompt,
            system_prompt="你是面向 Verilog 代码优化任务的 AI Agent，需要结合外部 EDA 指标持续改进代码。",
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=dynamic_top_p,
            top_k=dynamic_top_k,
            repetition_penalty=dynamic_rep,
        )
        if not generated:
            return None

        self._dump(iteration, cand_idx, "generated_tail.txt", generated)
        self._dump(iteration, cand_idx, "prompt.txt", prompt)

        code = _extract_code_block(generated).strip()
        self._dump(iteration, cand_idx, "extracted.v", code)

        if not code and "优化后的代码" in generated:
            code = generated.split("优化后的代码")[-1].strip(":： \n")

        return code.strip() or None

    # -- helpers ------------------------------------------------------------

    def _is_meaningfully_different(
        self, base: str, cand: str, threshold: float = 0.98,
    ) -> bool:
        bn = _normalize_code(base)
        cn = _normalize_code(cand)
        if not cn or bn == cn:
            return False
        return SequenceMatcher(a=bn, b=cn).ratio() < threshold

    def _dump(self, iteration: int, cand_idx: int, name: str, content: str) -> None:
        if not self.debug_gen or not self.debug_dir:
            return
        try:
            d = Path(self.debug_dir) / f"iter_{iteration:02d}" / f"cand_{cand_idx:02d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(content or "", encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Calls EDA tools (Yosys) to measure objective metrics."""

    def __init__(self, eda: EDAWrapper):
        self.eda = eda

    def evaluate(self, code: str, module_name: str) -> Dict[str, Any]:
        return analyze_verilog_api(code, module_name)

    def check_equivalence(
        self, original_code: str, candidate_code: str, module_name: str,
    ) -> bool:
        """Return True if candidate is functionally equivalent to original."""
        return self.eda.check_equivalence(original_code, candidate_code, module_name)

    def score(self, metrics: Dict[str, Any], weights: Dict[str, Any]) -> float:
        area_w = float(weights.get("area", 0.45))
        ff_w = float(weights.get("ff", 0.30))
        depth_w = float(weights.get("depth", 0.25))
        pass_w = float(weights.get("pass_bonus", 0.10))

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

        return (
            area_w * (1.0 / (1.0 + area / 1000.0))
            + ff_w * (1.0 / (1.0 + ff / 1000.0))
            + depth_w * (1.0 / (1.0 + depth / 10.0))
            + pass_w * pass_bonus
        )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class Judge:
    """Decides whether the optimization loop should continue or stop.

    Uses heuristic rules first; falls back to LLM only for ambiguous cases
    on complex designs.
    """

    def __init__(self, llm_client: BaseLLMClient, max_iterations: int):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.improvement_streak = 0
        self.stagnation_count = 0
        self.decline_count = 0

    def update_tracking(self, improved: bool, declined: bool) -> None:
        if improved:
            self.improvement_streak += 1
            self.stagnation_count = 0
            self.decline_count = 0
        elif declined:
            self.improvement_streak = 0
            self.stagnation_count += 1
            self.decline_count += 1
        else:
            self.improvement_streak = 0
            self.stagnation_count += 1

    def decide(
        self,
        iteration: int,
        plan: Dict[str, Any],
        score_history: List[float],
        initial_metrics: Dict[str, Any],
        best_metrics: Dict[str, Any],
        best_score: float,
        latest_score: float,
        target_description: str,
    ) -> Dict[str, Any]:
        min_iters = plan.get("min_iterations", 2)
        patience = plan.get("stagnation_patience", 3)

        # Rule 1: always run minimum iterations
        if iteration < min_iters:
            return self._cont(f"最低迭代要求 {min_iters} 轮，当前第 {iteration} 轮。")

        # Rule 2: max budget
        if iteration >= self.max_iterations:
            return self._stop(f"已达最大迭代次数 {self.max_iterations}。")

        # Rule 3: stagnation beyond patience
        if self.stagnation_count >= patience:
            return self._stop(
                f"连续 {self.stagnation_count} 轮未改进（耐心阈值 {patience}），停止。",
            )

        # Rule 4: persistent decline
        if self.decline_count >= patience + 1:
            return self._stop(f"连续 {self.decline_count} 轮下降，停止。")

        # Rule 5: still improving — keep going
        if self.improvement_streak >= 1:
            return self._cont(
                f"连续 {self.improvement_streak} 轮改进，继续当前方向。",
                focus="exploit",
            )

        # Rule 6: for complex designs with moderate stagnation, ask the LLM
        if plan.get("complexity") == "complex" and self.stagnation_count >= 2:
            return self._llm_decide(
                iteration, initial_metrics, best_metrics,
                best_score, latest_score, target_description,
            )

        return self._cont("优化仍有潜力，继续搜索。")

    @property
    def exploration_factor(self) -> float:
        if self.decline_count >= 2:
            return min(0.9, 0.3 + self.decline_count * 0.15)
        if self.stagnation_count >= 2:
            return min(0.7, 0.3 + self.stagnation_count * 0.1)
        if self.improvement_streak >= 2:
            return max(0.1, 0.3 - self.improvement_streak * 0.05)
        return 0.3

    # -- internals ----------------------------------------------------------

    def _cont(self, reason: str, focus: str = "") -> Dict[str, Any]:
        return {"action": "continue", "reason": reason, "focus": focus}

    def _stop(self, reason: str) -> Dict[str, Any]:
        return {"action": "stop", "reason": reason, "focus": "Output current best design."}

    def _llm_decide(
        self,
        iteration: int,
        initial_metrics: Dict[str, Any],
        best_metrics: Dict[str, Any],
        best_score: float,
        latest_score: float,
        target_description: str,
    ) -> Dict[str, Any]:
        payload = {
            "iteration": iteration,
            "target": target_description or "balanced",
            "initial_metrics": initial_metrics,
            "best_metrics": best_metrics,
            "current_best_score": round(best_score, 6),
            "latest_iteration_score": round(latest_score, 6),
            "stagnation_count": self.stagnation_count,
            "decline_count": self.decline_count,
        }
        prompt = (
            "You are the optimization control agent.\n"
            "Decide whether the optimization loop should continue based on the "
            "EDA metrics below.\n"
            "Return JSON only: {\"action\":\"continue|stop\",\"reason\":\"...\","
            "\"focus\":\"...\"}\n\n"
            + json.dumps(payload, ensure_ascii=True, indent=2)
        )
        try:
            raw = self.llm_client.generate(
                prompt,
                system_prompt=(
                    "You decide whether a Verilog optimization loop should "
                    "continue. Base your decision on objective EDA metrics."
                ),
                temperature=0.2,
                max_new_tokens=160,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.0,
            )
            parsed = _parse_agent_json(raw)
            if parsed:
                parsed["raw_model_output"] = raw
                return parsed
        except Exception as exc:
            logger.warning(f"LLM judge failed: {exc}")

        return self._cont("LLM 判断失败，默认继续。")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


class AgentOptimizer:
    """Orchestrates the Planner → Coder → Evaluator → Judge loop."""

    def __init__(
        self,
        model_path: str,
        max_iterations: int = 10,
        population_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        debug_gen: bool = False,
        debug_dir: Optional[str] = None,
        dynamic_strategy: bool = True,
        base_top_p: float = 0.9,
        base_top_k: int = 50,
        base_rep_penalty: float = 1.05,
    ):
        self.model_path = model_path
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.dynamic_strategy = dynamic_strategy
        self.debug_gen = debug_gen

        if debug_dir is None and debug_gen:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_root = Path("outputs") / f"debug_{ts}"
            debug_root.mkdir(parents=True, exist_ok=True)
            self.debug_dir: Optional[str] = str(debug_root)
        else:
            self.debug_dir = debug_dir

        self.eda = EDAWrapper()
        self.llm_client = build_llm_client(model_path)

        # Sampling defaults (forwarded to Coder)
        self._temperature = temperature
        self._top_p = base_top_p
        self._top_k = base_top_k
        self._rep_penalty = base_rep_penalty
        self._max_new_tokens = max_new_tokens

        logger.info(
            f"Optimizer initialized — client mode: "
            f"{getattr(self.llm_client, 'mode', 'unknown')}"
        )

    # -- main entry ---------------------------------------------------------

    def optimize(
        self,
        input_code: str,
        target_description: str = "",
        scenario: str = "",
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        target_profile = build_target_profile(target_description)
        module_name = _extract_module_name(input_code)
        if not module_name:
            return {"success": False, "error": "Unable to extract module name from input code."}

        weights = target_profile.get("weights", {})

        # ---- Phase 0: initial EDA evaluation ----
        self._report(progress_callback, phase="initial_eval", message="正在进行初始 EDA 评估…")
        try:
            initial_data = self.eda.synthesize(input_code, module_name, target_freq=100.0)
        except Exception as exc:
            return {"success": False, "error": f"Initial synthesis failed: {exc}"}

        if not initial_data.get("syntax_ok") or not initial_data.get("synth_ok"):
            return {
                "success": False,
                "error": "Input Verilog contains syntax or synthesis errors.",
            }

        # ---- Phase 1: Planner ----
        self._report(progress_callback, phase="planning", message="Planner 正在分析代码结构…")
        planner = Planner()
        plan = planner.analyze(input_code, initial_data, target_profile)
        logger.info(f"Plan: complexity={plan['complexity']}, strategy={plan['strategy']}")

        effective_max = min(self.max_iterations, plan["suggested_max_iterations"])

        # ---- Construct role instances ----
        evaluator = Evaluator(self.eda)
        initial_score = evaluator.score(initial_data, weights)

        coder = Coder(
            llm_client=self.llm_client,
            target_profile=target_profile,
            max_new_tokens=self._max_new_tokens,
            base_temperature=self._temperature,
            base_top_p=self._top_p,
            base_top_k=self._top_k,
            base_rep_penalty=self._rep_penalty,
            debug_gen=self.debug_gen,
            debug_dir=self.debug_dir,
        )
        judge = Judge(self.llm_client, effective_max)

        # ---- State ----
        best_score = initial_score
        best_code = input_code
        best_metrics = initial_data
        score_history: List[float] = []
        agent_decisions: List[Dict[str, Any]] = []
        optimization_history: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "code": input_code,
                "metrics": initial_data,
                "score": initial_score,
                "is_best": True,
            }
        ]

        self._report(
            progress_callback,
            phase="planning_done",
            message=f"计划完成：复杂度={plan['complexity']}，策略={plan['strategy']}，"
                    f"最大轮次={effective_max}",
            iteration=0,
            max_iterations=effective_max,
            initial_score=initial_score,
            best_score=initial_score,
            best_metrics=initial_data,
            initial_metrics=initial_data,
            plan_summary={
                "complexity": plan["complexity"],
                "strategy": plan["strategy"],
                "bottlenecks": plan["bottlenecks"],
            },
        )

        # ---- Phase 2: Optimization loop ----
        for iteration in range(1, effective_max + 1):
            logger.info(f"Iteration {iteration}/{effective_max}")

            # 2a. Planner refines strategy
            if self.dynamic_strategy and iteration > 1:
                strategy = planner.refine_strategy(
                    plan, score_history,
                    judge.stagnation_count, judge.decline_count,
                )
                plan["strategy"] = strategy
                coder.update_sampling(strategy, judge.exploration_factor)

            self._report(
                progress_callback,
                phase="generating",
                message=f"第 {iteration}/{effective_max} 轮：Coder 正在生成候选方案…",
                iteration=iteration,
                max_iterations=effective_max,
                best_score=best_score,
                initial_score=initial_score,
                strategy=plan["strategy"],
            )

            # 2b. Coder generates
            candidates = coder.generate_candidates(
                base_code=best_code,
                plan=plan,
                iteration=iteration,
                current_score=best_score,
                current_metrics=best_metrics,
                target_description=target_description,
                population_size=self.population_size,
            )

            # 2c. Evaluator evaluates
            iter_best_score = best_score
            iter_best_code = best_code
            iter_best_metrics = best_metrics

            for ci, candidate in enumerate(candidates, 1):
                ref_module = _detect_top_module(best_code) or "top"

                if not _has_complete_module(candidate):
                    continue

                candidate = _try_fix_module_name(candidate, ref_module)
                candidate = _align_module_header(best_code, candidate, ref_module)

                self._report(
                    progress_callback,
                    phase="evaluating",
                    message=f"第 {iteration}/{effective_max} 轮：Evaluator 评估候选 {ci}…",
                    iteration=iteration,
                    max_iterations=effective_max,
                    best_score=best_score,
                    initial_score=initial_score,
                )

                result = evaluator.evaluate(candidate, ref_module)
                if not result.get("success"):
                    optimization_history.append({
                        "iteration": iteration,
                        "candidate": ci,
                        "code": candidate,
                        "score": None,
                        "error": result.get("error", "EDA failed"),
                        "is_best": False,
                    })
                    continue

                metrics = result["data"]
                score = evaluator.score(metrics, weights)
                optimization_history.append({
                    "iteration": iteration,
                    "candidate": ci,
                    "code": candidate,
                    "metrics": metrics,
                    "score": score,
                    "is_best": False,
                })

                if score > iter_best_score:
                    # Equivalence gate: only accept if functionally equivalent
                    equiv_ok = evaluator.check_equivalence(
                        input_code, candidate, ref_module,
                    )
                    if not equiv_ok:
                        logger.info(
                            f"Candidate {ci} iter {iteration}: score {score:.4f} "
                            f"but FAILED equivalence check — rejected"
                        )
                        optimization_history[-1]["equiv_rejected"] = True
                        continue

                    iter_best_score = score
                    iter_best_code = candidate
                    iter_best_metrics = metrics
                    if self.debug_dir:
                        p = Path(self.debug_dir) / f"best_candidate_iter_{iteration}.v"
                        p.write_text(candidate, encoding="utf-8")

            score_history.append(iter_best_score)

            improved = iter_best_score > best_score
            declined = iter_best_score < best_score - 1e-6

            if improved:
                best_score = iter_best_score
                best_code = iter_best_code
                best_metrics = iter_best_metrics

            # 2d. Judge decides
            judge.update_tracking(improved, declined)
            decision = judge.decide(
                iteration=iteration,
                plan=plan,
                score_history=score_history,
                initial_metrics=initial_data,
                best_metrics=best_metrics,
                best_score=best_score,
                latest_score=iter_best_score,
                target_description=target_description,
            )
            agent_decisions.append(decision)

            self._report(
                progress_callback,
                phase="iteration_done",
                message=f"第 {iteration}/{effective_max} 轮完成 — "
                        f"Judge: {decision['action']}（{decision['reason']}）",
                iteration=iteration,
                max_iterations=effective_max,
                best_score=best_score,
                initial_score=initial_score,
                latest_score=iter_best_score,
                decision=decision,
                best_metrics=best_metrics,
                initial_metrics=initial_data,
                agent_decisions=agent_decisions,
            )

            if decision["action"] == "stop":
                logger.info(f"Judge stopped at iteration {iteration}: {decision['reason']}")
                break

        # ---- Phase 3: package results ----
        improvement = {
            "area_improvement": _safe_pct(initial_data["area"], best_metrics["area"]),
            "ff_improvement": _safe_pct(
                initial_data.get("num_ff", 0), best_metrics.get("num_ff", 0),
            ),
            "depth_improvement": _safe_pct(
                initial_data.get("logic_depth", 0), best_metrics.get("logic_depth", 0),
            ),
            "score_improvement": (
                (best_score - initial_score) / max(1e-9, abs(initial_score)) * 100
            ),
        }

        result: Dict[str, Any] = {
            "success": True,
            "original_code": input_code,
            "optimized_code": best_code,
            "original_metrics": initial_data,
            "optimized_metrics": best_metrics,
            "improvement": improvement,
            "optimization_history": optimization_history,
            "agent_decisions": agent_decisions,
            "total_iterations": len(score_history),
            "total_candidates": max(0, len(optimization_history) - 1),
            "target_profile": target_profile,
            "llm_mode": getattr(self.llm_client, "mode", "unknown"),
        }
        result["ai_insights"] = generate_ai_insights(
            original_code=input_code,
            optimized_code=best_code,
            original_metrics=initial_data,
            optimized_metrics=best_metrics,
            profile=target_profile,
            target_description=target_description,
            scenario=scenario,
        )
        result["competition_package"] = generate_competition_package(
            result=result,
            profile=target_profile,
            target_description=target_description,
            scenario=scenario,
        )
        return result

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _report(cb: ProgressCallback, **kwargs: Any) -> None:
        if cb is not None:
            try:
                cb(kwargs)
            except Exception:
                pass

    def cleanup(self) -> None:
        if hasattr(self, "eda"):
            self.eda.cleanup()
        if hasattr(self, "llm_client"):
            self.llm_client.cleanup()

    def save_optimization_report(self, result: Dict[str, Any], output_path: str) -> None:
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "optimization_config": {
                "max_iterations": self.max_iterations,
                "population_size": self.population_size,
                "llm_mode": getattr(self.llm_client, "mode", "unknown"),
            },
            "result": result,
        }
        Path(output_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(f"Optimization report saved to {output_path}")


# ---------------------------------------------------------------------------
# Shared utilities (module-level)
# ---------------------------------------------------------------------------

def _extract_module_name(code: str) -> Optional[str]:
    m = re.search(r"\bmodule\s+(\w+)", code, re.IGNORECASE)
    return m.group(1) if m else None


def _detect_top_module(code: str) -> Optional[str]:
    text = re.sub(r"/\*[\s\S]*?\*/", "", code)
    text = re.sub(r"//.*", "", text)
    names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if not names:
        return None
    for n in names:
        if n.lower() == "top":
            return n
    return names[0]


def _has_complete_module(text: str) -> bool:
    return "module" in text and "endmodule" in text


def _try_fix_module_name(code: str, module_name: str) -> str:
    pat = re.compile(
        r"(\bmodule\s+)([A-Za-z_][A-Za-z0-9_]*)(\s*(?:#\s*\([\s\S]*?\))?\s*\()",
        re.IGNORECASE,
    )
    return pat.sub(lambda m: f"{m.group(1)}{module_name}{m.group(3)}", code, count=1)


def _extract_module_ports(code: str, module_name: str) -> Optional[str]:
    pat = re.compile(
        rf"\bmodule\s+{re.escape(module_name)}\s*(?:#\s*\([\s\S]*?\))?\s*\(([\s\S]*?)\)\s*;",
        re.IGNORECASE,
    )
    m = pat.search(code)
    return m.group(1) if m else None


def _extract_module_params(code: str, module_name: str) -> Optional[str]:
    pat = re.compile(
        rf"\bmodule\s+{re.escape(module_name)}\s*(#\s*\([\s\S]*?\))\s*\(",
        re.IGNORECASE,
    )
    m = pat.search(code)
    return m.group(1) if m else None


def _align_module_header(base_code: str, cand_code: str, module_name: str) -> str:
    base_ports = _extract_module_ports(base_code, module_name)
    base_params = _extract_module_params(base_code, module_name)
    if not base_ports:
        return cand_code

    pat = re.compile(
        rf"(\bmodule\s+{re.escape(module_name)}\s*)(#\s*\([\s\S]*?\))?\s*\([\s\S]*?\)(\s*;)",
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        params = base_params if base_params is not None else (m.group(2) or "")
        suffix = m.group(3)
        return f"{prefix}{params}({base_ports}){suffix}"

    return pat.sub(repl, cand_code, count=1)


def _extract_code_block(text: str) -> str:
    fenced = list(
        re.finditer(r"```(?:verilog|systemverilog)?\s*\n([\s\S]*?)\n```", text, re.IGNORECASE),
    )
    if fenced:
        return fenced[-1].group(1).strip()
    module_match = re.search(r"\bmodule\b[\s\S]*?\bendmodule\b", text, re.IGNORECASE)
    if module_match:
        return module_match.group(0).strip()
    return text.strip()


def _normalize_code(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    return re.sub(r"\s+", " ", code).strip()


def _parse_agent_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("action") in ("continue", "stop"):
            return data
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and data.get("action") in ("continue", "stop"):
                return data
        except Exception:
            pass
    return None


def _safe_pct(old: Any, new: Any) -> float:
    old_f = float(old or 0)
    new_f = float(new or 0)
    if abs(old_f) < 1e-9:
        return 0.0
    return (old_f - new_f) / abs(old_f) * 100


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agent-driven Verilog optimizer")
    parser.add_argument("--model", required=True, help="Model or adapter path")
    parser.add_argument("--input", "-i", help="Input Verilog file")
    parser.add_argument("--code", "-c", help="Inline Verilog code")
    parser.add_argument("--target", "-t", default="", help="Optimization target")
    parser.add_argument("--scenario", default="", help="Application scenario")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--population", type=int, default=5, help="Candidates per iter")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", "-o", help="Output directory")
    args = parser.parse_args()

    if args.input:
        input_code = Path(args.input).read_text(encoding="utf-8")
    elif args.code:
        input_code = args.code
    else:
        print("Please provide --input or --code")
        return

    optimizer = AgentOptimizer(
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
                out = Path(args.output)
                out.mkdir(exist_ok=True)
                (out / "optimized_code.v").write_text(
                    result["optimized_code"], encoding="utf-8",
                )
                optimizer.save_optimization_report(
                    result, str(out / "optimization_report.json"),
                )
                md = result.get("competition_package", {}).get("markdown_report")
                if md:
                    (out / "competition_summary.md").write_text(md, encoding="utf-8")
                print(f"Saved to: {args.output}")
        else:
            print(f"Optimization failed: {result['error']}")
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()
