from __future__ import annotations

from typing import Any, Dict, List
import re


DEFAULT_PROFILE = {
    "name": "balanced",
    "label": "平衡优化",
    "description": "同时兼顾面积、寄存器数量与逻辑深度，适合通用型数字电路优化。",
    "weights": {
        "area": 0.40,
        "ff": 0.30,
        "depth": 0.15,
        "timing": 0.15,
        "pass_bonus": 0.10,
    },
    "prompt_focus": "优先在不破坏接口与语义的前提下，平衡面积、时序与资源使用。",
}


TARGET_PROFILES = [
    {
        "name": "area",
        "label": "面积优先",
        "description": "面向资源受限部署，重点压缩单元面积和冗余逻辑。",
        "keywords": ["面积", "资源", "成本", "低功耗", "边缘", "resource", "area", "small"],
        "weights": {"area": 0.55, "ff": 0.20, "depth": 0.10, "timing": 0.15, "pass_bonus": 0.10},
        "prompt_focus": "优先减少组合逻辑规模、复用中间信号并消除冗余状态。",
    },
    {
        "name": "timing",
        "label": "时序优先",
        "description": "面向高性能场景，重点缩短关键路径并降低逻辑深度。",
        "keywords": ["时序", "延迟", "频率", "吞吐", "高性能", "timing", "latency", "speed", "throughput"],
        "weights": {"area": 0.15, "ff": 0.15, "depth": 0.30, "timing": 0.40, "pass_bonus": 0.10},
        "prompt_focus": "优先拆分长组合路径，压缩关键路径深度并改善高频运行能力。",
    },
    {
        "name": "register",
        "label": "寄存器优先",
        "description": "面向寄存器密集型设计，重点减少触发器与状态冗余。",
        "keywords": ["寄存器", "触发器", "状态机", "flip-flop", "ff", "register"],
        "weights": {"area": 0.20, "ff": 0.50, "depth": 0.10, "timing": 0.20, "pass_bonus": 0.10},
        "prompt_focus": "优先压缩触发器数量、合并状态与寄存器复用。",
    },
    {
        "name": "readability",
        "label": "可维护优先",
        "description": "面向工程落地和教学展示，强调结构清晰、可解释与可维护。",
        "keywords": ["可读", "可维护", "解释", "教学", "readable", "maintainable", "explain"],
        "weights": {"area": 0.30, "ff": 0.20, "depth": 0.30, "timing": 0.20, "pass_bonus": 0.10},
        "prompt_focus": "在保证优化收益的同时，使结构更清晰、便于答辩解释与维护。",
    },
]


def build_target_profile(target_description: str) -> Dict[str, Any]:
    normalized = (target_description or "").lower()
    for profile in TARGET_PROFILES:
        for keyword in profile["keywords"]:
            if keyword.lower() in normalized:
                return {
                    "name": profile["name"],
                    "label": profile["label"],
                    "description": profile["description"],
                    "weights": dict(profile["weights"]),
                    "prompt_focus": profile["prompt_focus"],
                    "matched_keyword": keyword,
                }

    profile = dict(DEFAULT_PROFILE)
    profile["weights"] = dict(DEFAULT_PROFILE["weights"])
    profile["matched_keyword"] = ""
    return profile


def summarize_code_features(code: str) -> Dict[str, Any]:
    text = code or ""
    stripped = _strip_comments(text)
    lines = [line for line in stripped.splitlines() if line.strip()]
    always_count = len(re.findall(r"\balways\b", stripped))
    assign_count = len(re.findall(r"\bassign\b", stripped))
    case_count = len(re.findall(r"\bcase\b", stripped))
    if_count = len(re.findall(r"\bif\b", stripped))
    generate_count = len(re.findall(r"\bgenerate\b", stripped))
    parameter_count = len(re.findall(r"\bparameter\b", stripped))
    module_names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    register_decl_count = len(re.findall(r"\breg\b", stripped))
    wire_decl_count = len(re.findall(r"\bwire\b", stripped))

    return {
        "line_count": len(lines),
        "always_count": always_count,
        "assign_count": assign_count,
        "case_count": case_count,
        "if_count": if_count,
        "generate_count": generate_count,
        "parameter_count": parameter_count,
        "module_count": len(module_names),
        "register_decl_count": register_decl_count,
        "wire_decl_count": wire_decl_count,
        "has_pipeline_style": bool(re.search(r"\bpipeline\b|\bstage\b", stripped, flags=re.IGNORECASE)),
        "has_fsm_style": case_count > 0 and always_count >= 2,
    }


def generate_ai_insights(
    original_code: str,
    optimized_code: str,
    original_metrics: Dict[str, Any],
    optimized_metrics: Dict[str, Any],
    profile: Dict[str, Any],
    target_description: str = "",
    scenario: str = "",
) -> Dict[str, Any]:
    original_features = summarize_code_features(original_code)
    optimized_features = summarize_code_features(optimized_code)
    bottlenecks = _diagnose_bottlenecks(original_metrics, original_features, profile)
    recommendations = _build_recommendations(original_metrics, original_features, profile)
    diff_summary = _build_diff_summary(original_features, optimized_features)

    return {
        "target_profile": profile["label"],
        "target_description": target_description or "未指定，自适应平衡优化",
        "scenario": scenario or "智能芯片设计辅助/EDA优化",
        "design_profile": {
            "line_count": original_features["line_count"],
            "always_blocks": original_features["always_count"],
            "assign_statements": original_features["assign_count"],
            "fsm_like": original_features["has_fsm_style"],
            "pipeline_like": original_features["has_pipeline_style"],
        },
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "diff_summary": diff_summary,
        "innovation_tags": [
            "多目标奖励建模",
            "Agentic EDA闭环优化",
            "可解释AI诊断",
            "面向竞赛的自动化文档生成",
        ],
    }


def generate_competition_package(
    result: Dict[str, Any],
    profile: Dict[str, Any],
    target_description: str = "",
    scenario: str = "",
) -> Dict[str, Any]:
    scenario_text = scenario or "智能制造与工业互联网中的数字电路自动优化"
    target_text = target_description or profile["label"]
    improvement = result.get("improvement", {})
    ai_insights = result.get("ai_insights", {})
    metrics = result.get("optimized_metrics", {})

    summary = (
        f"OptiHDL 面向“{scenario_text}”场景，构建了一个以大模型代码生成、"
        f"EDA反馈评估、多目标奖励重加权和可解释诊断为核心的智能优化系统。"
        f"系统根据“{target_text}”自动选择优化策略，并输出可展示、可答辩的分析结论。"
    )

    innovation_points = [
        "把 HDL 代码生成与 EDA 指标反馈组成闭环，实现候选生成、综合评估、策略调整的自动迭代。",
        "引入多目标优化画像，根据面积优先、时序优先、寄存器优先等目标动态调整奖励函数。",
        "增加可解释 AI 模块，自动识别设计瓶颈、结构风格和优化建议，提升答辩可解释性。",
        "自动生成比赛报告摘要、演示脚本和用户说明草案，降低从原型到参赛作品的包装成本。",
    ]

    demo_script = [
        "输入待优化 Verilog 代码并选择场景与优化目标。",
        "系统先执行 EDA 分析，识别面积、触发器数量和逻辑深度等指标。",
        "智能优化代理根据目标画像生成候选方案，并在多轮反馈中选择最优设计。",
        "界面展示优化前后代码、关键指标变化、瓶颈诊断和比赛说明摘要。",
    ]

    defense_points = [
        "作品解决的是数字电路设计中人工优化效率低、经验依赖强、评估成本高的问题。",
        "AI 方法不仅体现在代码生成，还体现在奖励建模、策略调节和可解释结果生成。",
        "系统输出不是单一代码片段，而是完整的分析闭环与可验证指标。",
        "现场可直接演示输入代码、自动分析、自动优化和结果报告生成。",
    ]

    markdown_report = build_markdown_report(
        scenario_text=scenario_text,
        target_text=target_text,
        summary=summary,
        innovation_points=innovation_points,
        result=result,
        ai_insights=ai_insights,
        demo_script=demo_script,
        defense_points=defense_points,
        metrics=metrics,
        improvement=improvement,
    )

    return {
        "application_scenario": scenario_text,
        "target_focus": target_text,
        "summary": summary,
        "innovation_points": innovation_points,
        "demo_script": demo_script,
        "defense_points": defense_points,
        "markdown_report": markdown_report,
    }


def build_markdown_report(
    scenario_text: str,
    target_text: str,
    summary: str,
    innovation_points: List[str],
    result: Dict[str, Any],
    ai_insights: Dict[str, Any],
    demo_script: List[str],
    defense_points: List[str],
    metrics: Dict[str, Any],
    improvement: Dict[str, Any],
) -> str:
    bottlenecks = ai_insights.get("bottlenecks", [])
    recommendations = ai_insights.get("recommendations", [])

    lines = [
        "# OptiHDL 比赛方案摘要",
        "",
        "## 应用场景",
        scenario_text,
        "",
        "## 作品概述",
        summary,
        "",
        "## 技术方案",
        "系统由 Verilog 代码输入、EDA 分析器、LLM 候选生成器、多目标评分器和可解释诊断模块组成。",
        f"当前优化目标画像：{target_text}。",
        "",
        "## 创新点",
    ]
    lines.extend([f"- {item}" for item in innovation_points])
    lines.extend(
        [
            "",
            "## 指标结果",
            f"- 面积优化：{improvement.get('area_improvement', 0.0):.2f}%",
            f"- 触发器优化：{improvement.get('ff_improvement', 0.0):.2f}%",
            f"- 逻辑深度优化：{improvement.get('depth_improvement', 0.0):.2f}%",
            f"- 综合得分提升：{improvement.get('score_improvement', 0.0):.2f}%",
            f"- 优化后面积：{metrics.get('area', '--')}",
            f"- 优化后触发器数量：{metrics.get('num_ff', '--')}",
            f"- 优化后逻辑深度：{metrics.get('logic_depth', '--')}",
            "",
            "## 可解释诊断",
        ]
    )
    lines.extend([f"- 主要瓶颈：{item}" for item in bottlenecks] or ["- 当前未识别到明显瓶颈"])
    lines.extend([f"- 策略建议：{item}" for item in recommendations] or ["- 当前未生成额外建议"])
    lines.extend(
        [
            "",
            "## 演示流程",
        ]
    )
    lines.extend([f"{idx}. {item}" for idx, item in enumerate(demo_script, start=1)])
    lines.extend(
        [
            "",
            "## 答辩要点",
        ]
    )
    lines.extend([f"- {item}" for item in defense_points])
    lines.append("")
    return "\n".join(lines)


def _diagnose_bottlenecks(metrics: Dict[str, Any], features: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    bottlenecks: List[str] = []
    area = float(metrics.get("area", 0) or 0)
    ff = float(metrics.get("num_ff", 0) or 0)
    depth = float(metrics.get("logic_depth", 0) or 0)

    if area >= 150:
        bottlenecks.append("组合逻辑规模偏大，面积压缩空间明显。")
    if ff >= 32:
        bottlenecks.append("寄存器数量较多，状态或流水结构存在合并空间。")
    if depth >= 12:
        bottlenecks.append("逻辑深度偏高，关键路径可能限制频率。")
    if features["always_count"] >= 3 and features["case_count"] == 0:
        bottlenecks.append("时序过程较多但状态组织不清晰，结构可解释性偏弱。")
    if not bottlenecks:
        bottlenecks.append(f"当前设计整体均衡，后续应围绕“{profile['label']}”做细粒度优化。")

    return bottlenecks


def _build_recommendations(metrics: Dict[str, Any], features: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    recommendations: List[str] = [profile["prompt_focus"]]
    area = float(metrics.get("area", 0) or 0)
    ff = float(metrics.get("num_ff", 0) or 0)
    depth = float(metrics.get("logic_depth", 0) or 0)

    if area >= 150:
        recommendations.append("尝试提取公共子表达式，减少重复 assign/always 逻辑。")
    if ff >= 32:
        recommendations.append("检查是否可以合并状态寄存器、缩减中间寄存器或改为组合共享。")
    if depth >= 12:
        recommendations.append("针对长组合链条进行重写，必要时引入更规则的分层结构。")
    if features["has_fsm_style"]:
        recommendations.append("对于状态机模块，建议增加状态编码策略说明，提升比赛答辩解释性。")
    if features["has_pipeline_style"]:
        recommendations.append("对于流水设计，可增加吞吐量与时延的权衡分析，补强实验对比。")

    return recommendations


def _build_diff_summary(original_features: Dict[str, Any], optimized_features: Dict[str, Any]) -> List[str]:
    summary: List[str] = []
    line_delta = optimized_features["line_count"] - original_features["line_count"]
    always_delta = optimized_features["always_count"] - original_features["always_count"]
    assign_delta = optimized_features["assign_count"] - original_features["assign_count"]

    if line_delta != 0:
        direction = "增加" if line_delta > 0 else "减少"
        summary.append(f"代码有效行数{direction}了 {abs(line_delta)} 行。")
    if always_delta != 0:
        direction = "增加" if always_delta > 0 else "减少"
        summary.append(f"`always` 块数量{direction}了 {abs(always_delta)} 个。")
    if assign_delta != 0:
        direction = "增加" if assign_delta > 0 else "减少"
        summary.append(f"`assign` 语句数量{direction}了 {abs(assign_delta)} 个。")
    if not summary:
        summary.append("优化前后整体结构接近，主要收益来自细粒度逻辑改写。")

    return summary


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r"//.*", "", code)
    return code
