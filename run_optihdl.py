#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def auto_find_model(models_dir: Path) -> str:
    if os.environ.get("OPTIHDL_LLM_MODE", "").strip().lower() == "api":
        return "api-client"
    if not models_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    candidates = []
    for path in models_dir.iterdir():
        if path.is_dir() and ((path / "adapter_config.json").exists() or (path / "config.json").exists()):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No available model found under models/")

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="OptiHDL agent optimizer")
    parser.add_argument("--model", help="Model or adapter path. Optional in API mode.")
    parser.add_argument("--input", "-i", required=True, help="Input Verilog file")
    parser.add_argument("--target", "-t", default="", help="Optimization target description")
    parser.add_argument("--scenario", default="", help="Application scenario description")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--population", type=int, default=1, help="Candidates per iteration")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum generated tokens")
    parser.add_argument("--debug-gen", action="store_true", help="Enable generation debug output")
    parser.add_argument("--rl-mode", action="store_true", default=True, help="Enable dynamic strategy")
    parser.add_argument("--no-rl", dest="rl_mode", action="store_false", help="Disable dynamic strategy")
    parser.add_argument("--debug-dir", type=str, help="Custom debug output directory")
    parser.add_argument("--output", "-o", help="Output directory")
    args = parser.parse_args()

    model_path = args.model or auto_find_model(project_root / "models")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Verilog file not found: {input_path}")
    code = input_path.read_text(encoding="utf-8", errors="ignore")

    from optimization.rl_optimizer import RLOptimizer

    optimizer = RLOptimizer(
        model_path=model_path,
        max_iterations=args.iterations,
        population_size=args.population,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        debug_gen=args.debug_gen,
        debug_dir=args.debug_dir,
        rl_mode=args.rl_mode,
        base_top_p=getattr(args, "top_p", 0.9),
        base_top_k=getattr(args, "top_k", 50),
        base_rep_penalty=getattr(args, "repetition_penalty", 1.05),
    )

    try:
        result = optimizer.optimize(code, args.target, args.scenario)
    finally:
        optimizer.cleanup()

    if not result.get("success"):
        print(f"Optimization failed: {result.get('error')}")
        return

    print("=== Optimization Finished ===")
    print(f"Model: {model_path}")
    print(f"LLM mode: {result.get('llm_mode')}")
    print(f"Area improvement: {result['improvement']['area_improvement']:.2f}%")
    print(f"FF improvement: {result['improvement']['ff_improvement']:.2f}%")
    print(f"Depth improvement: {result['improvement']['depth_improvement']:.2f}%")
    print(f"Score improvement: {result['improvement']['score_improvement']:.2f}%")

    orig = result["original_metrics"]
    opt = result["optimized_metrics"]
    print("--- Metrics ---")
    print(f"Area: {orig['area']} -> {opt['area']}")
    print(f"FF: {orig.get('num_ff', 0)} -> {opt.get('num_ff', 0)}")
    print(f"Depth: {orig.get('logic_depth', 0)} -> {opt.get('logic_depth', 0)}")

    if args.output:
        out_dir = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"opt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "optimized_code.v").write_text(result["optimized_code"], encoding="utf-8")
    (out_dir / "optimization_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    competition_summary = result.get("competition_package", {}).get("markdown_report")
    if competition_summary:
        (out_dir / "competition_summary.md").write_text(competition_summary, encoding="utf-8")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
