#!/usr/bin/env python3
"""
模块4: OptiHDL 启动脚本
单一模式：自动选择 models/ 下的模型或用 --model 指定，
输入一个 Verilog 文件，先做 EDA 评分，然后进入 RL 循环优化，输出结果。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def auto_find_model(models_dir: Path) -> str:
    """在 models/ 下自动选择一个可用模型目录（优先含 adapter_config.json，其次 config.json）。"""
    if not models_dir.exists():
        raise FileNotFoundError(f"未找到模型目录: {models_dir}")
    candidates = []
    for p in models_dir.iterdir():
        if p.is_dir():
            if (p / 'adapter_config.json').exists() or (p / 'config.json').exists():
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"models/ 下未找到可用模型（缺少 adapter_config.json 或 config.json）")
    # 选择修改时间最新的
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main():
    parser = argparse.ArgumentParser(description="OptiHDL - 一键优化：models下自动选模型 + 输入Verilog文件")
    parser.add_argument('--model', help='模型或适配器目录（缺省时自动从 models/ 下选择）')
    parser.add_argument('--input', '-i', required=True, help='输入Verilog文件')
    parser.add_argument('--target', '-t', default='', help='优化目标（可选）')
    parser.add_argument('--iterations', type=int, default=10, help='最大迭代次数')
    parser.add_argument('--population', type=int, default=5, help='候选代码数量')
    parser.add_argument('--temperature', type=float, default=0.8, help='生成温度')
    parser.add_argument('--output', '-o', help='输出目录（保存优化代码与报告）')

    args = parser.parse_args()

    # 确定模型路径
    model_path = args.model or auto_find_model(project_root / 'models')

    # 读取输入代码
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入Verilog文件不存在: {input_path}")
    code = input_path.read_text(encoding='utf-8', errors='ignore')

    # 运行优化
    from optimization.rl_optimizer import RLOptimizer
    optimizer = RLOptimizer(
        model_path=model_path,
        max_iterations=args.iterations,
        population_size=args.population,
        temperature=args.temperature,
    )
    try:
        result = optimizer.optimize(code, args.target)
    finally:
        optimizer.cleanup()

    # 输出结果
    if result.get('success'):
        print("=== 优化完成 ===")
        print(f"模型: {model_path}")
        print(f"面积改进: {result['improvement']['area_improvement']:.2f}%")
        print(f"触发器数改进: {result['improvement']['ff_improvement']:.2f}%")
        print(f"逻辑深度改进: {result['improvement']['depth_improvement']:.2f}%")
        print(f"总分改进: {result['improvement']['score_improvement']:.2f}%")

        # 打印最后一次对比
        orig = result['original_metrics']
        opt = result['optimized_metrics']
        print("--- 指标对比 ---")
        print(f"面积: {orig['area']} -> {opt['area']}")
        print(f"触发器数: {orig.get('num_ff', 0)} -> {opt.get('num_ff', 0)}")
        print(f"逻辑深度: {orig.get('logic_depth', 0)} -> {opt.get('logic_depth', 0)}")

        # 保存输出
        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / 'optimized_code.v').write_text(result['optimized_code'], encoding='utf-8')
            import json as _json
            (out_dir / 'optimization_report.json').write_text(_json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"结果已保存到: {out_dir}")
    else:
        print(f"优化失败: {result.get('error')}")


if __name__ == "__main__":
    main()
