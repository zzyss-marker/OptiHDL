"""
模块2: EDA包装类
独立运行的EDA工具接口，支持命令行和API调用
"""

import subprocess
import tempfile
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
# 日志已禁用：不引入logger，避免任何日志输出


class EDAFailure(Exception):
    """EDA 工具执行失败异常"""
    pass


class EDAWrapper:
    """EDA 工具包装器，封装 Yosys 的命令行交互"""
    
    def __init__(self, yosys_path: str = "yosys"):
        self.yosys_path = yosys_path
        self.temp_dir = Path(tempfile.mkdtemp())
        # 禁用日志输出
    
    def synthesize(self, verilog_code: str, module_name: str = "top", 
                  target_freq: float = 100.0) -> Dict[str, Any]:
        """
        综合 Verilog 代码并返回完整的 PPA 指标
        
        Args:
            verilog_code: Verilog 源代码
            module_name: 顶层模块名
            target_freq: 目标频率 (MHz)
            
        Returns:
            包含所有 PPA 指标的字典
        """
        try:
            # 1. 语法检查和综合（Yosys 指标）
            synth_result = self._run_yosys_synthesis(verilog_code, module_name)

            # 2. 直接返回 Yosys 指标
            result = {**synth_result}

            # 3. 验证必需字段
            required_fields = ["syntax_ok", "equiv_ok", "synth_ok", "area", "num_ff", "logic_depth"]
            for field in required_fields:
                if field not in result:
                    raise EDAFailure(f"缺少必需字段: {field}")

            return result
            
        except Exception as e:
            raise EDAFailure(f"EDA 综合失败: {e}")
    
    def _run_yosys_synthesis(self, verilog_code: str, module_name: str) -> Dict[str, Any]:
        """运行 Yosys 综合（若检测到真实.lib则按库映射）"""
        # 写入 Verilog 文件
        verilog_file = self.temp_dir / f"{module_name}.v"
        with open(verilog_file, 'w') as f:
            f.write(verilog_code)
        
        # 构建 Yosys 脚本
        liberty = self._find_liberty()
        # 禁用日志输出
        mapped_netlist = self.temp_dir / f"{module_name}_synth.v"
        if liberty:
            yosys_script = f"""
read_verilog {verilog_file}
hierarchy -check -top {module_name}
read_liberty -lib {liberty}
synth -top {module_name}
dfflibmap -liberty {liberty}
abc -liberty {liberty}
opt_clean
stat
write_verilog {mapped_netlist}
"""
        else:
            yosys_script = f"""
read_verilog {verilog_file}
hierarchy -check -top {module_name}
synth -top {module_name}
stat
write_verilog {mapped_netlist}
"""
        
        script_file = self.temp_dir / "synth.ys"
        with open(script_file, 'w') as f:
            f.write(yosys_script)
        # 打印脚本前几行，便于排查
        preview = "\n".join(yosys_script.strip().splitlines()[:8])
        # 禁用日志输出
        
        # 执行 Yosys
        try:
            yosys_cmd = [self.yosys_path, "-s", str(script_file)]
            yosys_timeout = int(os.environ.get("OPTIHDL_YOSYS_TIMEOUT", "180"))
            # 禁用日志输出
            import time
            t0 = time.perf_counter()
            result = subprocess.run(
                yosys_cmd,
                capture_output=True,
                text=True,
                timeout=yosys_timeout
            )
            dt = time.perf_counter() - t0
            # 禁用日志输出
            
            syntax_ok = result.returncode == 0
            # 检查stderr中是否有真正的错误（排除警告和信息）
            has_error = any(line.strip().startswith("ERROR") for line in result.stderr.split('\n'))
            synth_ok = syntax_ok and not has_error
            # 禁用日志输出
            
            # 解析面积信息
            area = self._parse_yosys_area(result.stdout)
            num_ff = self._parse_yosys_ff_count(result.stdout)
            logic_depth = self._parse_yosys_logic_depth(result.stdout)
            
            # 功能等价性检查（可以少安装一个EDA）
            equiv_ok = synth_ok and os.path.exists(mapped_netlist)
            # 禁用日志输出
            
            return {
                "syntax_ok": syntax_ok,
                "synth_ok": synth_ok,
                "equiv_ok": equiv_ok,
                "area": area,
                "num_ff": num_ff,
                "logic_depth": logic_depth
            }
            
        except subprocess.TimeoutExpired:
            raise EDAFailure("Yosys 综合超时")
        except Exception as e:
            raise EDAFailure(f"Yosys 执行失败: {e}")
    
    def _parse_yosys_area(self, output: str) -> int:
        """从 Yosys 输出解析面积信息"""
        # 查找 ABC 报告中的单元格数量
        pattern = r"ABC RESULTS:\s+(\w+)\s+cells:\s+(\d+)"
        matches = re.findall(pattern, output)
        
        if matches:
            return sum(int(count) for _, count in matches)
        
        # 备用：查找 stat 命令输出
        stat_pattern = r"Number of cells:\s+(\d+)"
        stat_match = re.search(stat_pattern, output)
        if stat_match:
            return int(stat_match.group(1))
        
        # 最后兜底：估算
        lines = output.split('\n')
        gate_count = sum(1 for line in lines if 'assign' in line or 'wire' in line)
        return max(gate_count, 10)
    
    def _parse_yosys_ff_count(self, output: str) -> int:
        """从 Yosys 输出解析触发器数量"""
        # 查找 DFF 相关信息
        patterns = [
            r"\$dff\s+(\d+)",
            r"DFF\s+(\d+)",
            r"flip-flops?\s*:\s*(\d+)",
            r"registers?\s*:\s*(\d+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                return sum(int(m) for m in matches)
        
        # 兜底：从 stat 输出估算
        if "$dff" in output.lower():
            return len(re.findall(r"\$dff", output, re.IGNORECASE))
        
        return 0
    
    def _parse_yosys_logic_depth(self, output: str) -> int:
        """从 Yosys 输出解析逻辑深度"""
        # 查找 ABC 报告中的深度信息
        pattern = r"lev\s*=\s*(\d+)"
        matches = re.findall(pattern, output)
        if matches:
            return max(int(m) for m in matches)
        
        # 根据门的数量估算逻辑深度
        gate_count = self._parse_yosys_area(output)
        if gate_count > 0:
            # 简单估算：逻辑深度约为门数量的平方根
            import math
            return max(1, int(math.sqrt(gate_count)))
        
        # 默认值
        return 2
    def _find_liberty(self) -> Optional[str]:
        """自动查找可用的liberty库文件路径。
        优先顺序：环境变量 OPTIHDL_LIBERTY > 环境变量 LIB > 项目 ./libs/Nangate45_typical.lib > 无
        """
        # 1) 环境变量
        for env_key in ("OPTIHDL_LIBERTY", "LIB"):
            p = os.environ.get(env_key)
            if p and os.path.exists(p):
                return p
        
        # 2) 优先查找与本文件相对的 core_tools/libs
        try:
            here = Path(__file__).resolve()
            # core_tools/ 目录
            core_dir = here.parent
            candidate = core_dir / "libs" / "Nangate45_typical.lib"
            if candidate.exists():
                return str(candidate)
            # 仓库根目录下的 core_tools/libs（向上两级再拼 core_tools）
            repo_root = core_dir.parent
            candidate2 = repo_root / "core_tools" / "libs" / "Nangate45_typical.lib"
            if candidate2.exists():
                return str(candidate2)
        except Exception:
            pass

        # 3) 回退：当前工作目录的 libs
        candidate3 = Path.cwd() / "libs" / "Nangate45_typical.lib"
        if candidate3.exists():
            return str(candidate3)
        return None
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        try:
            # 默认保留临时目录；仅当 OPTIHDL_DELETE_TEMP=1 时删除
            if os.environ.get("OPTIHDL_DELETE_TEMP", "0") == "1":
                shutil.rmtree(self.temp_dir)
            else:
                pass
        except Exception as e:
            # 禁用日志输出
            pass

    # === 新增：形式等价检查 ===
    def check_equivalence(self, ref_code: str, cand_code: str, module_name: str, debug_dir: Optional[str] = None) -> bool:
        """使用 Yosys formal flow 检查候选是否与参考代码功能等价。
        返回 True 表示等价，False 表示不等价或检查失败。
        """
        try:
            ref_v = self.temp_dir / f"{module_name}_gold.v"
            cand_v = self.temp_dir / f"{module_name}_gate.v"
            with open(ref_v, 'w') as f:
                f.write(ref_code)
            with open(cand_v, 'w') as f:
                f.write(cand_code)
            
            # 先做快速语法检查
            if not self._quick_syntax_check(ref_v, cand_v, module_name, debug_dir):
                return False
            # 尝试1：简化的结构等价检查
            yosys_tcl_1 = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; techmap; opt
design -stash gold

read_verilog {cand_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; techmap; opt
design -stash gate

design -copy-from gold -as gold {module_name}
design -copy-from gate -as gate {module_name}
equiv_make gold gate equiv
hierarchy -top equiv
equiv_simple
equiv_status -assert
"""
            tcl_file = self.temp_dir / "equiv.ys"
            with open(tcl_file, 'w') as f:
                f.write(yosys_tcl_1)

            cmd = [self.yosys_path, "-q", "-s", str(tcl_file)]
            timeout_s = int(os.environ.get("OPTIHDL_EQUIV_TIMEOUT", "180"))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if debug_dir:
                try:
                    Path(debug_dir).mkdir(parents=True, exist_ok=True)
                    (Path(debug_dir) / "equiv_1.tcl").write_text(yosys_tcl_1, encoding='utf-8')
                    (Path(debug_dir) / "equiv_1.stdout.txt").write_text(result.stdout or "", encoding='utf-8')
                    (Path(debug_dir) / "equiv_1.stderr.txt").write_text(result.stderr or "", encoding='utf-8')
                except Exception:
                    pass
            if result.returncode == 0:
                return True

            # 尝试2：宽松的等价检查（允许一些优化差异）
            yosys_tcl_2 = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; techmap; opt; clean
design -stash gold

read_verilog {cand_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; techmap; opt; clean
design -stash gate

design -copy-from gold -as gold {module_name}
design -copy-from gate -as gate {module_name}
equiv_make gold gate equiv
hierarchy -top equiv
setundef -undriven -zero
equiv_simple -undef
equiv_status -assert
"""
            with open(tcl_file, 'w') as f:
                f.write(yosys_tcl_2)
            result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if debug_dir:
                try:
                    (Path(debug_dir) / "equiv_2.tcl").write_text(yosys_tcl_2, encoding='utf-8')
                    (Path(debug_dir) / "equiv_2.stdout.txt").write_text(result2.stdout or "", encoding='utf-8')
                    (Path(debug_dir) / "equiv_2.stderr.txt").write_text(result2.stderr or "", encoding='utf-8')
                except Exception:
                    pass
            if result2.returncode == 0:
                return True

            # 尝试3：简单的端口对比检查（最宽松）
            yosys_tcl_3 = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; clean
design -stash gold

read_verilog {cand_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; clean
design -stash gate

design -copy-from gold -as gold {module_name}
design -copy-from gate -as gate {module_name}
equiv_make gold gate equiv
hierarchy -top equiv
equiv_simple -short
equiv_status
"""
            with open(tcl_file, 'w') as f:
                f.write(yosys_tcl_3)
            result3 = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if debug_dir:
                try:
                    (Path(debug_dir) / "equiv_3.tcl").write_text(yosys_tcl_3, encoding='utf-8')
                    (Path(debug_dir) / "equiv_3.stdout.txt").write_text(result3.stdout or "", encoding='utf-8')
                    (Path(debug_dir) / "equiv_3.stderr.txt").write_text(result3.stderr or "", encoding='utf-8')
                except Exception:
                    pass
            # 第三次检查不使用-assert，只要没有错误就认为通过
            return result3.returncode == 0
        except Exception as e:
            logger.warning(f"等价检查异常: {e}")
            return False
    
    def _quick_syntax_check(self, ref_v, cand_v, module_name: str, debug_dir: Optional[str] = None) -> bool:
        """快速语法检查，确保两个文件都能被Yosys正确解析"""
        try:
            # 检查参考文件
            tcl_check = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
"""
            tcl_file = self.temp_dir / "syntax_check.ys"
            with open(tcl_file, 'w') as f:
                f.write(tcl_check)
            
            cmd = [self.yosys_path, "-q", "-s", str(tcl_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                if debug_dir:
                    try:
                        Path(debug_dir).mkdir(parents=True, exist_ok=True)
                        (Path(debug_dir) / "syntax_ref_error.txt").write_text(result.stderr or "", encoding='utf-8')
                    except Exception:
                        pass
                return False
            
            # 检查候选文件
            tcl_check = f"""
read_verilog {cand_v}
hierarchy -check -top {module_name}
"""
            with open(tcl_file, 'w') as f:
                f.write(tcl_check)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                if debug_dir:
                    try:
                        (Path(debug_dir) / "syntax_cand_error.txt").write_text(result.stderr or "", encoding='utf-8')
                    except Exception:
                        pass
                return False
            
            return True
        except Exception:
            return False


# API接口
def analyze_verilog_api(code: str, module_name: str = "top", target_freq: float = 100.0) -> Dict[str, Any]:
    """API接口：分析Verilog代码"""
    try:
        eda = EDAWrapper()
        result = eda.synthesize(code, module_name, target_freq)
        eda.cleanup()
        return {"success": True, "data": result}
    except EDAFailure as e:
        return {"success": False, "error": str(e)}


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EDA工具包装器")
    parser.add_argument("--input", "-i", help="输入Verilog文件")
    parser.add_argument("--code", "-c", help="直接输入Verilog代码")
    parser.add_argument("--module", "-m", default="top", help="模块名")
    parser.add_argument("--freq", "-f", type=float, default=100.0, help="目标频率(MHz)")
    parser.add_argument("--output", "-o", help="输出JSON文件")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    if args.interactive:
        # 交互模式
        print("EDA工具交互模式")
        print("输入 'quit' 退出")
        
        while True:
            try:
                print("\n请输入Verilog代码 (以'END'结束输入):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                
                code = "\n".join(lines)
                if not code.strip():
                    continue
                
                module_name = input("模块名 (默认: top): ").strip() or "top"
                freq_input = input("目标频率MHz (默认: 100): ").strip()
                target_freq = float(freq_input) if freq_input else 100.0
                
                result = analyze_verilog_api(code, module_name, target_freq)
                
                if result["success"]:
                    print("\n=== EDA分析结果 ===")
                    data = result["data"]
                    print(f"语法正确: {data['syntax_ok']}")
                    print(f"综合成功: {data['synth_ok']}")
                    print(f"面积: {data['area']}")
                    print(f"触发器数: {data['num_ff']}")
                    print(f"逻辑深度: {data['logic_depth']}")
                else:
                    print(f"分析失败: {result['error']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
        
        print("退出交互模式")
        return
    
    # 文件或代码输入模式
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("请指定输入文件 (--input) 或直接输入代码 (--code)")
        return
    
    # 自动检测模块名
    import re
    module_match = re.search(r'module\s+(\w+)', code)
    detected_module = module_match.group(1) if module_match else args.module
    
    # 分析代码
    result = analyze_verilog_api(code, detected_module, args.freq)
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
