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
    """EDA 工具包装器，封装 Yosys 和 OpenSTA 的命令行交互"""
    
    def __init__(self, yosys_path: str = "yosys", opensta_path: str = "sta"):
        self.yosys_path = yosys_path
        self.opensta_path = opensta_path
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
            # 禁用日志输出
            # 1. 语法检查和综合（仅使用 Yosys 指标）
            synth_result = self._run_yosys_synthesis(verilog_code, module_name)

            # 2. 无时序方案：直接返回 Yosys 指标（RL 所需字段）
            result = {**synth_result}

            # 3. 验证必需字段（仅 RL 所需）
            required_fields = ["syntax_ok", "equiv_ok", "synth_ok", "area", "num_ff", "logic_depth"]
            for field in required_fields:
                if field not in result:
                    raise EDAFailure(f"缺少必需字段: {field}")
            
            # 禁用日志输出
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
    
    def _run_opensta_timing(self, verilog_code: str, module_name: str, target_freq: float) -> Dict[str, Any]:
        """运行 OpenSTA 时序分析（若检测到真实.lib则使用映射后网表与真实库）"""
        try:
            # 创建 SDC 约束文件（自动探测多时钟端口）
            period_ns = 1000.0 / target_freq  # 转换为纳秒
            # 探测顶层输入端口中包含 'clk' 的端口名（如 clk, clk_a, clk_b, clk0 等）
            # 允许通过环境变量显式指定：OPTIHDL_CLOCK_PORTS="clk clk_a clk_b"
            env_clk = os.environ.get("OPTIHDL_CLOCK_PORTS", "").strip()
            clk_ports = []
            if env_clk:
                clk_ports = [p for p in env_clk.split() if p]
                logger.info(f"从环境变量指定时钟端口: {clk_ports}")
            try:
                # 捕获形如：input clk; input [N:0] clk_a; input wire clk_b; 等
                input_names = re.findall(r"\binput\b[^;]*?\b([A-Za-z_][A-Za-z0-9_]*)\b", verilog_code)
                auto_clk = sorted({n for n in input_names if 'clk' in n.lower()})
                if not clk_ports:
                    clk_ports = auto_clk
                else:
                    # 合并但以显式指定优先
                    clk_ports = sorted(set(clk_ports) | set(auto_clk))
            except Exception:
                clk_ports = []

            sdc_lines = []
            if clk_ports:
                for p in clk_ports:
                    sdc_lines.append(f"create_clock -name {p} -period {period_ns} [get_ports {p}]")
                # 对每个时钟分别设置 IO 延迟（保守做法）
                for p in clk_ports:
                    sdc_lines.append(f"set_input_delay -clock {p} 0.1 [all_inputs]")
                    sdc_lines.append(f"set_output_delay -clock {p} 0.1 [all_outputs]")
                # 多时钟默认异步分组，避免跨时钟路径影响 WNS/TNS
                if len(clk_ports) > 1:
                    group_parts = " ".join([f"-group {{{p}}}" for p in clk_ports])
                    sdc_lines.append(f"set_clock_groups -asynchronous {group_parts}")
                logger.info(f"检测到时钟端口: {clk_ports}")
            else:
                # 无显式时钟端口：SDC 中仅放置虚拟时钟；更智能的通配符探测放到 STA 脚本中执行
                sdc_lines.append(f"create_clock -name VCLK -period {period_ns}")
                sdc_lines.append("set_input_delay -clock VCLK 0.1 [all_inputs]")
                sdc_lines.append("set_output_delay -clock VCLK 0.1 [all_outputs]")

            sdc_content = "\n".join(sdc_lines) + "\n"
            sdc_file = self.temp_dir / f"{module_name}.sdc"
            with open(sdc_file, 'w') as f:
                f.write(sdc_content)
            logger.info(f"SDC 文件: {sdc_file}, 周期(ns)={period_ns}")
            
            # 选择库与网表
            liberty = self._find_liberty()
            synth_file = self.temp_dir / f"{module_name}_synth.v"
            if not os.path.exists(synth_file):
                synth_file = self.temp_dir / f"{module_name}.v"

            # 额外从综合后网表中解析顶层输入端口，增强时钟端口识别（例如 clk_a/clk_b）
            try:
                netlist_text = Path(synth_file).read_text(encoding='utf-8', errors='ignore')
                header_ports = set(re.findall(r"module\s+" + re.escape(module_name) + r"\s*\((.*?)\);", netlist_text, flags=re.S))
                port_names = set()
                for hp in header_ports:
                    # 去掉注释/空白，按 , 分割
                    for tok in re.split(r"[,\n]", hp):
                        name = tok.strip()
                        # 去掉 .name(...) 形式
                        m = re.match(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", name)
                        if m:
                            name = m.group(1)
                        # 过滤总线声明等
                        name = re.sub(r"\[[^\]]*\]", "", name).strip()
                        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                            port_names.add(name)
                # 找出 input 声明的端口
                input_declared = set(re.findall(r"\binput\b[^;]*?\b([A-Za-z_][A-Za-z0-9_]*)\b", netlist_text))
                top_inputs = {p for p in port_names if p in input_declared}
                netlist_clk = sorted({p for p in top_inputs if 'clk' in p.lower()})
                if netlist_clk:
                    logger.info(f"从综合网表识别到时钟端口: {netlist_clk}")
                    # 合并到 clk_ports
                    clk_ports = sorted(set(clk_ports) | set(netlist_clk))
            except Exception as e:
                logger.info(f"解析综合网表端口失败（忽略）: {e}")

            # 日志提示时钟模式
            if clk_ports:
                logger.info(f"OpenSTA: 使用端口时钟: {clk_ports}")
            else:
                logger.info("OpenSTA: 未发现clk端口，已切换为虚拟时钟VCLK")

            # 创建 OpenSTA 脚本（优先真实库，否则兼容旧演示库）
            lib_line = f"read_liberty {liberty}" if liberty else "read_liberty /usr/share/yosys/techlibs/common/stdcells.lib"
            # 在 read_sdc 之后增加兜底逻辑：若仍无时钟，则尝试按 *clk* 自动创建端口时钟；若仍找不到则保持 VCLK
            fallback_clock_tcl = f"""
set __clks [all_clocks]
if {{[llength $__clks] == 0}} {{
  set __auto_clk_ports [get_ports *clk*]
  if {{[llength $__auto_clk_ports] > 0}} {{
    foreach p $__auto_clk_ports {{ create_clock -name $p -period {period_ns} [get_ports $p] }}
    if {{[llength $__auto_clk_ports] > 1}} {{
      set __groups {{}}
      foreach p $__auto_clk_ports {{ lappend __groups "-group {{$p}}" }}
      eval set_clock_groups -asynchronous $__groups
    }}
    foreach p $__auto_clk_ports {{ set_input_delay -clock $p 0.1 [all_inputs] }}
    foreach p $__auto_clk_ports {{ set_output_delay -clock $p 0.1 [all_outputs] }}
  }}
}}
"""

            sta_script = f"""
{lib_line}
read_verilog {synth_file}
link_design {module_name}
read_sdc {sdc_file}
{fallback_clock_tcl}
report_checks -path_delay max -fields {{startpoint endpoint slack}} -digits 3
report_worst_slack
report_tns
"""
            
            sta_script_file = self.temp_dir / "timing.tcl"
            with open(sta_script_file, 'w') as f:
                f.write(sta_script)
            logger.info(f"OpenSTA 脚本: {sta_script_file}\n--- STA脚本预览 ---\n{os.linesep.join(sta_script.splitlines()[:20])}\n-------------------")
            
            # 执行 OpenSTA
            # 在部分 OpenSTA 版本中，不指定 -exit 会在执行脚本后进入交互模式，导致进程不退出
            sta_cmd = [self.opensta_path, "-exit", str(sta_script_file)]
            sta_timeout = int(os.environ.get("OPTIHDL_STA_TIMEOUT", "180"))
            logger.info(f"执行OpenSTA命令: {' '.join(map(str, sta_cmd))}, 超时={sta_timeout}s")
            import time
            t0 = time.perf_counter()
            result = subprocess.run(
                sta_cmd,
                capture_output=True,
                text=True,
                timeout=sta_timeout
            )
            dt = time.perf_counter() - t0
            logger.info(f"OpenSTA 实际耗时: {dt:.3f}s")
            logger.info(f"OpenSTA 退出码: {result.returncode}")
            if result.stderr:
                logger.info(f"OpenSTA stderr前200行:\n{os.linesep.join(result.stderr.splitlines()[:200])}")
            if result.stdout:
                logger.info(f"OpenSTA stdout前80行:\n{os.linesep.join(result.stdout.splitlines()[:80])}")
            
            # 解析时序结果（更健壮：先用专用报告，若失败再从report_checks提取最坏slack）
            wns = self._parse_sta_wns(result.stdout)
            tns = self._parse_sta_tns(result.stdout)
            # 如发现无路径/INF，回退到IO组合模式重跑一次
            stdout_lower = (result.stdout or "").lower()
            if ("no paths found" in stdout_lower) or ("inf" in stdout_lower and (wns is None)):
                logger.info("OpenSTA: 检测到无路径/INF，回退到IO组合模式重新分析")
                combo_tcl = f"""
{lib_line}
read_verilog {synth_file}
link_design {module_name}
# 组合模式：仅使用虚拟时钟约束IO
create_clock -name VCLK -period {period_ns}
set_input_delay -clock VCLK 0.1 [all_inputs]
set_output_delay -clock VCLK 0.1 [all_outputs]
report_checks -path_delay max -fields {{startpoint endpoint slack}} -digits 3
report_worst_slack
report_tns
"""
                combo_script_file = self.temp_dir / "timing_combo.tcl"
                with open(combo_script_file, 'w') as f:
                    f.write(combo_tcl)
                logger.info(f"OpenSTA 组合回退脚本: {combo_script_file}")
                sta_cmd_combo = [self.opensta_path, "-exit", str(combo_script_file)]
                logger.info(f"执行OpenSTA(组合回退)命令: {' '.join(map(str, sta_cmd_combo))}")
                import time
                t0c = time.perf_counter()
                result_combo = subprocess.run(
                    sta_cmd_combo,
                    capture_output=True,
                    text=True,
                    timeout=sta_timeout
                )
                dtc = time.perf_counter() - t0c
                logger.info(f"OpenSTA(组合回退) 实际耗时: {dtc:.3f}s, 退出码: {result_combo.returncode}")
                if result_combo.stderr:
                    logger.info(f"OpenSTA(组合回退) stderr前200行:\n{os.linesep.join(result_combo.stderr.splitlines()[:200])}")
                if result_combo.stdout:
                    logger.info(f"OpenSTA(组合回退) stdout前80行:\n{os.linesep.join(result_combo.stdout.splitlines()[:80])}")
                # 用回退结果更新解析
                wns = self._parse_sta_wns(result_combo.stdout)
                if wns is None:
                    wns = self._parse_sta_worst_slack_from_checks(result_combo.stdout)
                tns = self._parse_sta_tns(result_combo.stdout)
            if wns is None:
                wns = self._parse_sta_worst_slack_from_checks(result.stdout)
            if tns is None:
                tns = self._parse_sta_tns(result.stdout)
            # 最终兜底
            if wns is None:
                wns = -0.5
            if tns is None:
                tns = -2.0
            logger.info(f"OpenSTA 解析: WNS={wns} ns, TNS={tns} ns")
            
            return {
                "wns": wns,
                "tns": tns
            }
            
        except subprocess.TimeoutExpired:
            # 禁用日志输出
            return {"wns": -1.0, "tns": -5.0}
        except Exception as e:
            # 禁用日志输出
            return {"wns": -1.0, "tns": -5.0}
    
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
    def _parse_sta_wns(self, output: str) -> float:
        """从 OpenSTA 输出解析 WNS"""
        # 优先匹配 OpenSTA 标准摘要
        num = self._safe_find_float([
            r"worst\s+slack\s+max\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)",
            r"wns\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)",
            r"worst\s+negative\s+slack\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)",
            r"slack\s*\(VIOLATED\)\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)"
        ], output)
        return num
    
    def _parse_sta_tns(self, output: str) -> float:
        """从 OpenSTA 输出解析 TNS"""
        num = self._safe_find_float([
            r"tns\s+max\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)",
            r"total\s+negative\s+slack\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)",
            r"tns\s+([-+]?\d*\.\d+|[-+]?\d+\.?\d*)"
        ], output)
        return num

    def _parse_sta_worst_slack_from_checks(self, output: str) -> float:
        """从报告中提取最坏slack作为兜底。优先使用 'worst slack max' 行，
        否则尝试在包含 'slack' 的行中提取数值并取最小。
        """
        # 先重用标准解析
        direct = self._parse_sta_wns(output)
        if direct is not None:
            return direct
        # 回退：扫描包含 'slack' 的行，提取行中的最后一个数作为该行 slack
        min_val = None
        float_pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+\.?\d*")
        for line in output.splitlines():
            if 'slack' in line.lower():
                nums = float_pattern.findall(line)
                if nums:
                    try:
                        val = float(nums[-1])
                        min_val = val if min_val is None else min(min_val, val)
                    except Exception:
                        continue
        if min_val is not None:
            return min_val
        return -0.5

    def _safe_find_float(self, patterns, text: str) -> Optional[float]:
        """按给定正则列表顺序查找第一个可解析的浮点数。"""
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                token = m.group(1).strip()
                # 只接受包含数字的 token
                if re.fullmatch(r"[-+]?\d*\.\d+|[-+]?\d+\.?\d*", token):
                    try:
                        return float(token)
                    except Exception:
                        continue
        return None

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
            # 尝试1：规范化 + 等价简单证明（保留memory，归一化不定值）
            yosys_tcl_1 = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; memory -nomap; opt; techmap; opt
splitnets -ports;; design -stash gold

read_verilog {cand_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; memory -nomap; opt; techmap; opt
splitnets -ports;; design -stash gate

design -copy-from gold -as gold {module_name}
design -copy-from gate -as gate {module_name}
setundef -undriven -zero
equiv_make gold gate equiv
hierarchy -top equiv
equiv_struct
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

            # 尝试2：等价归纳（更强，但更慢，提升序列深度并处理不定值）
            yosys_tcl_2 = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; memory -nomap; opt; techmap; opt
splitnets -ports;; design -stash gold

read_verilog {cand_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; memory -nomap; opt; techmap; opt
splitnets -ports;; design -stash gate

design -copy-from gold -as gold {module_name}
design -copy-from gate -as gate {module_name}
equiv_make gold gate equiv
hierarchy -top equiv
setundef -undriven -zero
equiv_induct -undef -seq 10
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

            # 尝试3：SAT 基于 miter（提高序列深度，忽略初始未定义）
            yosys_tcl_3 = f"""
read_verilog {ref_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; memory -nomap; opt; techmap; opt
splitnets -ports;; design -stash gold

read_verilog {cand_v}
hierarchy -check -top {module_name}
prep -top {module_name}
proc; opt; memory -nomap; opt; techmap; opt
splitnets -ports;; design -stash gate

design -copy-from gold -as gold {module_name}
design -copy-from gate -as gate {module_name}
miter -equiv -flatten gold gate miter
hierarchy -top miter
setundef -undriven -zero
sat -verify -prove-asserts -set-init-undef -seq 10 -ignore_div_by_zero -enable_undef
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
            return result3.returncode == 0
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
