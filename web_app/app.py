from pathlib import Path
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from loguru import logger as log

from core_tools.eda_wrapper import analyze_verilog_api
from optimization.rl_optimizer import RLOptimizer


class OptimizationManager:
    """负责协调EDA分析与RL优化"""

    def __init__(self):
        self._optimizer = None

    def ensure_optimizer(self, model_path: str):
        if self._optimizer is None:
            log.info(f"[Manager] 初始化优化器: {model_path}")
            self._optimizer = RLOptimizer(
                model_path=model_path,
                max_iterations=5,
                population_size=1,
                temperature=0.8,
                max_new_tokens=1024,
                debug_gen=False,
                rl_mode=True,
            )
        return self._optimizer

    def optimize(self, code: str, target: str, model_path: str):
        """直接执行优化并返回结果（同步）"""
        log.info(f"[Manager] 开始优化任务")
        
        try:
            optimizer = self.ensure_optimizer(model_path)
            result = optimizer.optimize(code, target)
            log.info(f"[Manager] 优化完成，success={result.get('success')}")
            return result
        except Exception as exc:
            log.error(f"[Manager] 优化异常: {exc}")
            import traceback
            log.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(exc),
            }

    def cleanup(self):
        if self._optimizer is not None:
            self._optimizer.cleanup()


def get_default_model_path() -> str:
    models_dir = Path("models")
    candidates = [p for p in models_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("models目录下未找到可用模型，请先训练或放置模型")
    # 选择最新目录
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def create_app():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    
    # 启用 CORS（解决 Docker 环境跨域问题）
    CORS(app)
    
    # 配置日志
    log.add("logs/web_app.log", rotation="10 MB", retention="7 days")

    manager = OptimizationManager()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        module_name = payload.get("module", "top")
        if not code.strip():
            return jsonify({"success": False, "error": "请输入Verilog代码"}), 400

        result = analyze_verilog_api(code, module_name)
        return jsonify(result)

    @app.route("/api/optimize", methods=["POST"])
    def api_optimize():
        """直接执行优化并返回结果（同步）"""
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        target = payload.get("target", "")
        
        if not code.strip():
            return jsonify({"success": False, "error": "请输入待优化的Verilog代码"}), 400

        log.info("[API] 收到优化请求，开始执行...")
        result = manager.optimize(code, target, get_default_model_path())
        log.info(f"[API] 优化完成，返回结果: success={result.get('success')}")
        
        return jsonify(result)

    @app.teardown_appcontext
    def shutdown_session(exception=None):  # noqa: ANN001
        manager.cleanup()
        return exception

    return app


def main():
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
