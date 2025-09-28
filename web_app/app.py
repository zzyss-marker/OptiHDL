from pathlib import Path
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify

from core_tools.eda_wrapper import analyze_verilog_api
from optimization.rl_optimizer import RLOptimizer


class OptimizerThread(threading.Thread):
    def __init__(self, code: str, target: str, manager: "OptimizationManager"):
        super().__init__(daemon=True)
        self.code = code
        self.target = target
        self.manager = manager

    def run(self):
        try:
            result = self.manager.run_optimizer(self.code, self.target)
            self.manager.update_result(result)
        except Exception as exc:  # noqa: BLE001
            self.manager.update_result({
                "success": False,
                "error": str(exc),
            })


class OptimizationManager:
    """负责协调EDA分析与RL优化，维护状态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._optimizer = None
        self._latest_result = None
        self._running = False

    def ensure_optimizer(self, model_path: str):
        if self._optimizer is None:
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

    def start_optimization(self, code: str, target: str, model_path: str):
        with self._lock:
            if self._running:
                raise RuntimeError("已有优化任务在运行，请稍候")
            self._running = True
            self._latest_result = None

        optimizer = self.ensure_optimizer(model_path)

        thread = OptimizerThread(code, target, self)
        thread.start()
        return thread

    def run_optimizer(self, code: str, target: str):
        optimizer = self.ensure_optimizer(get_default_model_path())
        return optimizer.optimize(code, target)

    def update_result(self, result: dict):
        with self._lock:
            self._latest_result = result
            self._running = False

    def get_status(self):
        with self._lock:
            return {
                "running": self._running,
                "result": self._latest_result,
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
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        target = payload.get("target", "")
        if not code.strip():
            return jsonify({"success": False, "error": "请输入待优化的Verilog代码"}), 400

        try:
            manager.start_optimization(code, target, get_default_model_path())
        except Exception as exc:  # noqa: BLE001
            return jsonify({"success": False, "error": str(exc)}), 400

        return jsonify({"success": True})

    @app.route("/api/optimize/status", methods=["GET"])
    def api_optimize_status():
        status = manager.get_status()
        return jsonify(status)

    @app.route("/api/optimize/result", methods=["GET"])
    def api_optimize_result():
        status = manager.get_status()
        if status["result"] is None:
            return jsonify({"success": False, "error": "暂无结果"}), 404
        return jsonify(status["result"])

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
