from pathlib import Path
import os
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from loguru import logger as log

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_tools.eda_wrapper import analyze_verilog_api
from optimization.rl_optimizer import RLOptimizer

SETTINGS_FILE = PROJECT_ROOT / "web_app" / "runtime_settings.json"


def _default_runtime_settings() -> dict:
    return {
        "llm_mode": os.environ.get("OPTIHDL_LLM_MODE", "auto"),
        "api_base_url": os.environ.get("OPTIHDL_API_BASE_URL", ""),
        "api_key": os.environ.get("OPTIHDL_API_KEY", ""),
        "api_model": os.environ.get("OPTIHDL_API_MODEL", ""),
        "local_model_path": "",
    }


def load_runtime_settings() -> dict:
    settings = _default_runtime_settings()
    if SETTINGS_FILE.exists():
        try:
            import json
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update(data)
        except Exception:
            log.warning("[Settings] failed to load runtime_settings.json, using defaults")
    return settings


def save_runtime_settings(settings: dict) -> dict:
    merged = _default_runtime_settings()
    merged.update({key: value for key, value in settings.items() if key in merged})
    import json
    SETTINGS_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return merged


def apply_runtime_settings(settings: dict) -> None:
    os.environ["OPTIHDL_LLM_MODE"] = str(settings.get("llm_mode", "auto") or "auto")
    os.environ["OPTIHDL_API_BASE_URL"] = str(settings.get("api_base_url", "") or "")
    os.environ["OPTIHDL_API_KEY"] = str(settings.get("api_key", "") or "")
    os.environ["OPTIHDL_API_MODEL"] = str(settings.get("api_model", "") or "")


class OptimizationManager:
    def __init__(self) -> None:
        self._optimizer = None
        self._model_path = None
        self._config_signature = None

    def ensure_optimizer(self, model_path: str, runtime_settings: dict) -> RLOptimizer:
        config_signature = (
            runtime_settings.get("llm_mode", "auto"),
            runtime_settings.get("api_base_url", ""),
            runtime_settings.get("api_key", ""),
            runtime_settings.get("api_model", ""),
            runtime_settings.get("local_model_path", ""),
        )
        if self._optimizer is None or self._model_path != model_path or self._config_signature != config_signature:
            if self._optimizer is not None:
                self._optimizer.cleanup()
            log.info(f"[Manager] initialize optimizer: {model_path}")
            apply_runtime_settings(runtime_settings)
            self._optimizer = RLOptimizer(
                model_path=model_path,
                max_iterations=5,
                population_size=1,
                temperature=0.8,
                max_new_tokens=1024,
                debug_gen=False,
                rl_mode=True,
            )
            self._model_path = model_path
            self._config_signature = config_signature
        return self._optimizer

    def optimize(self, code: str, target: str, scenario: str, model_path: str, runtime_settings: dict):
        try:
            optimizer = self.ensure_optimizer(model_path, runtime_settings)
            return optimizer.optimize(code, target, scenario)
        except Exception as exc:
            log.exception("[Manager] optimization failed")
            return {"success": False, "error": str(exc)}

    def cleanup(self) -> None:
        if self._optimizer is not None:
            self._optimizer.cleanup()
            self._optimizer = None
            self._model_path = None
            self._config_signature = None


def get_default_model_path(runtime_settings: dict | None = None) -> str:
    runtime_settings = runtime_settings or load_runtime_settings()
    llm_mode = str(runtime_settings.get("llm_mode", os.environ.get("OPTIHDL_LLM_MODE", "auto"))).strip().lower()
    local_model_path = str(runtime_settings.get("local_model_path", "") or "").strip()

    if local_model_path and llm_mode != "api":
        return local_model_path

    if llm_mode == "api":
        return "api-client"

    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("models directory not found")

    candidates = [path for path in models_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError("no available model found under models/")

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return str(candidates[0])


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    CORS(app)
    log.add("logs/web_app.log", rotation="10 MB", retention="7 days")

    manager = OptimizationManager()
    runtime_settings = load_runtime_settings()
    apply_runtime_settings(runtime_settings)
    jobs: dict[str, dict] = {}

    def start_job(job_type: str, target_fn, *args):
        job_id = uuid.uuid4().hex
        jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
        }

        def runner():
            jobs[job_id]["status"] = "running"
            jobs[job_id]["updated_at"] = time.time()
            try:
                result = target_fn(*args)
                jobs[job_id]["result"] = result
                jobs[job_id]["status"] = "completed"
            except Exception as exc:
                log.exception(f"[Job] {job_type} failed")
                jobs[job_id]["error"] = str(exc)
                jobs[job_id]["status"] = "failed"
            finally:
                jobs[job_id]["updated_at"] = time.time()

        threading.Thread(target=runner, daemon=True).start()
        return job_id

    def job_response(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify({"success": True, "job": job})

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/settings", methods=["GET"])
    def api_get_settings():
        return jsonify({"success": True, "settings": runtime_settings})

    @app.route("/api/settings", methods=["POST"])
    def api_save_settings():
        payload = request.get_json(force=True) or {}
        settings_payload = payload.get("settings", payload)
        saved_settings = save_runtime_settings(settings_payload)
        runtime_settings.clear()
        runtime_settings.update(saved_settings)
        apply_runtime_settings(runtime_settings)
        manager.cleanup()
        return jsonify({"success": True, "settings": runtime_settings})

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        module_name = payload.get("module", "top")
        if not code.strip():
            return jsonify({"success": False, "error": "Please input Verilog code"}), 400
        return jsonify(analyze_verilog_api(code, module_name))

    @app.route("/api/analyze_async", methods=["POST"])
    def api_analyze_async():
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        module_name = payload.get("module", "top")
        if not code.strip():
            return jsonify({"success": False, "error": "Please input Verilog code"}), 400
        job_id = start_job("analyze", analyze_verilog_api, code, module_name)
        return jsonify({"success": True, "job_id": job_id})

    @app.route("/api/optimize", methods=["POST"])
    def api_optimize():
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        target = payload.get("target", "")
        scenario = payload.get("scenario", "")

        if not code.strip():
            return jsonify({"success": False, "error": "Please input Verilog code to optimize"}), 400

        result = manager.optimize(
            code,
            target,
            scenario,
            get_default_model_path(runtime_settings),
            runtime_settings,
        )
        return jsonify(result)

    @app.route("/api/optimize_async", methods=["POST"])
    def api_optimize_async():
        payload = request.get_json(force=True)
        code = payload.get("code", "")
        target = payload.get("target", "")
        scenario = payload.get("scenario", "")

        if not code.strip():
            return jsonify({"success": False, "error": "Please input Verilog code to optimize"}), 400

        model_path = get_default_model_path(runtime_settings)
        job_id = start_job(
            "optimize",
            manager.optimize,
            code,
            target,
            scenario,
            model_path,
            runtime_settings,
        )
        return jsonify({"success": True, "job_id": job_id})

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def api_job_status(job_id: str):
        return job_response(job_id)

    @app.errorhandler(Exception)
    def handle_api_exception(exc):  # noqa: ANN001
        if request.path.startswith("/api/"):
            log.exception("[API] unhandled exception")
            return jsonify({"success": False, "error": str(exc)}), 500
        raise exc

    @app.teardown_appcontext
    def shutdown_session(exception=None):  # noqa: ANN001
        manager.cleanup()
        return exception

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
