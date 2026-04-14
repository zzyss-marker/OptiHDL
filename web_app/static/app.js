const btnAnalyze = document.getElementById("btn-analyze");
const btnOptimize = document.getElementById("btn-optimize");
const btnSaveSettings = document.getElementById("btn-save-settings");
const inputArea = document.getElementById("verilog-input");
const targetInput = document.getElementById("target-input");
const scenarioInput = document.getElementById("scenario-input");
const llmModeInput = document.getElementById("llm-mode-input");
const localModelInput = document.getElementById("local-model-input");
const apiBaseUrlInput = document.getElementById("api-base-url-input");
const apiModelInput = document.getElementById("api-model-input");
const apiKeyInput = document.getElementById("api-key-input");
const statusBox = document.getElementById("global-status");
const edaOutput = document.getElementById("eda-output");
const originalCode = document.getElementById("original-code");
const optimizedCode = document.getElementById("optimized-code");
const metricsTable = document.getElementById("metrics-table").querySelector("tbody");
const logContainer = document.getElementById("log-container");
const toggleOriginalBtn = document.getElementById("toggle-original");
const agentOutput = document.getElementById("agent-output");

function setStatus(message, type = "info") {
    statusBox.textContent = message;
    statusBox.dataset.type = type;
    statusBox.className = `status ${type}`;
}

function addLogEntry(message, type = "info") {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement("div");
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = `[${timestamp}] ${message}`;
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;

    const entries = logContainer.querySelectorAll(".log-entry");
    if (entries.length > 100) {
        entries[0].remove();
    }
}

function toggleCodeCollapse() {
    originalCode.classList.toggle("collapsed");
}

function extractModuleName(code) {
    const match = code.match(/module\s+(\w+)/);
    return match ? match[1] : "top";
}

function applySettingsToForm(settings = {}) {
    llmModeInput.value = settings.llm_mode || "auto";
    localModelInput.value = settings.local_model_path || "";
    apiBaseUrlInput.value = settings.api_base_url || "";
    apiModelInput.value = settings.api_model || "";
    apiKeyInput.value = settings.api_key || "";
}

async function loadSettings() {
    try {
        const response = await fetch("/api/settings");
        const result = await response.json();
        if (result.success) {
            applySettingsToForm(result.settings || {});
            addLogEntry("已加载运行时模型配置", "info");
        }
    } catch (err) {
        addLogEntry(`加载配置失败: ${err.message}`, "warning");
    }
}

async function saveSettings() {
    const settings = {
        llm_mode: llmModeInput.value,
        local_model_path: localModelInput.value.trim(),
        api_base_url: apiBaseUrlInput.value.trim(),
        api_model: apiModelInput.value.trim(),
        api_key: apiKeyInput.value.trim(),
    };

    btnSaveSettings.disabled = true;
    try {
        const response = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ settings }),
        });
        const result = await response.json();
        if (result.success) {
            applySettingsToForm(result.settings || settings);
            setStatus(`模型配置已保存，当前模式 ${result.settings?.llm_mode || settings.llm_mode}`, "success");
            addLogEntry("运行时模型配置已保存", "success");
        } else {
            setStatus(result.error || "保存配置失败", "error");
            addLogEntry(`保存配置失败: ${result.error || "未知错误"}`, "error");
        }
    } catch (err) {
        setStatus(`保存配置失败: ${err.message}`, "error");
        addLogEntry(`保存配置失败: ${err.message}`, "error");
    } finally {
        btnSaveSettings.disabled = false;
    }
}

function formatEDAResult(result) {
    if (!result.success) {
        return `分析失败\n原因: ${result.error || "未知错误"}`;
    }

    const data = result.data || {};
    return [
        "EDA 分析成功",
        `语法检查: ${data.syntax_ok ? "通过" : "失败"}`,
        `综合状态: ${data.synth_ok ? "通过" : "失败"}`,
        `面积 (cells): ${data.area}`,
        `触发器数量: ${data.num_ff}`,
        `逻辑深度: ${data.logic_depth}`,
    ].join("\n");
}

function renderMetrics(result) {
    const original = result.original_metrics || {};
    const optimized = result.optimized_metrics || {};
    const improvement = result.improvement || {};
    const rows = [
        { key: "area", label: "面积 (cells)" },
        { key: "num_ff", label: "触发器数量" },
        { key: "logic_depth", label: "逻辑深度" },
        { key: "score_improvement", label: "综合得分改进%", isImprovementOnly: true },
    ];

    metricsTable.innerHTML = rows.map(({ key, label, isImprovementOnly }) => {
        const origVal = original[key];
        const optVal = optimized[key];
        const impVal = improvement[`${key}_improvement`] ?? (isImprovementOnly ? improvement[key] : undefined);
        const renderedImp = impVal !== undefined ? `${Number(impVal).toFixed(2)}%` : "--";
        return `
            <tr>
                <td>${label}</td>
                <td>${origVal !== undefined ? origVal : "--"}</td>
                <td>${optVal !== undefined ? optVal : "--"}</td>
                <td>${renderedImp}</td>
            </tr>
        `;
    }).join("");
}

function renderAgentDecisions(result) {
    const decisions = result.agent_decisions || [];
    if (!decisions.length) {
        agentOutput.textContent = "暂无 Agent 决策记录";
        return;
    }

    agentOutput.textContent = decisions.map((decision, index) => {
        const focus = decision.focus ? `\nfocus: ${decision.focus}` : "";
        return `[Judge] round ${index + 1}\naction: ${decision.action}\nreason: ${decision.reason}${focus}`;
    }).join("\n\n");
}

function renderProgress(progress) {
    if (!progress) return;

    // Drive pipeline visualization
    updatePipeline(progress);

    const phase = progress.phase || "";
    const iter = progress.iteration || 0;
    const maxIter = progress.max_iterations || "?";
    const bestScore = progress.best_score;
    const initScore = progress.initial_score;
    const message = progress.message || "";
    const strategy = progress.strategy || "";

    // Update status bar with detailed progress
    if (iter > 0) {
        let scoreInfo = "";
        if (bestScore !== undefined && initScore !== undefined) {
            const delta = ((bestScore - initScore) / Math.max(1e-9, Math.abs(initScore)) * 100).toFixed(2);
            scoreInfo = `  |  得分 ${bestScore.toFixed(4)} (${delta >= 0 ? "+" : ""}${delta}%)`;
        }
        const strategyInfo = strategy ? `  |  策略: ${strategy}` : "";
        setStatus(`优化中: ${iter}/${maxIter} 轮${scoreInfo}${strategyInfo}`, "info");
    } else if (message) {
        setStatus(message, "info");
    }

    // Show plan summary when available
    if (progress.plan_summary) {
        const ps = progress.plan_summary;
        addLogEntry(
            `[Planner] 复杂度=${ps.complexity}, 策略=${ps.strategy}, 瓶颈=${(ps.bottlenecks || []).join("/")}`,
            "info"
        );
    }

    // Log iteration phase messages
    if (phase === "generating" && iter > 0) {
        addLogEntry(`[Coder] 第 ${iter} 轮生成候选方案… (策略: ${strategy || "balanced"})`, "info");
    }

    // Show decisions incrementally
    if (phase === "iteration_done") {
        const decision = progress.decision;
        if (decision) {
            const tag = decision.action === "stop" ? "warning" : "success";
            addLogEntry(
                `[Judge] 第 ${iter} 轮: ${decision.action} — ${decision.reason}`,
                tag
            );
        }

        // Update agent output panel with all decisions so far
        const decisions = progress.agent_decisions || [];
        if (decisions.length) {
            agentOutput.textContent = decisions.map((d, i) => {
                const focus = d.focus ? `\nfocus: ${d.focus}` : "";
                return `[Judge] round ${i + 1}\naction: ${d.action}\nreason: ${d.reason}${focus}`;
            }).join("\n\n");
        }

        // Update metrics table with current best vs initial
        if (progress.best_metrics && progress.initial_metrics) {
            renderMetrics({
                original_metrics: progress.initial_metrics,
                optimized_metrics: progress.best_metrics,
                improvement: {
                    area_improvement: safePct(progress.initial_metrics.area, progress.best_metrics.area),
                    ff_improvement: safePct(progress.initial_metrics.num_ff, progress.best_metrics.num_ff),
                    depth_improvement: safePct(progress.initial_metrics.logic_depth, progress.best_metrics.logic_depth),
                    score_improvement: bestScore !== undefined && initScore !== undefined
                        ? ((bestScore - initScore) / Math.max(1e-9, Math.abs(initScore)) * 100)
                        : 0,
                },
            });
        }
    }
}

function safePct(oldVal, newVal) {
    const o = parseFloat(oldVal || 0);
    const n = parseFloat(newVal || 0);
    if (Math.abs(o) < 1e-9) return 0;
    return (o - n) / Math.abs(o) * 100;
}

/* ─── Pipeline visualization ─── */
const pipelineNodes = {
    planner:   document.getElementById("node-planner"),
    coder:     document.getElementById("node-coder"),
    evaluator: document.getElementById("node-evaluator"),
    judge:     document.getElementById("node-judge"),
};
const pipelineStatus = {
    planner:   document.getElementById("st-planner"),
    coder:     document.getElementById("st-coder"),
    evaluator: document.getElementById("st-evaluator"),
    judge:     document.getElementById("st-judge"),
};
const pipes = {
    pc: document.getElementById("pipe-pc"),
    ce: document.getElementById("pipe-ce"),
    ej: document.getElementById("pipe-ej"),
};
const feedbackArc  = document.getElementById("feedback-arc");
const feedbackText = document.getElementById("feedback-text");
let lastPipelinePhase = "";

function resetPipeline() {
    Object.values(pipelineNodes).forEach(n => n.classList.remove("active", "done"));
    Object.values(pipes).forEach(p => p.classList.remove("active"));
    Object.values(pipelineStatus).forEach(s => { s.textContent = "idle"; });
    feedbackArc.classList.remove("active");
    feedbackText.textContent = "";
    lastPipelinePhase = "";
}

function setPipelineActive(role) {
    // Clear active from all, keep done
    Object.values(pipelineNodes).forEach(n => n.classList.remove("active"));
    Object.values(pipes).forEach(p => p.classList.remove("active"));
    if (pipelineNodes[role]) {
        pipelineNodes[role].classList.add("active");
    }
}

function setPipelineDone(role) {
    if (pipelineNodes[role]) {
        pipelineNodes[role].classList.remove("active");
        pipelineNodes[role].classList.add("done");
    }
}

function setNodeStatus(role, text) {
    if (pipelineStatus[role]) {
        pipelineStatus[role].textContent = text;
    }
}

function updatePipeline(progress) {
    if (!progress) return;
    const phase = progress.phase || "";

    // Avoid re-processing the exact same progress object
    const phaseKey = `${phase}_${progress.iteration || 0}`;
    if (phaseKey === lastPipelinePhase) return;
    lastPipelinePhase = phaseKey;

    const iter = progress.iteration || 0;
    const maxIter = progress.max_iterations || "?";

    switch (phase) {
        case "initial_eval":
            resetPipeline();
            setPipelineActive("evaluator");
            setNodeStatus("evaluator", "initial EDA...");
            break;

        case "planning":
            setPipelineDone("evaluator");
            setNodeStatus("evaluator", "done");
            setPipelineActive("planner");
            setNodeStatus("planner", "analyzing...");
            break;

        case "planning_done": {
            setPipelineDone("planner");
            const ps = progress.plan_summary || {};
            setNodeStatus("planner", ps.strategy || "done");
            break;
        }

        case "generating":
            // Planner done, Coder active
            setPipelineDone("planner");
            setPipelineActive("coder");
            pipes.pc.classList.add("active");
            setNodeStatus("coder", `iter ${iter}/${maxIter}`);
            // Show feedback loop from iteration 2 onward
            if (iter >= 2) {
                feedbackArc.classList.add("active");
                feedbackText.textContent = `iteration ${iter}`;
            }
            break;

        case "evaluating":
            setPipelineDone("coder");
            setNodeStatus("coder", "done");
            setPipelineActive("evaluator");
            pipes.ce.classList.add("active");
            setNodeStatus("evaluator", "EDA running...");
            break;

        case "iteration_done": {
            setPipelineDone("evaluator");
            setNodeStatus("evaluator", "done");
            setPipelineActive("judge");
            pipes.ej.classList.add("active");
            const decision = progress.decision || {};
            const action = decision.action || "?";
            setNodeStatus("judge", action);

            if (action === "stop") {
                // Final state — mark judge done, update feedback text
                setTimeout(() => {
                    setPipelineDone("judge");
                    setNodeStatus("judge", "stopped");
                    feedbackText.textContent = `${iter} rounds, stopped`;
                    Object.values(pipes).forEach(p => p.classList.remove("active"));
                }, 600);
            } else {
                // Continue — briefly show judge active, then loop back
                feedbackArc.classList.add("active");
                feedbackText.textContent = `iteration ${iter} done`;
            }
            break;
        }

        default:
            break;
    }
}

async function parseApiResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    const rawText = await response.text();
    if (contentType.includes("application/json")) {
        return JSON.parse(rawText);
    }

    const normalized = rawText.replace(/\s+/g, " ").trim().slice(0, 240);
    throw new Error(`HTTP ${response.status}: non-JSON response: ${normalized}`);
}

async function waitForJob(jobId, { onProgress } = {}) {
    const maxAttempts = 600;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        const response = await fetch(`/api/jobs/${jobId}`);
        const payload = await parseApiResponse(response);
        const job = payload.job || {};
        if (onProgress) {
            onProgress(job);
        }
        if (job.status === "completed") {
            return job.result;
        }
        if (job.status === "failed") {
            throw new Error(job.error || "Job failed");
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    throw new Error("Job polling timed out");
}

async function analyzeCode() {
    const code = inputArea.value;
    if (!code.trim()) {
        setStatus("请输入 Verilog 代码后再分析", "warn");
        addLogEntry("分析失败：代码为空", "warning");
        return;
    }

    const detectedModule = extractModuleName(code);
    setStatus("EDA 分析进行中...", "info");
    addLogEntry(`开始分析模块 ${detectedModule}`, "info");
    btnAnalyze.disabled = true;

    try {
        const response = await fetch("/api/analyze_async", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, module: detectedModule }),
        });
        const startPayload = await parseApiResponse(response);
        const result = await waitForJob(startPayload.job_id, {
            onProgress(job) {
                setStatus(`EDA 分析中... ${job.status}`, "info");
            },
        });
        edaOutput.textContent = formatEDAResult(result);
        if (result.success) {
            originalCode.textContent = code;
            originalCode.classList.add("collapsed");
            setStatus("EDA 分析完成", "success");
            addLogEntry("EDA 分析成功完成", "success");
        } else {
            setStatus(result.error || "EDA 分析失败", "error");
            addLogEntry(`EDA 分析失败: ${result.error || "未知错误"}`, "error");
        }
    } catch (err) {
        setStatus(`请求失败: ${err.message}`, "error");
        addLogEntry(`网络请求失败: ${err.message}`, "error");
    } finally {
        btnAnalyze.disabled = false;
    }
}

function updateOptimizationResult(result) {
    if (!result.success) {
        setStatus(result.error || "优化失败", "error");
        addLogEntry(`优化失败: ${result.error || "未知错误"}`, "error");
        resetPipeline();
        return;
    }

    // Mark all pipeline nodes as done
    Object.values(pipelineNodes).forEach(n => { n.classList.remove("active"); n.classList.add("done"); });
    Object.values(pipes).forEach(p => p.classList.remove("active"));
    setNodeStatus("planner", "done");
    setNodeStatus("coder", "done");
    setNodeStatus("evaluator", "done");
    setNodeStatus("judge", "done");
    feedbackText.textContent = `${result.total_iterations || "?"} rounds, completed`;

    originalCode.textContent = result.original_code || "--";
    originalCode.classList.add("collapsed");
    optimizedCode.textContent = result.optimized_code || "--";
    optimizedCode.classList.add("collapsed");
    renderMetrics(result);
    renderAgentDecisions(result);

    // Show circuit section and store codes for later generation
    const circuitSection = document.getElementById("circuit-section");
    circuitSection.style.display = "";
    circuitSection._originalCode = result.original_code;
    circuitSection._optimizedCode = result.optimized_code;

    const llmMode = result.llm_mode ? `，模式 ${result.llm_mode}` : "";
    setStatus(`优化完成${llmMode}`, "success");
    addLogEntry("Agent 优化成功完成", "success");
}

async function optimizeCode() {
    const code = inputArea.value;
    const target = targetInput.value.trim();
    const scenario = scenarioInput.value.trim();

    if (!code.trim()) {
        setStatus("请输入 Verilog 代码后再优化", "warn");
        addLogEntry("优化失败：代码为空", "warning");
        return;
    }

    setStatus("Agent 正在优化，请稍候...", "info");
    addLogEntry("开始 Agent 优化任务", "info");
    btnOptimize.disabled = true;
    resetPipeline();

    try {
        const response = await fetch("/api/optimize_async", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, target, scenario }),
        });
        const startPayload = await parseApiResponse(response);
        addLogEntry("优化任务已提交，等待 Agent 启动…", "info");
        const result = await waitForJob(startPayload.job_id, {
            onProgress(job) {
                renderProgress(job.progress);
            },
        });
        updateOptimizationResult(result);
    } catch (err) {
        setStatus(`请求失败: ${err.message}`, "error");
        addLogEntry(`优化请求失败: ${err.message}`, "error");
    } finally {
        btnOptimize.disabled = false;
    }
}

btnAnalyze.addEventListener("click", analyzeCode);
btnOptimize.addEventListener("click", optimizeCode);
toggleOriginalBtn.addEventListener("click", toggleCodeCollapse);
document.getElementById("toggle-optimized").addEventListener("click", () => {
    optimizedCode.classList.toggle("collapsed");
});
btnSaveSettings.addEventListener("click", saveSettings);

/* ─── Circuit simulation (DigitalJS) ─── */
let circuitInstances = { original: null, optimized: null };

async function fetchCircuitJson(code) {
    const modMatch = code.match(/module\s+(\w+)/);
    const moduleName = modMatch ? modMatch[1] : "top";
    const response = await fetch("/api/circuit_json", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, module: moduleName }),
    });
    return response.json();
}

function renderCircuit(containerId, circuitJson) {
    const container = document.getElementById(containerId);
    container.textContent = "";
    if (typeof digitaljs === "undefined") {
        container.textContent = "DigitalJS library not loaded";
        return null;
    }
    try {
        const circuit = new digitaljs.Circuit(circuitJson);
        const paper = circuit.displayOn($(container));
        circuit.start();
        return circuit;
    } catch (e) {
        container.textContent = "Circuit render error: " + e.message;
        return null;
    }
}

async function generateCircuits() {
    const section = document.getElementById("circuit-section");
    const origCode = section._originalCode;
    const optCode = section._optimizedCode;
    if (!origCode || !optCode) return;

    const btn = document.getElementById("btn-gen-circuit");
    btn.disabled = true;
    btn.textContent = "生成中…";

    // Clean up previous instances
    if (circuitInstances.original) { circuitInstances.original.stop(); circuitInstances.original = null; }
    if (circuitInstances.optimized) { circuitInstances.optimized.stop(); circuitInstances.optimized = null; }
    document.getElementById("circuit-original").textContent = "加载中…";
    document.getElementById("circuit-optimized").textContent = "加载中…";

    try {
        const [origResult, optResult] = await Promise.all([
            fetchCircuitJson(origCode),
            fetchCircuitJson(optCode),
        ]);

        if (origResult.success) {
            circuitInstances.original = renderCircuit("circuit-original", origResult.output);
        } else {
            document.getElementById("circuit-original").textContent = origResult.error || "Failed";
        }

        if (optResult.success) {
            circuitInstances.optimized = renderCircuit("circuit-optimized", optResult.output);
        } else {
            document.getElementById("circuit-optimized").textContent = optResult.error || "Failed";
        }

        addLogEntry("电路仿真图已生成", "success");
    } catch (err) {
        addLogEntry(`电路图生成失败: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        btn.textContent = "生成电路图";
    }
}

document.getElementById("btn-gen-circuit").addEventListener("click", generateCircuits);

loadSettings();
