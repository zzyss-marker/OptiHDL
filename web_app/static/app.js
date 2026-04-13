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
        return `round ${index + 1}\naction: ${decision.action}\nreason: ${decision.reason}${focus}`;
    }).join("\n\n");
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
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, module: detectedModule }),
        });
        const result = await response.json();
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
        return;
    }

    originalCode.textContent = result.original_code || "--";
    originalCode.classList.add("collapsed");
    optimizedCode.textContent = result.optimized_code || "--";
    renderMetrics(result);
    renderAgentDecisions(result);

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

    try {
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, target, scenario }),
        });
        const result = await response.json();
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
btnSaveSettings.addEventListener("click", saveSettings);
loadSettings();
