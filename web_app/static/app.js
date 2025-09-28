const btnAnalyze = document.getElementById("btn-analyze");
const btnOptimize = document.getElementById("btn-optimize");
const inputArea = document.getElementById("verilog-input");
const targetInput = document.getElementById("target-text");
const statusBox = document.getElementById("global-status");
const edaOutput = document.getElementById("eda-output");
const originalCode = document.getElementById("original-code");
const optimizedCode = document.getElementById("optimized-code");
const metricsTable = document.getElementById("metrics-table").querySelector("tbody");

let pollTimer = null;

function setStatus(message, type = "info") {
    statusBox.textContent = message;
    statusBox.dataset.type = type;
    statusBox.className = `status ${type}`;
}

function formatEDAResult(result) {
    if (!result.success) {
        return `分析失败\n原因: ${result.error || "未知错误"}`;
    }
    const data = result.data || {};
    const lines = [
        "EDA分析成功",
        `语法检查: ${data.syntax_ok ? "通过" : "失败"}`,
        `综合状态: ${data.synth_ok ? "通过" : "失败"}`,
        `面积 (cells): ${data.area}`,
        `触发器数量: ${data.num_ff}`,
        `逻辑深度: ${data.logic_depth}`,
    ];
    return lines.join("\n");
}

async function analyzeCode() {
    const code = inputArea.value;
    if (!code.trim()) {
        setStatus("请输入Verilog代码后再分析", "warn");
        return;
    }
    setStatus("EDA分析进行中...");
    btnAnalyze.disabled = true;

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                code,
                module: "top",
            }),
        });
        const result = await response.json();
        edaOutput.textContent = formatEDAResult(result);
        if (result.success) {
            originalCode.textContent = code;
            setStatus("EDA分析完成", "success");
        } else {
            setStatus(result.error || "EDA分析失败", "error");
        }
    } catch (err) {
        console.error(err);
        setStatus(`请求失败: ${err.message}`, "error");
    } finally {
        btnAnalyze.disabled = false;
    }
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

    const htmlRows = rows.map(({ key, label, isImprovementOnly }) => {
        const origVal = original[key];
        const optVal = optimized[key];
        const impVal = improvement[`${key}_improvement`] ?? (isImprovementOnly ? improvement[key] : undefined);

        return `<tr>
            <td>${label}</td>
            <td>${origVal !== undefined ? origVal : "--"}</td>
            <td>${optVal !== undefined ? optVal : "--"}</td>
            <td>${impVal !== undefined ? `${impVal.toFixed(2)}%` : "--"}</td>
        </tr>`;
    }).join("");

    metricsTable.innerHTML = htmlRows;
}

function updateOptimizationResult(result) {
    if (!result.success) {
        setStatus(result.error || "优化失败", "error");
        return;
    }
    originalCode.textContent = result.original_code || "--";
    optimizedCode.textContent = result.optimized_code || "--";
    renderMetrics(result);
    setStatus("优化完成", "success");
}

async function pollOptimizationStatus() {
    try {
        const statusResp = await fetch("/api/optimize/status");
        const statusData = await statusResp.json();
        if (statusData.running) {
            setStatus("正在优化，请稍候...", "info");
            return;
        }
        const resultResp = await fetch("/api/optimize/result");
        if (resultResp.status === 404) {
            setStatus("暂无结果，请稍后重试", "warn");
            stopPolling();
            return;
        }
        const result = await resultResp.json();
        updateOptimizationResult(result);
        stopPolling();
    } catch (err) {
        console.error(err);
        setStatus(`轮询失败: ${err.message}`, "error");
        stopPolling();
    }
}

function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollOptimizationStatus, 3000);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function optimizeCode() {
    const code = inputArea.value;
    const target = targetInput.value;
    if (!code.trim()) {
        setStatus("请输入Verilog代码后再优化", "warn");
        return;
    }
    setStatus("提交优化任务...");
    btnOptimize.disabled = true;

    try {
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, target }),
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            setStatus(result.error || "优化启动失败", "error");
            btnOptimize.disabled = false;
            return;
        }
        setStatus("优化任务已启动，正在等待结果...", "info");
        startPolling();
    } catch (err) {
        console.error(err);
        setStatus(`请求失败: ${err.message}`, "error");
        btnOptimize.disabled = false;
    }
}

btnAnalyze.addEventListener("click", analyzeCode);
btnOptimize.addEventListener("click", optimizeCode);

window.addEventListener("beforeunload", () => stopPolling());
