const btnAnalyze = document.getElementById("btn-analyze");
const btnOptimize = document.getElementById("btn-optimize");
const inputArea = document.getElementById("verilog-input");
const statusBox = document.getElementById("global-status");
const edaOutput = document.getElementById("eda-output");
const originalCode = document.getElementById("original-code");
const optimizedCode = document.getElementById("optimized-code");
const metricsTable = document.getElementById("metrics-table").querySelector("tbody");
const logContainer = document.getElementById("log-container");
const toggleOriginalBtn = document.getElementById("toggle-original");

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
    
    // 限制日志条目数量，避免内存占用过多
    const entries = logContainer.querySelectorAll(".log-entry");
    if (entries.length > 100) {
        entries[0].remove();
    }
}

function toggleCodeCollapse() {
    const isCollapsed = originalCode.classList.contains("collapsed");
    if (isCollapsed) {
        originalCode.classList.remove("collapsed");
    } else {
        originalCode.classList.add("collapsed");
    }
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

function extractModuleName(code) {
    const match = code.match(/module\s+(\w+)/);
    return match ? match[1] : "top";
}

async function analyzeCode() {
    const code = inputArea.value;
    if (!code.trim()) {
        setStatus("请输入Verilog代码后再分析", "warn");
        addLogEntry("分析失败：代码为空", "warning");
        return;
    }
    
    const detectedModule = extractModuleName(code);
    setStatus("EDA分析进行中...");
    addLogEntry(`开始分析模块: ${detectedModule}`, "info");
    btnAnalyze.disabled = true;

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                code,
                module: detectedModule,
            }),
        });
        const result = await response.json();
        edaOutput.textContent = formatEDAResult(result);
        if (result.success) {
            originalCode.textContent = code;
            originalCode.classList.add("collapsed"); // 默认折叠
            setStatus("EDA分析完成", "success");
            addLogEntry("EDA分析成功完成", "success");
        } else {
            setStatus(result.error || "EDA分析失败", "error");
            addLogEntry(`EDA分析失败: ${result.error || "未知错误"}`, "error");
        }
    } catch (err) {
        console.error(err);
        setStatus(`请求失败: ${err.message}`, "error");
        addLogEntry(`网络请求失败: ${err.message}`, "error");
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
        addLogEntry(`优化失败: ${result.error || "未知错误"}`, "error");
        return;
    }
    originalCode.textContent = result.original_code || "--";
    originalCode.classList.add("collapsed"); // 默认折叠
    optimizedCode.textContent = result.optimized_code || "--";
    renderMetrics(result);
    setStatus("优化完成", "success");
    addLogEntry("代码优化成功完成", "success");
}

async function optimizeCode() {
    const code = inputArea.value;
    if (!code.trim()) {
        setStatus("请输入Verilog代码后再优化", "warn");
        addLogEntry("优化失败：代码为空", "warning");
        return;
    }
    
    setStatus("正在优化，请稍候...", "info");
    addLogEntry("开始优化任务（这可能需要几分钟）", "info");
    btnOptimize.disabled = true;

    try {
        // 直接调用优化接口，等待结果返回
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code, target: "" }),
        });
        
        const result = await response.json();
        console.log("[优化] 收到结果:", result);
        
        // 直接更新结果
        updateOptimizationResult(result);
        btnOptimize.disabled = false;
        
    } catch (err) {
        console.error(err);
        setStatus(`请求失败: ${err.message}`, "error");
        addLogEntry(`优化请求失败: ${err.message}`, "error");
        btnOptimize.disabled = false;
    }
}

btnAnalyze.addEventListener("click", analyzeCode);
btnOptimize.addEventListener("click", optimizeCode);
toggleOriginalBtn.addEventListener("click", toggleCodeCollapse);
