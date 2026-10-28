"use strict";

const $ = (id) => document.getElementById(id);

const dropZone = $("dropZone");
const fileInput = $("fileInput");
const fileChip = $("fileChip");
const fileNameEl = $("fileName");
const fileSizeEl = $("fileSize");
const fileIconEl = $("fileIcon");
const uploadBtn = $("uploadBtn");
const uploadStatus = $("uploadStatus");
const chatEl = $("chat");
const questionInput = $("questionInput");
const askBtn = $("askBtn");
const docList = $("docList");
const docCount = $("docCount");
const refreshBtn = $("refreshBtn");
const modeHint = $("modeHint");
const segEl = document.querySelector(".seg");
const modeRadios = document.querySelectorAll('input[name="mode"]');

const ACCEPT_RE = /\.(txt|pdf|docx)$/i;
let currentFile = null;

/* ---------- 多轮对话历史 ---------- */
/* 后端 /ask 与 /ask_agent 都支持 history 参数（结合历史消除代词、注入生成上下文），
   前端必须随请求带上最近几轮，多轮能力才是活的 */
const HISTORY_LIMIT = 6; // 随请求发送的最近条数（约 3 轮问答）
const HISTORY_CONTENT_MAX = 400; // 单条历史截断长度，避免请求体膨胀
const chatHistory = [];

function rememberTurn(role, content) {
  if (!content) return;
  if (content.length > HISTORY_CONTENT_MAX) {
    content = content.slice(0, HISTORY_CONTENT_MAX) + "…（已截断）";
  }
  chatHistory.push({ role, content });
  // 内存里也只保留最近若干条，防止长会话无限增长
  if (chatHistory.length > HISTORY_LIMIT * 2) {
    chatHistory.splice(0, chatHistory.length - HISTORY_LIMIT);
  }
}

/* ---------- 通用工具 ---------- */

function esc(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c],
  );
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    /* 非 JSON 响应 */
  }
  if (!res.ok)
    throw new Error((data && data.error) || `请求失败（HTTP ${res.status}）`);
  return data;
}

function showStatus(kind, html) {
  uploadStatus.className = "status show " + kind;
  uploadStatus.innerHTML = html;
}

/* ---------- 模式切换 ---------- */

function currentMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked && checked.value === "agent" ? "agent" : "plain";
}

modeRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    const mode = radio.value === "agent" ? "agent" : "plain";
    segEl.dataset.mode = mode;
    modeHint.textContent = mode === "agent" ? "多轮检索 · 自主规划" : "单轮检索";
  });
});

/* ---------- 编号标尺：标出当前所在的区块 ---------- */

const railItems = [...document.querySelectorAll(".rail-item")];
const railSections = railItems
  .map((item) => document.querySelector(item.getAttribute("href")))
  .filter(Boolean);

function markActiveSection(id) {
  railItems.forEach((item) => {
    item.classList.toggle("is-active", item.getAttribute("href") === "#" + id);
  });
}

function updateActiveSection() {
  if (railSections.length === 0) return;
  const LINE = 140;
  let current = railSections[0];
  railSections.forEach((section) => {
    if (section.getBoundingClientRect().top - LINE <= 0) current = section;
  });
  markActiveSection(current.id);
}

window.addEventListener("scroll", updateActiveSection, { passive: true });
window.addEventListener("resize", updateActiveSection);

/* ---------- 文档入库 ---------- */

function selectFile(file) {
  if (!file) return;
  if (!ACCEPT_RE.test(file.name)) {
    clearFile();
    showStatus(
      "error",
      `不支持的文件格式，仅支持 .txt / .pdf / .docx：${esc(file.name)}`,
    );
    return;
  }
  currentFile = file;
  fileNameEl.textContent = file.name;
  fileNameEl.title = file.name;
  fileSizeEl.textContent = formatSize(file.size);
  const ext = (file.name.match(/\.([^.]+)$/)?.[1] || "txt").toUpperCase();
  fileIconEl.textContent = ext;
  fileChip.classList.add("show");
  uploadStatus.classList.remove("show");
}

function clearFile() {
  currentFile = null;
  fileInput.value = "";
  fileChip.classList.remove("show");
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

$("fileRemove").addEventListener("click", clearFile);

["dragover", "dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.toggle("dragover", evt === "dragover");
    if (evt === "drop") selectFile(e.dataTransfer.files[0]);
  });
});

uploadBtn.addEventListener("click", async () => {
  if (!currentFile) {
    showStatus("error", "请先选择文件");
    return;
  }

  uploadBtn.disabled = true;
  uploadBtn.textContent = "上传中…";
  showStatus(
    "loading",
    '<span class="spinner dark"></span><span>正在切分并写入向量库，请稍候…</span>',
  );

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const data = await fetchJSON("/upload_knowledge", {
      method: "POST",
      body: formData,
    });
    showStatus("success", esc(data.message || "上传成功"));
    clearFile();
    loadDocuments();
  } catch (error) {
    showStatus("error", esc(error.message));
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "上传入库";
  }
});

/* ---------- 知识库问答 ---------- */

function scrollChat() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeChatEmpty() {
  const empty = $("chatEmpty");
  if (empty) empty.remove();
}

/** 一次问答就是一条记录：左侧标「问」或「答」，正文各自成段 */
function appendTurn(kind, tagText) {
  const div = document.createElement("div");
  div.className = "turn turn-" + kind;

  const tag = document.createElement("span");
  tag.className = "turn-tag";
  tag.textContent = tagText;

  const body = document.createElement("div");
  body.className = "turn-body";

  div.append(tag, body);
  chatEl.appendChild(div);
  return { turn: div, body };
}

function appendUserMsg(text) {
  const { body } = appendTurn("q", "问");
  body.textContent = text;
}

function appendBotMsg() {
  const { turn, body } = appendTurn("a", "答");
  body.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  return turn;
}

/* ---------- Agent 产出渲染：SQL 表格 / 计算器 / 工具异常 ---------- */

const SQL_MAX_DISPLAY_ROWS = 50;

/** 单个 SQL 块：SQL 原文折叠 + 结果表格（或空态）+ 截断提示 */
function sqlBlockHtml(block, label) {
  const columns = block.columns || [];
  const rows = block.rows || [];

  const src = block.sql
    ? `<details class="sql-src"><summary>${esc(label)} · SQL 原文</summary><pre>${esc(block.sql)}</pre></details>`
    : "";

  let tableHtml;
  if (!columns.length || !rows.length) {
    tableHtml = '<div class="sql-empty">查询结果为空（该条件下没有数据记录）</div>';
  } else {
    const shown = rows.slice(0, SQL_MAX_DISPLAY_ROWS);
    const head = `<tr>${columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const bodyRows = shown
      .map(
        (row) =>
          `<tr>${columns
            .map((c) => `<td>${esc(row[c] == null ? "" : row[c])}</td>`)
            .join("")}</tr>`,
      )
      .join("");
    tableHtml = `<div class="sql-table-wrap"><table class="sql-table"><thead>${head}</thead><tbody>${bodyRows}</tbody></table></div>`;
    if (rows.length > SQL_MAX_DISPLAY_ROWS) {
      tableHtml += `<div class="sql-note">共 ${rows.length} 行，仅展示前 ${SQL_MAX_DISPLAY_ROWS} 行</div>`;
    }
  }

  if (block.truncated) {
    tableHtml += '<div class="sql-note">结果行数过多，已被后端截断</div>';
  }

  return `<div class="sql-block">${src}${tableHtml}</div>`;
}

function renderSqlBlocks(blocks) {
  const list = Array.isArray(blocks) ? blocks.filter((b) => b && (b.sql || b.columns)) : [];
  if (!list.length) return "";
  const html = list
    .map((b, i) => sqlBlockHtml(b, list.length > 1 ? `查询 ${i + 1}` : "查询"))
    .join("");
  return `<div class="sql-section">${html}</div>`;
}

/**
 * 渲染机器人消息
 * @param {HTMLElement} turn 消息容器
 * @param {object} data       接口返回（answer/sources/retry_count/sql_blocks/calc_result/tool_errors）
 * @param {boolean} isAgent   是否为 Agent 模式
 */
function renderBotMsg(turn, data, isAgent) {
  turn.querySelector(".turn-body").textContent = data.answer || "（未返回内容）";

  // 重渲染时先清掉旧扩展块
  turn.querySelectorAll(".sql-section, .calc-result, .tool-errors, .sources").forEach((n) => n.remove());

  if (isAgent) {
    const sqlHtml = renderSqlBlocks(data.sql_blocks);
    if (sqlHtml) turn.insertAdjacentHTML("beforeend", sqlHtml);

    if (data.calc_result) {
      turn.insertAdjacentHTML(
        "beforeend",
        `<div class="calc-result">计算器结果：${esc(data.calc_result)}</div>`,
      );
    }

    if (Array.isArray(data.tool_errors) && data.tool_errors.length) {
      const items = data.tool_errors.map((e) => `<li>${esc(e)}</li>`).join("");
      turn.insertAdjacentHTML(
        "beforeend",
        `<div class="tool-errors"><div class="te-head">部分工具执行异常，以下回答可能不完整：</div><ul>${items}</ul></div>`,
      );
    }
  }

  const unique = [...new Set(data.sources || [])];
  const tags = [];

  if (unique.length > 0) {
    const sourceTags = unique
      .map((s) => `<span class="tag" title="${esc(s)}">${esc(s)}</span>`)
      .join("");
    tags.push(`<span class="sources-label">来源</span>${sourceTags}`);
  }

  if (isAgent) {
    const meta = [`<span class="tag agent">Agent</span>`];
    if (typeof data.retry_count === "number") {
      meta.push(`<span class="tag retry">检索 ${data.retry_count} 轮</span>`);
    }
    tags.push(meta.join(""));
  }

  if (tags.length > 0) {
    turn.insertAdjacentHTML(
      "beforeend",
      `<div class="sources">${tags.join("")}</div>`,
    );
  }
}

askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) askQuestion();
});

document.querySelectorAll(".example").forEach((btn) => {
  btn.addEventListener("click", () => {
    questionInput.value = btn.dataset.q || btn.textContent.trim();
    questionInput.focus();
  });
});

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question || askBtn.disabled) return;

  const useAgent = currentMode() === "agent";
  const endpoint = useAgent ? "/ask_agent" : "/ask";

  removeChatEmpty();
  appendUserMsg(question);
  rememberTurn("user", question);
  questionInput.value = "";
  const pending = appendBotMsg();
  scrollChat();

  askBtn.disabled = true;
  askBtn.innerHTML = '<span class="spinner small"></span>检索中';
  try {
    const data = await fetchJSON(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        // 带上最近几轮对话，后端用它做查询改写（消解代词）与生成上下文注入
        history: chatHistory.slice(-HISTORY_LIMIT),
      }),
    });
    renderBotMsg(pending, data, useAgent);
    // 成功才计入历史；失败的轮次不进上下文
    rememberTurn("assistant", data.answer || "");
  } catch (error) {
    pending.classList.add("is-error");
    pending.querySelector(".turn-body").textContent = error.message;
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "提问";
    questionInput.focus();
    scrollChat();
  }
}

/* ---------- 已入库文档 ---------- */

let knownDocs = null;

function renderDocs(documents) {
  const names = documents || [];
  // 首次加载不算「新入列」，只有上传后新增的文档才闪一下
  const fresh = knownDocs === null ? [] : names.filter((n) => !knownDocs.includes(n));

  docList.innerHTML = "";

  if (names.length === 0) {
    docList.innerHTML =
      '<li class="doc-empty">知识库为空，请先在上方上传文档。</li>';
    docCount.hidden = true;
    knownDocs = names;
    return;
  }

  names.forEach((name) => {
    const ext = (name.match(/\.([^.]+)$/)?.[1] || "txt").toUpperCase();
    const li = document.createElement("li");
    li.className = fresh.includes(name) ? "doc-item is-new" : "doc-item";
    li.innerHTML =
      `<span class="name" title="${esc(name)}">${esc(name)}</span>` +
      `<span class="kind">${esc(ext)}</span>`;
    docList.appendChild(li);
  });

  docCount.textContent = `${names.length} 份`;
  docCount.hidden = false;
  knownDocs = names;
}

async function loadDocuments() {
  refreshBtn.disabled = true;
  refreshBtn.classList.add("spin");
  try {
    const data = await fetchJSON("/documents");
    renderDocs(data.documents);
  } catch (error) {
    docList.innerHTML = `<li class="doc-empty">${esc(error.message)}</li>`;
    docCount.hidden = true;
    knownDocs = null;
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.classList.remove("spin");
  }
}

refreshBtn.addEventListener("click", loadDocuments);

loadDocuments();
updateActiveSection();
