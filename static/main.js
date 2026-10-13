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

/**
 * 渲染机器人消息
 * @param {HTMLElement} turn 消息容器
 * @param {string} answer    回答正文
 * @param {string[]} sources 参考来源
 * @param {number|null} retryCount Agent 模式下的检索轮数
 * @param {boolean} isAgent  是否为 Agent 模式
 */
function renderBotMsg(turn, answer, sources, retryCount, isAgent) {
  turn.querySelector(".turn-body").textContent = answer;

  const unique = [...new Set(sources || [])];
  const tags = [];

  if (unique.length > 0) {
    const sourceTags = unique
      .map((s) => `<span class="tag" title="${esc(s)}">${esc(s)}</span>`)
      .join("");
    tags.push(`<span class="sources-label">来源</span>${sourceTags}`);
  }

  if (isAgent) {
    const meta = [`<span class="tag agent">Agent</span>`];
    if (typeof retryCount === "number") {
      meta.push(`<span class="tag retry">检索 ${retryCount} 轮</span>`);
    }
    tags.push(meta.join(""));
  }

  const old = turn.querySelector(".sources");
  if (old) old.remove();

  if (tags.length > 0) {
    const box = document.createElement("div");
    box.className = "sources";
    box.innerHTML = tags.join("");
    turn.appendChild(box);
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
  questionInput.value = "";
  const pending = appendBotMsg();
  scrollChat();

  askBtn.disabled = true;
  askBtn.innerHTML = '<span class="spinner small"></span>检索中';
  try {
    const data = await fetchJSON(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderBotMsg(
      pending,
      data.answer || "（未返回内容）",
      data.sources,
      typeof data.retry_count === "number" ? data.retry_count : null,
      useAgent,
    );
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
