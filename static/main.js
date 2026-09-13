'use strict';

const $ = id => document.getElementById(id);

const dropZone = $('dropZone');
const fileInput = $('fileInput');
const fileChip = $('fileChip');
const fileNameEl = $('fileName');
const fileSizeEl = $('fileSize');
const fileIconEl = $('fileIcon');
const uploadBtn = $('uploadBtn');
const uploadStatus = $('uploadStatus');
const chatEl = $('chat');
const questionInput = $('questionInput');
const askBtn = $('askBtn');
const docList = $('docList');
const docCount = $('docCount');
const refreshBtn = $('refreshBtn');

const ACCEPT_RE = /\.(txt|pdf|docx)$/i;
let currentFile = null;

/* ---------- 通用工具 ---------- */

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

async function fetchJSON(url, options) {
    const res = await fetch(url, options);
    let data = null;
    try { data = await res.json(); } catch (_) { /* 非 JSON 响应 */ }
    if (!res.ok) throw new Error((data && data.error) || `请求失败（HTTP ${res.status}）`);
    return data;
}

function showStatus(kind, html) {
    uploadStatus.className = 'status show ' + kind;
    uploadStatus.innerHTML = html;
}

/* ---------- 文档入库 ---------- */

function selectFile(file) {
    if (!file) return;
    if (!ACCEPT_RE.test(file.name)) {
        clearFile();
        showStatus('error', `❌ 不支持的文件格式，仅支持 .txt / .pdf / .docx：${esc(file.name)}`);
        return;
    }
    currentFile = file;
    fileNameEl.textContent = file.name;
    fileNameEl.title = file.name;
    fileSizeEl.textContent = formatSize(file.size);
    fileIconEl.textContent = file.name.toLowerCase().endsWith('.pdf') ? '📕'
        : file.name.toLowerCase().endsWith('.docx') ? '📘' : '📄';
    fileChip.classList.add('show');
    uploadStatus.classList.remove('show');
}

function clearFile() {
    currentFile = null;
    fileInput.value = '';
    fileChip.classList.remove('show');
}

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));

$('fileRemove').addEventListener('click', clearFile);

['dragover', 'dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.classList.toggle('dragover', evt === 'dragover');
        if (evt === 'drop') selectFile(e.dataTransfer.files[0]);
    });
});

uploadBtn.addEventListener('click', async () => {
    if (!currentFile) {
        showStatus('error', '❌ 请先选择文件');
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.textContent = '上传中…';
    showStatus('loading', '<span class="spinner dark"></span><span>正在切分并存入向量库，请稍候…</span>');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const data = await fetchJSON('/upload_knowledge', { method: 'POST', body: formData });
        showStatus('success', `✅ ${esc(data.message || '上传成功')}`);
        clearFile();
        loadDocuments();
    } catch (error) {
        showStatus('error', `❌ ${esc(error.message)}`);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '上传入库';
    }
});

/* ---------- 知识库问答 ---------- */

function scrollChat() {
    chatEl.scrollTop = chatEl.scrollHeight;
}

function removeChatEmpty() {
    const empty = $('chatEmpty');
    if (empty) empty.remove();
}

function appendUserMsg(text) {
    const div = document.createElement('div');
    div.className = 'msg user';
    div.textContent = text;
    chatEl.appendChild(div);
}

function appendBotMsg() {
    const div = document.createElement('div');
    div.className = 'msg bot';
    div.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    chatEl.appendChild(div);
    return div;
}

function renderBotMsg(div, answer, sources) {
    let html = esc(answer);
    const unique = [...new Set(sources || [])];
    if (unique.length > 0) {
        const tags = unique.map(s => `<span class="tag" title="${esc(s)}">${esc(s)}</span>`).join('');
        html += `<div class="sources">📎 参考来源：${tags}</div>`;
    }
    div.innerHTML = html;
}

askBtn.addEventListener('click', askQuestion);
questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.isComposing) askQuestion();
});

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question || askBtn.disabled) return;

    removeChatEmpty();
    appendUserMsg(question);
    questionInput.value = '';
    const typing = appendBotMsg();
    scrollChat();

    askBtn.disabled = true;
    try {
        const data = await fetchJSON('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        renderBotMsg(typing, data.answer || '（未返回内容）', data.sources);
    } catch (error) {
        typing.className = 'msg bot error';
        typing.textContent = '❌ ' + error.message;
    } finally {
        askBtn.disabled = false;
        questionInput.focus();
        scrollChat();
    }
}

/* ---------- 已入库文档 ---------- */

function renderDocs(documents) {
    docList.innerHTML = '';

    if (!documents || documents.length === 0) {
        docList.innerHTML = '<li class="doc-empty">知识库为空，请先在上方上传文档。</li>';
        docCount.hidden = true;
        return;
    }

    documents.forEach(name => {
        const ext = (name.match(/\.([^.]+)$/)?.[1] || 'txt').toUpperCase();
        const icon = ext === 'PDF' ? '📕' : ext === 'DOCX' ? '📘' : '📄';
        const li = document.createElement('li');
        li.className = 'doc-item';
        li.innerHTML = `<span>${icon}</span><span class="name" title="${esc(name)}">${esc(name)}</span><span class="kind">${esc(ext)}</span>`;
        docList.appendChild(li);
    });

    docCount.textContent = `${documents.length} 个`;
    docCount.hidden = false;
}

async function loadDocuments() {
    refreshBtn.disabled = true;
    refreshBtn.classList.add('spin');
    try {
        const data = await fetchJSON('/documents');
        renderDocs(data.documents);
    } catch (error) {
        docList.innerHTML = `<li class="doc-empty">❌ ${esc(error.message)}</li>`;
        docCount.hidden = true;
    } finally {
        refreshBtn.disabled = false;
        refreshBtn.classList.remove('spin');
    }
}

refreshBtn.addEventListener('click', loadDocuments);

loadDocuments();
