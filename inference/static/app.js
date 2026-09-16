"use strict";

// Browser client for the streaming demo. Layout and interaction follow the
// internal lite demo; the wire protocol is this project's own:
//   * the init frame carries direction, latency_mode, the segmentation policy,
//     and the session glossary;
//   * `asr` events carry only the newly recognized increment plus a `reset`
//     flag marking the end of an utterance.

const $ = (id) => document.getElementById(id);

const MAX_TERMS = 200;
const TERMS_STORAGE_KEY = "t3po.demo.terms";
const LANG_STORAGE_KEY = "t3po.demo.lang";
// Gap between text chunks. Fast enough to feel live, slow enough that each
// chunk is a separate arrival rather than one batched request.
const TEXT_CHUNK_INTERVAL_MS = 260;
const METRICS_STORAGE_KEY = "t3po.demo.metrics";

const state = {
  socket: null,
  initialized: false,
  direction: "zh2en",
  latencyMode: "native",
  displayMode: "sentence",
  textTimer: null,
  textQueue: [],
  textTotal: 0,
  recording: false,
  paused: false,
  ending: false,
  inputMode: "microphone",
  mediaStream: null,
  audioContext: null,
  sourceNode: null,
  processor: null,
  silentGain: null,
  fileTimer: null,
  fileContext: null,
  fileSamples: null,
  fileOffset: 0,
  fileDuration: 0,
  sourceText: "",
  pairs: [],
  asrConfigured: false,
};

// --- i18n -----------------------------------------------------------------
// Two languages, resolved by data-i18n attributes so the markup carries the zh
// fallback and the script only swaps strings.
const STRINGS = {
  zh: {
    "label.direction": "方向",
    "label.latency": "延迟档位",
    "label.display": "显示",
    "label.audioSource": "音频来源",
    "label.textInput": "文本输入",
    "label.chunkSize": "每次",
    "label.chunkUnit": "单位",
    "display.sentence": "句对",
    "display.segment": "模型段",
    "dir.zh2en": "中文 → 英文",
    "dir.en2zh": "英文 → 中文",
    "latency.low": "低延迟",
    "latency.native": "原生",
    "latency.high": "高质量",
    "src.microphone": "麦克风",
    "src.tab": "浏览器标签页音频",
    "src.file": "音频文件",
    "btn.start": "🎤 开始",
    "btn.upload": "📁 上传音频",
    "btn.pause": "暂停",
    "btn.resume": "继续",
    "btn.end": "结束",
    "btn.reset": "新会话",
    "btn.sendText": "流式送入",
    "btn.stopText": "停止",
    "btn.metrics": "指标",
    "glossary.button": "术语",
    "glossary.title": "术语表",
    "glossary.hint": "每行一条，格式为「源词 -> 译词」。可直接粘贴多行。",
    "glossary.add": "添加",
    "glossary.empty": "还没有术语。",
    "glossary.locked": "术语在会话开始时读取，需结束会话后再修改。",
    "glossary.clear": "全部清除",
    "glossary.done": "完成",
    "glossary.invalid": "格式应为「源词 -> 译词」。",
    "glossary.full": `最多 ${MAX_TERMS} 条术语。`,
    "panel.compare": "完整对照",
    "panel.fullSource": "完整原文",
    "panel.fullTarget": "完整译文",
    "placeholder.source": "等待原文…",
    "placeholder.translation": "译文会在模型决定输出后显示。",
    "placeholder.textInput": "粘贴或输入一段文本，Ctrl+Enter 开始流式送入",
    "metric.segments": "段数",
    "metric.calls": "模型调用",
    "metric.buffer": "缓冲单位",
    "metric.asr": "ASR",
    "hint.asr": "未配置 ASR 服务时，可用下方的文本输入区验证翻译效果。",
    "title.sentence": (a, b) => `句对（${a} → ${b}）`,
    "title.segment": (a, b) => `模型段（${a} → ${b}）`,
    "count.sentences": (n) => `${n} 句`,
    "count.segments": (n) => `${n} 段`,
    "pair.sentence": "sentence",
    "pair.segment": "segment",
    "pair.pending": "未完句",
    "pair.fromSegments": (n) => `${n} 段`,
    "count.chars": (n) => `${n} 字`,
    "status.idle": "尚未开始",
    "status.connected": "已连接，等待输入",
    "status.loadingTranslation": "正在连接翻译服务…",
    "status.loadingAsr": "正在连接 ASR 服务…",
    "status.mic": "麦克风采集中",
    "status.tab": "标签页采集中",
    "status.file": "音频文件播放中",
    "status.paused": "已暂停",
    "status.live": "同传进行中",
    "status.ending": "正在结束会话…",
    "status.ended": "会话已结束",
    "status.fileDone": "文件播放完成",
    "status.textStreaming": (done, total) => `文本流式送入中 ${done}/${total}`,
    "status.textDone": "文本已送完",
    "status.disconnected": "连接已断开",
    "asr.connected": "已连接",
    "asr.unconfigured": "未配置",
    "asr.connecting": "连接中",
    "asr.recognizing": "识别中",
    "asr.sentenceDone": "已完成一句",
    "asr.notConnected": "未连接",
    "err.initTimeout": "连接初始化超时",
    "err.initFailed": "服务端初始化失败",
    "err.noAudioApi": "请在 HTTPS 或 localhost 页面中使用音频采集",
    "err.noTabAudio": "当前浏览器不支持标签页音频共享",
    "err.needTabAudio": "请选择标签页并勾选共享音频",
    "err.noFile": "请先选择音频文件",
    "err.audioStopped": "音频来源已停止",
    "err.startFailed": "无法开始音频输入",
    "err.connectFailed": "无法连接服务",
    "err.server": "服务端错误",
  },
  en: {
    "label.direction": "Direction",
    "label.latency": "Latency",
    "label.display": "Display",
    "label.audioSource": "Audio source",
    "label.textInput": "Text input",
    "label.chunkSize": "Chunk",
    "label.chunkUnit": "units",
    "display.sentence": "Sentence pairs",
    "display.segment": "Model segments",
    "dir.zh2en": "Chinese → English",
    "dir.en2zh": "English → Chinese",
    "latency.low": "Low latency",
    "latency.native": "Native",
    "latency.high": "High quality",
    "src.microphone": "Microphone",
    "src.tab": "Browser tab audio",
    "src.file": "Audio file",
    "btn.start": "🎤 Start",
    "btn.upload": "📁 Upload audio",
    "btn.pause": "Pause",
    "btn.resume": "Resume",
    "btn.end": "End",
    "btn.reset": "New session",
    "btn.sendText": "Stream in",
    "btn.stopText": "Stop",
    "btn.metrics": "Metrics",
    "glossary.button": "Glossary",
    "glossary.title": "Glossary",
    "glossary.hint": "One term per line, written as \"source -> target\". Multi-line paste works.",
    "glossary.add": "Add",
    "glossary.empty": "No terms yet.",
    "glossary.locked": "Terms are read when a session starts; end the session to edit them.",
    "glossary.clear": "Clear all",
    "glossary.done": "Done",
    "glossary.invalid": "Expected \"source -> target\".",
    "glossary.full": `At most ${MAX_TERMS} terms.`,
    "panel.compare": "Full comparison",
    "panel.fullSource": "Full source",
    "panel.fullTarget": "Full translation",
    "placeholder.source": "Waiting for source…",
    "placeholder.translation": "Translations appear once the model commits a segment.",
    "placeholder.textInput": "Paste or type a passage, Ctrl+Enter to stream it in",
    "metric.segments": "Segments",
    "metric.calls": "Model calls",
    "metric.buffer": "Buffered units",
    "metric.asr": "ASR",
    "hint.asr": "With no ASR service configured, use the text input below to check translation.",
    "title.sentence": (a, b) => `Sentence pairs (${a} → ${b})`,
    "title.segment": (a, b) => `Model segments (${a} → ${b})`,
    "count.sentences": (n) => `${n} sentences`,
    "count.segments": (n) => `${n} segments`,
    "pair.sentence": "sentence",
    "pair.segment": "segment",
    "pair.pending": "incomplete",
    "pair.fromSegments": (n) => `${n} segments`,
    "count.chars": (n) => `${n} words`,
    "status.idle": "Not started",
    "status.connected": "Connected, waiting for input",
    "status.loadingTranslation": "Connecting to the translation service…",
    "status.loadingAsr": "Connecting to the ASR service…",
    "status.mic": "Capturing microphone",
    "status.tab": "Capturing tab audio",
    "status.file": "Playing audio file",
    "status.paused": "Paused",
    "status.live": "Translating",
    "status.ending": "Ending the session…",
    "status.ended": "Session ended",
    "status.fileDone": "File finished",
    "status.textStreaming": (done, total) => `Streaming text ${done}/${total}`,
    "status.textDone": "Text sent",
    "status.disconnected": "Disconnected",
    "asr.connected": "Connected",
    "asr.unconfigured": "Not configured",
    "asr.connecting": "Connecting",
    "asr.recognizing": "Recognizing",
    "asr.sentenceDone": "Sentence complete",
    "asr.notConnected": "Not connected",
    "err.initTimeout": "Timed out during initialization",
    "err.initFailed": "The server failed to initialize the session",
    "err.noAudioApi": "Audio capture needs an HTTPS or localhost page",
    "err.noTabAudio": "This browser cannot share tab audio",
    "err.needTabAudio": "Pick a tab and tick \"share audio\"",
    "err.noFile": "Choose an audio file first",
    "err.audioStopped": "The audio source stopped",
    "err.startFailed": "Could not start audio input",
    "err.connectFailed": "Could not reach the service",
    "err.server": "Server error",
  },
};

let lang = (() => {
  try {
    const saved = localStorage.getItem(LANG_STORAGE_KEY);
    if (saved === "zh" || saved === "en") return saved;
  } catch (_) {}
  return "zh";
})();

function t(key, ...args) {
  const value = STRINGS[lang][key] ?? STRINGS.zh[key] ?? key;
  return typeof value === "function" ? value(...args) : value;
}

function applyLanguage() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  $("textInput").placeholder = t("placeholder.textInput");
  $("glossaryButton").querySelector("[data-i18n]").textContent = t("glossary.button");
  const toggle = $("uiLanguage");
  toggle.querySelector(".lang-active").textContent = lang === "zh" ? "ZH" : "EN";
  toggle.querySelector(".lang-idle").textContent = lang === "zh" ? "EN" : "ZH";
  $("pause").textContent = state.paused ? t("btn.resume") : t("btn.pause");
  updateLabels();
  renderSource();
  renderPairs();
  renderGlossary();
  updateControls();
}

// --- glossary -------------------------------------------------------------
// Terms are sent once in the init frame, so the server pins one glossary per
// session and editing is locked while a session runs.
let terms = (() => {
  try {
    const raw = JSON.parse(localStorage.getItem(TERMS_STORAGE_KEY) || "[]");
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((item) => item && typeof item.src === "string" && typeof item.trg === "string")
      .slice(0, MAX_TERMS);
  } catch (_) {
    return [];
  }
})();

function saveTerms() {
  try {
    localStorage.setItem(TERMS_STORAGE_KEY, JSON.stringify(terms));
  } catch (_) {
    // A full or blocked storage should not break the session.
  }
}

// Accepts "src -> trg", "src->trg", "src=trg" and full-width arrows, so pasted
// glossaries from different sources land without hand-editing.
function parseTermLine(line) {
  const text = String(line || "").trim();
  if (!text || text.startsWith("#")) return null;
  const match = text.match(/^(.*?)\s*(?:->|=>|→|＝|=)\s*(.*)$/);
  if (!match) return null;
  const src = match[1].trim();
  const trg = match[2].trim();
  return src && trg ? { src, trg } : null;
}

function addTermLines(raw) {
  let added = 0;
  let rejected = 0;
  for (const line of String(raw || "").split("\n")) {
    if (!line.trim()) continue;
    const term = parseTermLine(line);
    if (!term) {
      rejected += 1;
      continue;
    }
    if (terms.length >= MAX_TERMS) return { added, rejected, full: true };
    if (terms.some((item) => item.src === term.src && item.trg === term.trg)) continue;
    terms.push(term);
    added += 1;
  }
  return { added, rejected, full: false };
}

function glossaryLocked() {
  return state.recording || state.ending;
}

function renderGlossary() {
  const locked = glossaryLocked();
  $("glossaryBadge").textContent = String(terms.length);
  $("glossaryButton").dataset.glossaryCount = String(terms.length);
  $("glossaryCount").textContent = `${terms.length} / ${MAX_TERMS}`;
  $("glossaryEmpty").hidden = terms.length > 0;
  $("glossaryLocked").hidden = !locked;
  $("glossaryInput").disabled = locked;
  $("glossaryAdd").disabled = locked;
  $("glossaryClear").disabled = locked || terms.length === 0;

  const list = $("glossaryList");
  list.replaceChildren();
  terms.forEach((term, index) => {
    const item = document.createElement("li");
    item.className = "glossary-chip";
    const text = document.createElement("span");
    text.className = "glossary-chip-text";
    text.textContent = `${term.src} → ${term.trg}`;
    item.appendChild(text);
    if (!locked) {
      const remove = document.createElement("button");
      remove.className = "glossary-chip-remove";
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${term.src}`);
      remove.addEventListener("click", () => {
        terms.splice(index, 1);
        saveTerms();
        renderGlossary();
      });
      item.appendChild(remove);
    }
    list.appendChild(item);
  });
}

function showGlossaryError(message) {
  const node = $("glossaryError");
  node.textContent = message || "";
  node.hidden = !message;
}

function openGlossary() {
  $("glossaryPanel").hidden = false;
  $("glossaryButton").setAttribute("aria-expanded", "true");
  showGlossaryError("");
  renderGlossary();
  if (!glossaryLocked()) $("glossaryInput").focus();
}

function closeGlossary() {
  $("glossaryPanel").hidden = true;
  $("glossaryButton").setAttribute("aria-expanded", "false");
}

// --- rendering ------------------------------------------------------------
function setStatus(message, kind = "idle") {
  $("status").textContent = message;
  $("statusDot").className = `dot${kind === "live" ? " live" : kind === "error" ? " error" : ""}`;
}

function directionLabels() {
  return state.direction === "zh2en"
    ? [t("dir.zh2en").split(" → ")[0], t("dir.zh2en").split(" → ")[1]]
    : [t("dir.en2zh").split(" → ")[0], t("dir.en2zh").split(" → ")[1]];
}

function updateLabels() {
  state.direction = $("direction").value;
  state.latencyMode = $("latencyMode").value;
  state.displayMode = $("displayMode").value;
  const [from, to] = directionLabels();
  $("targetTitle").textContent = state.displayMode === "sentence"
    ? t("title.sentence", from, to)
    : t("title.segment", from, to);
}

// The server sends only the newly fixed increment, so the transcript
// accumulates rather than being replaced.
function appendSource(increment) {
  const value = String(increment || "");
  if (value) state.sourceText += value;
  renderSource();
}

function renderSource() {
  const container = $("fullSourceText");
  container.textContent = state.sourceText || "";
  if (!state.sourceText) {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.textContent = t("placeholder.source");
    container.appendChild(placeholder);
  }
  container.scrollTop = container.scrollHeight;
}

function appendNatural(left, right, language) {
  const previous = String(left || "").trimEnd();
  const next = String(right || "").trimStart();
  if (!previous) return next;
  if (!next) return previous;
  if (language === "zh") return `${previous}${next}`;
  return `${previous} ${next}`
    .replace(/\s+([,.;:!?%\)\]\}])/g, "$1")
    .replace(/([\(\[\{])\s+/g, "$1")
    .replace(/\b([A-Za-z]+)\s+('(?:s|re|ve|ll|d|m|t)\b)/g, "$1$2");
}

// --- sentence grouping ----------------------------------------------------
// The model commits segments, which are usually shorter than a sentence. The
// default view merges consecutive segments up to a sentence-final mark, so the
// reading order matches how the text would be spoken; "model segments" shows
// the raw commits instead.
function sentenceMarks(sourceSide) {
  const chineseSide = sourceSide ? state.direction === "zh2en" : state.direction === "en2zh";
  return new Set(chineseSide ? [..."。！？!?；;…～~"] : [...".!?;"]);
}

function sourceEndsSentence(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return false;
  const terminal = state.direction === "zh2en" ? "[。！？!?；;…～~]" : "[.!?;]";
  return new RegExp(`${terminal}[\\s”’"'）)\\]]*$`, "u").test(trimmed);
}

// A single commit can contain more than one sentence. Splitting it lets the
// grouping below close a sentence mid-segment instead of carrying the tail into
// the next card.
function splitSentenceText(text, sourceSide) {
  const value = String(text || "");
  const marks = sentenceMarks(sourceSide);
  const closing = new Set([..."”’\"'）)]】」』》 "]);
  const parts = [];
  let start = 0;
  for (let index = 0; index < value.length; index += 1) {
    if (!marks.has(value[index])) continue;
    let end = index + 1;
    while (end < value.length && closing.has(value[end])) end += 1;
    const sentence = value.slice(start, end).trim();
    if (sentence) parts.push(sentence);
    start = end;
    index = end - 1;
  }
  const remainder = value.slice(start).trim();
  if (remainder) parts.push(remainder);
  return parts.length ? parts : [value];
}

// Only split when both sides agree on the sentence count; otherwise the halves
// would be misaligned, which is worse than one oversized card.
function expandSentenceAlignedPair(pair) {
  const sourceParts = splitSentenceText(pair.source, true);
  const targetParts = splitSentenceText(pair.text, false);
  if (sourceParts.length <= 1 || sourceParts.length !== targetParts.length) return [pair];
  return sourceParts.map((source, index) => ({ source, text: targetParts[index] }));
}

function buildSentencePairs(pairs) {
  const sourceLanguage = state.direction === "zh2en" ? "zh" : "en";
  const targetLanguage = state.direction === "zh2en" ? "en" : "zh";
  const sentences = [];
  let source = "";
  let text = "";
  let segments = 0;
  for (const raw of pairs) {
    for (const pair of expandSentenceAlignedPair(raw)) {
      source = appendNatural(source, pair.source, sourceLanguage);
      text = appendNatural(text, pair.text, targetLanguage);
      segments += 1;
      if (sourceEndsSentence(source)) {
        sentences.push({ source, text, segments, pending: false });
        source = "";
        text = "";
        segments = 0;
      }
    }
  }
  // The trailing group has no sentence-final mark yet: it is shown, but marked
  // so it does not read as a finished sentence.
  if (source || text) sentences.push({ source, text, segments, pending: true });
  return sentences;
}

function visiblePairs() {
  if (state.displayMode !== "sentence") {
    return state.pairs.map((pair) => ({ ...pair, segments: 1, pending: false }));
  }
  return buildSentencePairs(state.pairs);
}

function renderPairs() {
  const container = $("translation");
  const pairs = visiblePairs();
  container.replaceChildren();
  if (!pairs.length) {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder translation-placeholder";
    placeholder.textContent = t("placeholder.translation");
    container.appendChild(placeholder);
  }
  const sentenceMode = state.displayMode === "sentence";
  pairs.forEach((pair, index) => {
    const card = document.createElement("article");
    const classes = ["pair-card"];
    if (index === pairs.length - 1) classes.push("current");
    if (pair.pending) classes.push("pending");
    card.className = classes.join(" ");

    const header = document.createElement("div");
    header.className = "pair-header";
    const label = document.createElement("span");
    label.textContent = sentenceMode ? t("pair.sentence") : t("pair.segment");
    header.appendChild(label);
    if (pair.pending) {
      const pending = document.createElement("span");
      pending.className = "pair-pending";
      pending.textContent = t("pair.pending");
      header.appendChild(pending);
    }
    const detail = document.createElement("span");
    detail.className = "pair-detail";
    // In sentence mode the segment count says how many commits were merged,
    // which is the detail lost by grouping.
    detail.textContent = sentenceMode && pair.segments > 1
      ? `#${index + 1} · ${t("pair.fromSegments", pair.segments)}`
      : `#${index + 1}`;
    header.appendChild(detail);
    card.appendChild(header);

    for (const [cls, text, tag] of [
      ["pair-source", pair.source, "SRC"],
      ["pair-target", pair.text, "MT"],
    ]) {
      const row = document.createElement("div");
      row.className = `pair-row ${cls}`;
      const rowLabel = document.createElement("span");
      rowLabel.className = "pair-label";
      rowLabel.textContent = tag;
      const value = document.createElement("span");
      value.className = "pair-value";
      value.textContent = text;
      row.append(rowLabel, value);
      card.appendChild(row);
    }
    container.appendChild(card);
  });

  $("segmentCount").textContent = sentenceMode
    ? t("count.sentences", pairs.length)
    : t("count.segments", pairs.length);
  container.scrollTop = container.scrollHeight;

  // The full-text panel always joins the raw commits, independent of the view.
  const targetLanguage = state.direction === "zh2en" ? "en" : "zh";
  const full = state.pairs.reduce(
    (acc, pair) => appendNatural(acc, pair.text, targetLanguage),
    "",
  );
  $("fullTranslationText").textContent = full;
  $("fullTranslationText").scrollTop = $("fullTranslationText").scrollHeight;
}

// --- transport ------------------------------------------------------------
function sendJson(payload) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN || !state.initialized) return false;
  state.socket.send(JSON.stringify(payload));
  return true;
}

function socketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/simul-demo`;
}

function connect() {
  if (state.socket?.readyState === WebSocket.OPEN && state.initialized) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(socketUrl());
    socket.binaryType = "arraybuffer";
    state.socket = socket;
    const timer = window.setTimeout(() => {
      reject(new Error(t("err.initTimeout")));
      socket.close();
    }, 30 * 60 * 1000);
    socket.onopen = () =>
      socket.send(
        JSON.stringify({
          type: "init",
          direction: state.direction,
          latency_mode: state.latencyMode,
          // Segmentation policy for live speech. A 15-unit force break with
          // sentence-final forcing keeps segments short enough to read as
          // subtitles, and a 0.8s silence flush translates a trailing clause
          // instead of leaving it on screen: a speaker finishing a clause
          // rarely pauses longer, and going lower risks cutting mid-clause.
          force_break_threshold: 15,
          punct_force: true,
          idle_force_seconds: 0.8,
          // Pinned for the session: the glossary is part of the prompt, so
          // editing it mid-stream would invalidate the prefix cache.
          terms,
        }),
      );
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      handleMessage(message);
      if (message.type === "init_ok") {
        window.clearTimeout(timer);
        state.initialized = true;
        resolve();
      }
      if (message.type === "error" && !state.initialized) {
        window.clearTimeout(timer);
        reject(new Error(message.message || t("err.initFailed")));
      }
    };
    socket.onerror = () => {};
    socket.onclose = () => {
      window.clearTimeout(timer);
      const wasActive = state.recording;
      state.socket = null;
      state.initialized = false;
      if (wasActive && !state.ending) {
        state.recording = false;
        stopCapture();
        setStatus(t("status.disconnected"), "error");
        updateControls();
      }
    };
  });
}

function handleMessage(message) {
  switch (message.type) {
    case "init_ok":
      state.asrConfigured = Boolean(message.asr_configured);
      $("asrState").textContent = state.asrConfigured
        ? t("asr.connected")
        : t("asr.unconfigured");
      $("asrHint").hidden = state.asrConfigured;
      setStatus(t("status.connected"), "live");
      renderGlossary();
      break;
    case "loading":
      if (message.component === "asr") {
        $("asrState").textContent = t("asr.connecting");
        setStatus(t("status.loadingAsr"));
      } else {
        setStatus(t("status.loadingTranslation"));
      }
      break;
    case "asr":
      appendSource(message.text);
      $("asrState").textContent = message.reset
        ? t("asr.sentenceDone")
        : t("asr.recognizing");
      break;
    case "translation":
      if (message.text) {
        state.pairs.push({ source: String(message.source || ""), text: String(message.text || "") });
        renderPairs();
      }
      break;
    case "metrics":
      $("segments").textContent = String(message.segments ?? state.pairs.length);
      $("calls").textContent = String((message.probe_calls || 0) + (message.translation_calls || 0));
      $("bufferUnits").textContent = String(message.buffer_units || 0);
      break;
    case "pause_ok":
      setStatus(t("status.paused"));
      break;
    case "resume_ok":
      setStatus(t("status.live"), "live");
      break;
    case "ended":
      state.recording = false;
      state.ending = false;
      stopCapture();
      setStatus(t("status.ended"));
      updateControls();
      renderGlossary();
      break;
    case "error":
      setStatus(message.message || t("err.server"), "error");
      break;
    default:
      break;
  }
}

// --- audio ----------------------------------------------------------------
function resample(input, inputRate, outputRate = 16000) {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const output = new Float32Array(Math.max(1, Math.round(input.length / ratio)));
  for (let i = 0; i < output.length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    output[i] = sum / Math.max(1, end - start);
  }
  return output;
}

function pcm16(float32) {
  const output = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const value = Math.max(-1, Math.min(1, float32[i]));
    output[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return output.buffer;
}

async function acquireStream(mode) {
  if (!navigator.mediaDevices) throw new Error(t("err.noAudioApi"));
  if (mode === "tab") {
    if (!navigator.mediaDevices.getDisplayMedia) throw new Error(t("err.noTabAudio"));
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true, audio: true, selfBrowserSurface: "exclude",
    });
    if (!stream.getAudioTracks().length) {
      stream.getTracks().forEach((track) => track.stop());
      throw new Error(t("err.needTabAudio"));
    }
    return stream;
  }
  return navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
}

async function startLive(mode) {
  await connect();
  state.inputMode = mode;
  state.mediaStream = await acquireStream(mode);
  state.audioContext = new AudioContext();
  state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
  state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
  state.silentGain = state.audioContext.createGain();
  state.silentGain.gain.value = 0;
  state.processor.onaudioprocess = (event) => {
    if (!state.recording || state.paused || !state.socket || state.socket.readyState !== WebSocket.OPEN) return;
    const samples = resample(event.inputBuffer.getChannelData(0), state.audioContext.sampleRate);
    if (samples.length) state.socket.send(pcm16(samples));
  };
  state.sourceNode.connect(state.processor);
  state.processor.connect(state.silentGain);
  state.silentGain.connect(state.audioContext.destination);
  state.mediaStream.getTracks().forEach((track) =>
    track.addEventListener("ended", () => finish(t("err.audioStopped")), { once: true }),
  );
  state.recording = true;
  state.paused = false;
  setStatus(mode === "tab" ? t("status.tab") : t("status.mic"), "live");
  updateControls();
  renderGlossary();
}

async function startFile(file) {
  if (!file) return;
  await connect();
  state.fileContext = new AudioContext();
  const decoded = await state.fileContext.decodeAudioData(await file.arrayBuffer());
  const mono = new Float32Array(decoded.length);
  for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
    const samples = decoded.getChannelData(channel);
    for (let i = 0; i < decoded.length; i += 1) mono[i] += samples[i] / decoded.numberOfChannels;
  }
  state.fileSamples = resample(mono, decoded.sampleRate);
  state.fileOffset = 0;
  state.fileDuration = state.fileSamples.length / 16000;
  $("playerRow").classList.remove("hidden");
  $("fileName").textContent = file.name;
  // Optional local playback so the operator can hear what is being streamed.
  // Revoked on reset; failure here must not stop the session.
  try {
    const player = $("audioPlayer");
    if (player.src) URL.revokeObjectURL(player.src);
    player.src = URL.createObjectURL(file);
    $("play").disabled = false;
  } catch (_) {}
  state.recording = true;
  state.paused = false;
  setStatus(t("status.file"), "live");
  updateControls();
  renderGlossary();
  streamFile();
}

function streamFile() {
  if (!state.recording || state.paused || !state.fileSamples) return;
  const size = 1280; // 80 ms at 16 kHz
  const end = Math.min(state.fileSamples.length, state.fileOffset + size);
  if (end <= state.fileOffset) {
    finish(t("status.fileDone"));
    return;
  }
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(pcm16(state.fileSamples.slice(state.fileOffset, end)));
  }
  state.fileOffset = end;
  $("progress").value = (state.fileOffset / state.fileSamples.length) * 100;
  $("audioTime").textContent = `${formatTime(state.fileOffset / 16000)} / ${formatTime(state.fileDuration)}`;
  state.fileTimer = window.setTimeout(streamFile, 80);
}

function stopCapture() {
  if (state.fileTimer) window.clearTimeout(state.fileTimer);
  state.fileTimer = null;
  try { state.processor?.disconnect(); } catch (_) {}
  try { state.sourceNode?.disconnect(); } catch (_) {}
  try { state.silentGain?.disconnect(); } catch (_) {}
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
  if (state.audioContext && state.audioContext.state !== "closed") state.audioContext.close().catch(() => {});
  if (state.fileContext && state.fileContext.state !== "closed") state.fileContext.close().catch(() => {});
  state.audioContext = null;
  state.fileContext = null;
  state.processor = null;
  state.sourceNode = null;
  state.silentGain = null;
}

// --- controls -------------------------------------------------------------
function finish(message = t("status.ending")) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    state.recording = false;
    stopCapture();
    updateControls();
    renderGlossary();
    return;
  }
  state.ending = true;
  state.recording = false;
  stopCapture();
  setStatus(message);
  sendJson({ type: "end" });
  $("end").disabled = true;
}

function togglePause() {
  if (!state.recording) return;
  state.paused = !state.paused;
  sendJson({ type: state.paused ? "pause" : "resume" });
  $("pause").textContent = state.paused ? t("btn.resume") : t("btn.pause");
  if (!state.paused && state.fileSamples) streamFile();
}

function updateControls() {
  const active = state.recording || state.ending;
  $("record").disabled = active;
  $("pause").disabled = !state.recording;
  $("end").disabled = !state.recording && !state.ending;
  $("direction").disabled = active;
  // Both are bound when the session is created and cannot change mid-session.
  $("latencyMode").disabled = active;
  $("audioInputMode").disabled = active;
}

async function startSelected() {
  try {
    const mode = $("audioInputMode").value;
    if (mode === "file") {
      const file = $("fileInput").files[0];
      if (!file) throw new Error(t("err.noFile"));
      await startFile(file);
    } else {
      await startLive(mode);
    }
  } catch (error) {
    stopCapture();
    state.recording = false;
    setStatus(error.message || t("err.startFailed"), "error");
    updateControls();
    renderGlossary();
  }
}

// Split a passage into the model's source units: characters for zh2en (keeping
// runs of Latin/digits together so a word is not cut mid-token), whitespace
// tokens for en2zh. Mirrors split_source() on the server.
function splitSourceUnits(text) {
  const value = String(text || "");
  if (state.direction !== "zh2en") {
    return value.split(/\s+/).filter(Boolean);
  }
  const units = [];
  let latin = "";
  for (const ch of value) {
    if (/[A-Za-z0-9]/.test(ch)) {
      latin += ch;
      continue;
    }
    if (latin) {
      units.push(latin);
      latin = "";
    }
    if (!/\s/.test(ch)) units.push(ch);
  }
  if (latin) units.push(latin);
  return units;
}

function stopTextStream(message) {
  if (state.textTimer) window.clearTimeout(state.textTimer);
  state.textTimer = null;
  state.textQueue = [];
  $("sendText").hidden = false;
  $("stopText").hidden = true;
  $("textInput").disabled = false;
  if (message) setStatus(message);
}

// Feed the queued chunks one interval at a time. Each chunk is one `text`
// command, which is exactly what an ASR increment would be, so the model sees
// the same arrival pattern as live speech.
function pumpTextStream() {
  if (!state.textQueue.length) {
    const total = state.textTotal;
    stopTextStream(t("status.textDone"));
    // Close the utterance so the tail is translated rather than left buffered.
    sendJson({ type: "end" });
    state.textTotal = total;
    return;
  }
  const chunk = state.textQueue.shift();
  const sent = sendJson({ type: "text", text: chunk });
  if (!sent) {
    stopTextStream(t("err.connectFailed"));
    return;
  }
  setStatus(
    t("status.textStreaming", state.textTotal - state.textQueue.length, state.textTotal),
    "live",
  );
  state.textTimer = window.setTimeout(pumpTextStream, TEXT_CHUNK_INTERVAL_MS);
}

async function sendText() {
  const value = $("textInput").value.trim();
  if (!value) return;
  if (state.textTimer) return;
  try {
    await connect();
  } catch (error) {
    setStatus(error.message || t("err.connectFailed"), "error");
    return;
  }
  const units = splitSourceUnits(value);
  if (!units.length) return;
  const size = Math.max(1, Number($("chunkSize").value) || 2);
  const joiner = state.direction === "zh2en" ? "" : " ";
  const queue = [];
  for (let i = 0; i < units.length; i += size) {
    queue.push(units.slice(i, i + size).join(joiner));
  }
  state.textQueue = queue;
  state.textTotal = queue.length;
  $("sendText").hidden = true;
  $("stopText").hidden = false;
  $("textInput").disabled = true;
  pumpTextStream();
}

function resetSession() {
  stopTextStream();
  state.ending = false;
  state.recording = false;
  state.paused = false;
  stopCapture();
  try { state.socket?.close(1000, "new session"); } catch (_) {}
  state.socket = null;
  state.initialized = false;
  state.sourceText = "";
  state.pairs = [];
  renderSource();
  renderPairs();
  $("segments").textContent = "0";
  $("calls").textContent = "0";
  $("bufferUnits").textContent = "0";
  $("asrState").textContent = t("asr.notConnected");
  $("playerRow").classList.add("hidden");
  const player = $("audioPlayer");
  try {
    player.pause();
    if (player.src) URL.revokeObjectURL(player.src);
    player.removeAttribute("src");
  } catch (_) {}
  $("play").disabled = true;
  $("play").textContent = "▶";
  $("fileInput").value = "";
  $("progress").value = 0;
  $("audioTime").textContent = "0:00 / 0:00";
  setStatus(t("status.idle"));
  updateControls();
  renderGlossary();
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  return `${minutes}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
}

// --- wiring ---------------------------------------------------------------
$("direction").addEventListener("change", () => {
  updateLabels();
  renderPairs();
});
$("latencyMode").addEventListener("change", () => {
  updateLabels();
  // A live socket already bound the previous mode; drop it so the next start
  // re-inits with the new one.
  if (state.socket && !state.recording && !state.ending) {
    try { state.socket.close(1000, "latency mode changed"); } catch (_) {}
    state.socket = null;
    state.initialized = false;
  }
  setStatus(`${t("label.latency")}: ${$("latencyMode").selectedOptions[0].textContent}`);
});
$("displayMode").addEventListener("change", () => {
  // Presentation only: it regroups what has already been committed, so unlike
  // direction and latency it is safe to change mid-session.
  updateLabels();
  renderPairs();
});
$("audioInputMode").addEventListener("change", (event) => {
  state.inputMode = event.target.value;
  updateControls();
});
$("record").addEventListener("click", startSelected);
$("fileInput").addEventListener("change", () => {
  if ($("audioInputMode").value === "file") startSelected();
});
$("pause").addEventListener("click", togglePause);
$("end").addEventListener("click", () => finish());
$("reset").addEventListener("click", resetSession);
$("play").addEventListener("click", () => {
  const player = $("audioPlayer");
  if (!player.src) return;
  if (player.paused) {
    player.play().catch(() => {});
    $("play").textContent = "⏸";
  } else {
    player.pause();
    $("play").textContent = "▶";
  }
});
$("sendText").addEventListener("click", sendText);
$("stopText").addEventListener("click", () => stopTextStream(t("status.idle")));
$("textInput").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") sendText();
});

$("fullCompareToggle").addEventListener("click", () => {
  const panel = $("fullCompare");
  const collapsed = panel.classList.toggle("is-collapsed");
  $("fullCompareToggle").setAttribute("aria-expanded", collapsed ? "false" : "true");
});

$("glossaryButton").addEventListener("click", openGlossary);
$("glossaryBackdrop").addEventListener("click", closeGlossary);
$("glossaryDone").addEventListener("click", closeGlossary);
$("glossaryForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("glossaryInput");
  const { added, rejected, full } = addTermLines(input.value);
  if (added) {
    saveTerms();
    input.value = "";
  }
  showGlossaryError(full ? t("glossary.full") : rejected && !added ? t("glossary.invalid") : "");
  renderGlossary();
});
$("glossaryClear").addEventListener("click", () => {
  terms = [];
  saveTerms();
  showGlossaryError("");
  renderGlossary();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("glossaryPanel").hidden) closeGlossary();
});

function applyMetricsVisibility(visible) {
  $("metricsBar").classList.toggle("hidden", !visible);
  $("metricsToggle").setAttribute("aria-expanded", visible ? "true" : "false");
  try { localStorage.setItem(METRICS_STORAGE_KEY, visible ? "1" : "0"); } catch (_) {}
}

$("metricsToggle").addEventListener("click", () => {
  applyMetricsVisibility($("metricsBar").classList.contains("hidden"));
});

$("uiLanguage").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  try { localStorage.setItem(LANG_STORAGE_KEY, lang); } catch (_) {}
  applyLanguage();
});

applyLanguage();
(() => {
  let visible = false;
  try { visible = localStorage.getItem(METRICS_STORAGE_KEY) === "1"; } catch (_) {}
  applyMetricsVisibility(visible);
})();
resetSession();
