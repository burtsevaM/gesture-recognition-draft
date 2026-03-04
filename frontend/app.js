const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('overlay');
const ctx = canvasEl.getContext('2d');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const rawSkeletonToggleEl = document.getElementById('rawSkeletonToggle');
const normSkeletonToggleEl = document.getElementById('normSkeletonToggle');
const inferenceLocationEl = document.getElementById('inferenceLocation');
const inferenceStatusEl = document.getElementById('inferenceStatus');

const statusEl = document.getElementById('status');
const tokenLabelEl = document.getElementById('tokenLabel');
const letterEl = document.getElementById('letter');
const scoreEl = document.getElementById('score');
const confidenceEl = document.getElementById('confidence');
const holdEl = document.getElementById('hold');
const remainEl = document.getElementById('remain');
const progressBar = document.getElementById('progressBar');
const textValueEl = document.getElementById('textValue');
const committedWordsEl = document.getElementById('committedWords');
const sentencesListEl = document.getElementById('sentencesList');
const topkEl = document.getElementById('topk');
const debugEl = document.getElementById('debug');
const dbgModeEl = document.getElementById('dbgMode');
const dbgLatencyEl = document.getElementById('dbgLatency');
const dbgFpsCamEl = document.getElementById('dbgFpsCam');
const dbgFpsInEl = document.getElementById('dbgFpsIn');
const dbgFpsPoseEl = document.getElementById('dbgFpsPose');
const dbgFpsTotalEl = document.getElementById('dbgFpsTotal');
const dbgSegEnabledEl = document.getElementById('dbgSegEnabled');
const dbgWinStepEl = document.getElementById('dbgWinStep');
const dbgThEl = document.getElementById('dbgTh');
const dbgMinMergeEl = document.getElementById('dbgMinMerge');
const dbgLastSegEl = document.getElementById('dbgLastSeg');
const segmentsTimelineEl = document.getElementById('segmentsTimeline');

const vlmUsedEl = document.getElementById('vlmUsed');
const vlmLetterEl = document.getElementById('vlmLetter');
const vlmConfEl = document.getElementById('vlmConf');
const vlmReasonEl = document.getElementById('vlmReason');

const BODY_EDGES = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
];

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const MIN_LANDMARK_CONF = 0.05;
const INFERENCE_LOCATION_STORAGE_KEY = 'inference_location';
const OFFLINE_MODEL_BASE = '/static/assets/models';
const OFFLINE_WORD_MODEL_URL = `${OFFLINE_MODEL_BASE}/pose_word_model.onnx`;
const OFFLINE_BIO_MODEL_URL = `${OFFLINE_MODEL_BASE}/bio_segmenter.onnx`;
const OFFLINE_MANIFEST_URL = `${OFFLINE_MODEL_BASE}/manifest.json`;

// Важно: датасет снят с зеркалом, поэтому live-кадр в распознавание тоже зеркалим.
const MIRROR_STREAM = true;

let stream = null;
let ws = null;
let sendTimer = null;
let renderReq = null;
let sendBusy = false;
let latest = null;
let sendFps = 12;
let jpegQuality = 0.75;
let shouldRun = false;
let reconnectTimer = null;
let lastStateAtMs = 0;
let awaitingServer = false;
let lastSendAtMs = 0;
let stableVisibleLetter = 'NONE';
let recognitionMode = 'letters';
let committedTokens = [];
let lastCommitToken = '';
let lastCommitAtMs = 0;
let cameraFps = 0;
let cameraFpsCounter = 0;
let cameraFpsWindowStartMs = 0;
let lastServerPerf = null;
let segmentCommitFlashUntilMs = 0;
let segmentHistory = [];
let segmentEventSeen = new Set();
let lastSegmentInfoText = 'none';
let inferenceLocation = 'backend';
let effectiveInferenceLocation = 'backend';
let browserInferenceBackend = null;
let backendInferenceBackend = null;
const TIMELINE_WINDOW_MS = 8000;

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

function drawVideoFrame(targetCtx, sourceVideo, width, height, mirror = false) {
  targetCtx.save();
  if (mirror) {
    targetCtx.translate(width, 0);
    targetCtx.scale(-1, 1);
  }
  targetCtx.drawImage(sourceVideo, 0, 0, width, height);
  targetCtx.restore();
}

function applyMirrorStyles() {
  const transform = MIRROR_STREAM ? 'scaleX(-1)' : 'none';
  videoEl.style.transform = transform;
  canvasEl.style.transform = transform;
  videoEl.style.transformOrigin = 'center center';
  canvasEl.style.transformOrigin = 'center center';
}

function normalizeInferenceLocation(value) {
  return String(value || '').toLowerCase() === 'browser' ? 'browser' : 'backend';
}

function setInferenceStatus(text, tone = 'normal') {
  if (!inferenceStatusEl) return;
  inferenceStatusEl.textContent = String(text || '');
  inferenceStatusEl.classList.remove('warn', 'ok', 'err');
  if (tone === 'warn') inferenceStatusEl.classList.add('warn');
  if (tone === 'ok') inferenceStatusEl.classList.add('ok');
  if (tone === 'err') inferenceStatusEl.classList.add('err');
}

function syncInferenceSelector() {
  if (!inferenceLocationEl) return;
  inferenceLocationEl.value = normalizeInferenceLocation(inferenceLocation);
}

function loadInferencePreference() {
  try {
    inferenceLocation = normalizeInferenceLocation(localStorage.getItem(INFERENCE_LOCATION_STORAGE_KEY));
  } catch (err) {
    inferenceLocation = 'backend';
  }
  syncInferenceSelector();
}

function saveInferencePreference(value) {
  inferenceLocation = normalizeInferenceLocation(value);
  try {
    localStorage.setItem(INFERENCE_LOCATION_STORAGE_KEY, inferenceLocation);
  } catch (err) {
    // ignore storage failures
  }
  syncInferenceSelector();
}

async function initializeInferenceBackends() {
  const offline = window.OfflineInference || {};
  const BackendClass = offline.BackendWsInferenceBackend;
  if (typeof BackendClass === 'function') {
    backendInferenceBackend = new BackendClass();
    try {
      await backendInferenceBackend.init();
    } catch (err) {
      // keep backend path via existing WS logic
    }
  } else {
    backendInferenceBackend = null;
  }

  effectiveInferenceLocation = 'backend';
  browserInferenceBackend = null;

  if (inferenceLocation !== 'browser') {
    setInferenceStatus('Используется backend WebSocket.', 'normal');
    return;
  }

  const BrowserClass = offline.BrowserOrtInferenceBackend;
  if (typeof BrowserClass !== 'function') {
    setInferenceStatus('Browser mode недоступен: модуль ORT не подключен. Используется backend.', 'warn');
    return;
  }

  browserInferenceBackend = new BrowserClass();
  const result = await browserInferenceBackend.init({
    wordModelUrl: OFFLINE_WORD_MODEL_URL,
    bioModelUrl: OFFLINE_BIO_MODEL_URL,
    manifestUrl: OFFLINE_MANIFEST_URL,
    executionProviders: ['wasm'],
    wasmThreads: 1,
  });

  if (result?.ok) {
    setInferenceStatus('Browser mode experimental: модели загружены, активен fallback на backend WS.', 'ok');
  } else {
    const reason = String(result?.reason || browserInferenceBackend?.lastError || 'init failed');
    setInferenceStatus(`Browser mode experimental недоступен (${reason}). Используется backend WS.`, 'warn');
  }
}

function setupInferenceControls() {
  loadInferencePreference();
  if (!inferenceLocationEl) return;
  inferenceLocationEl.addEventListener('change', () => {
    const selected = normalizeInferenceLocation(inferenceLocationEl.value);
    saveInferencePreference(selected);
    if (stream) {
      setInferenceStatus('Новый inference_location применится после перезапуска камеры.', 'warn');
    } else {
      setInferenceStatus(
        selected === 'browser'
          ? 'Browser mode experimental выбран. Запустите камеру для инициализации.'
          : 'Используется backend WebSocket.',
        selected === 'browser' ? 'warn' : 'normal',
      );
    }
  });
}

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/ws/stream`;
}

function compactSpaces(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function formatTranscript(tokens) {
  if (!Array.isArray(tokens) || tokens.length === 0) return '';
  const raw = compactSpaces(tokens.join(' '));
  return raw.replace(/\s+([,.;:!?])/g, '$1');
}

function splitSentences(text) {
  const source = compactSpaces(text);
  if (!source) return [];

  if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
    try {
      const segmenter = new Intl.Segmenter('ru', { granularity: 'sentence' });
      const segments = [];
      for (const part of segmenter.segment(source)) {
        const sentence = compactSpaces(part.segment);
        if (sentence) {
          segments.push(sentence);
        }
      }
      if (segments.length > 0) {
        return segments;
      }
    } catch (err) {
      // fallback ниже
    }
  }

  const fallback = source.match(/[^.?!]+[.?!]*/g) || [];
  const cleaned = fallback.map((item) => compactSpaces(item)).filter(Boolean);
  if (cleaned.length > 0) {
    return cleaned;
  }
  return [source];
}

function renderSentencesPanel() {
  const transcript = formatTranscript(committedTokens);
  committedWordsEl.textContent = transcript || '—';

  const sentences = splitSentences(transcript);
  sentencesListEl.innerHTML = '';
  if (sentences.length === 0) {
    const li = document.createElement('li');
    li.className = 'small';
    li.textContent = 'Пока нет предложений.';
    sentencesListEl.appendChild(li);
    return;
  }

  for (const sentence of sentences) {
    const li = document.createElement('li');
    li.textContent = sentence;
    sentencesListEl.appendChild(li);
  }
}

function extractCommitToken(data, mode) {
  if (mode !== 'words' && mode !== 'pose_words') {
    return '';
  }

  const committedNow = Boolean(data?.text_state?.committed);
  const status = String(data?.status || '').toUpperCase();
  const isCommitState = committedNow || status === 'COMMIT' || status === 'COMMITTED';
  if (!isCommitState) {
    return '';
  }

  const token = compactSpaces(String(data?.word || data?.letter || ''));
  if (!token || token.toUpperCase() === 'NONE') {
    return '';
  }
  return token;
}

function appendCommitToken(data, mode) {
  const token = extractCommitToken(data, mode);
  if (!token) {
    return;
  }

  const nowMs = Date.now();
  if (token === lastCommitToken && (nowMs - lastCommitAtMs) < 900) {
    return;
  }

  committedTokens.push(token);
  lastCommitToken = token;
  lastCommitAtMs = nowMs;
  renderSentencesPanel();
}

function isFinitePoint(point) {
  return Array.isArray(point)
    && point.length >= 2
    && Number.isFinite(Number(point[0]))
    && Number.isFinite(Number(point[1]));
}

function hasVisiblePoint(group, idx) {
  if (!group || !Array.isArray(group.points)) return false;
  if (idx < 0 || idx >= group.points.length) return false;
  const point = group.points[idx];
  if (!isFinitePoint(point)) return false;
  if (!Array.isArray(group.confidence)) return true;
  if (idx >= group.confidence.length) return true;
  return Number(group.confidence[idx]) >= MIN_LANDMARK_CONF;
}

function mapRawPoint(point, width, height) {
  return [Number(point[0]) * width, Number(point[1]) * height];
}

function collectAllXY(groups) {
  const xs = [];
  const ys = [];
  for (const group of groups) {
    if (!group || !Array.isArray(group.points)) continue;
    group.points.forEach((point, idx) => {
      if (!hasVisiblePoint(group, idx)) return;
      xs.push(Number(point[0]));
      ys.push(Number(point[1]));
    });
  }
  return { xs, ys };
}

function makeNormTransform(groups, width, height) {
  const { xs, ys } = collectAllXY(groups);
  if (!xs.length || !ys.length) return null;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const x of xs) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
  }
  for (const y of ys) {
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const scale = 0.82 * Math.min(width / spanX, height / spanY);

  return {
    centerX: (minX + maxX) * 0.5,
    centerY: (minY + maxY) * 0.5,
    scale,
  };
}

function mapNormPoint(point, width, height, transform) {
  if (!transform) return [0, 0];
  const px = (Number(point[0]) - transform.centerX) * transform.scale + width * 0.5;
  const py = (Number(point[1]) - transform.centerY) * transform.scale + height * 0.5;
  return [px, py];
}

function drawEdges(group, edges, mapFn, style, width) {
  if (!group || !Array.isArray(group.points)) return false;
  ctx.strokeStyle = style;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  let drawn = false;
  for (const [a, b] of edges) {
    if (!hasVisiblePoint(group, a) || !hasVisiblePoint(group, b)) continue;
    const p1 = mapFn(group.points[a]);
    const p2 = mapFn(group.points[b]);
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
    drawn = true;
  }
  return drawn;
}

function drawJoints(group, mapFn, style, radius) {
  if (!group || !Array.isArray(group.points)) return false;
  ctx.fillStyle = style;
  let drawn = false;
  group.points.forEach((point, idx) => {
    if (!hasVisiblePoint(group, idx)) return;
    const p = mapFn(point);
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx.fill();
    drawn = true;
  });
  return drawn;
}

function drawSkeletonSpace(spacePayload, spaceName) {
  if (!spacePayload || typeof spacePayload !== 'object') return false;
  const body = spacePayload.body || null;
  const leftHand = spacePayload.lh || null;
  const rightHand = spacePayload.rh || null;
  const groups = [body, leftHand, rightHand];

  let mapFn = (point) => mapRawPoint(point, canvasEl.width, canvasEl.height);
  if (spaceName === 'norm') {
    const transform = makeNormTransform(groups, canvasEl.width, canvasEl.height);
    if (!transform) return false;
    mapFn = (point) => mapNormPoint(point, canvasEl.width, canvasEl.height, transform);
  }

  const palette = spaceName === 'norm'
    ? {
        body: 'rgba(255, 194, 96, 0.9)',
        left: 'rgba(118, 255, 176, 0.9)',
        right: 'rgba(255, 143, 92, 0.9)',
        joints: 'rgba(255, 255, 255, 0.88)',
      }
    : {
        body: 'rgba(69, 207, 255, 0.95)',
        left: 'rgba(121, 255, 181, 0.95)',
        right: 'rgba(255, 126, 154, 0.95)',
        joints: 'rgba(255, 255, 255, 0.9)',
      };

  let drawn = false;
  drawn = drawEdges(body, BODY_EDGES, mapFn, palette.body, 2.6) || drawn;
  drawn = drawEdges(leftHand, HAND_EDGES, mapFn, palette.left, 2.0) || drawn;
  drawn = drawEdges(rightHand, HAND_EDGES, mapFn, palette.right, 2.0) || drawn;
  drawn = drawJoints(body, mapFn, palette.joints, 2.2) || drawn;
  drawn = drawJoints(leftHand, mapFn, palette.joints, 1.8) || drawn;
  drawn = drawJoints(rightHand, mapFn, palette.joints, 1.8) || drawn;
  return drawn;
}

function drawSkeletonOverlay(data) {
  const useRaw = Boolean(rawSkeletonToggleEl?.checked);
  const useNorm = Boolean(normSkeletonToggleEl?.checked);
  if (!useRaw && !useNorm) return false;
  const skeleton = data?.skeleton;
  if (!skeleton || typeof skeleton !== 'object') return false;

  let drawn = false;
  if (useRaw) {
    drawn = drawSkeletonSpace(skeleton.raw, 'raw') || drawn;
  }
  if (useNorm) {
    drawn = drawSkeletonSpace(skeleton.norm, 'norm') || drawn;
  }
  return drawn;
}

function captureSegmentEvents(data) {
  const nowMs = Date.now();
  const segments = Array.isArray(data?.segments?.sign) ? data.segments.sign : [];
  for (const seg of segments) {
    const start = Number(seg.start);
    const end = Number(seg.end);
    const score = Number(seg.score || 0);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    const key = `${start}:${end}`;
    if (segmentEventSeen.has(key)) continue;
    segmentEventSeen.add(key);
    const len = Math.max(0, end - start + 1);
    const event = {
      key,
      start,
      end,
      len,
      score,
      ts: nowMs,
      committed: false,
    };
    segmentHistory.push(event);
    lastSegmentInfoText = `start=${start} end=${end} len=${len} score=${score.toFixed(3)}`;
  }

  const status = String(data?.status || '').toUpperCase();
  const committed = status === 'COMMIT' || status === 'COMMITTED' || Boolean(data?.text_state?.committed);
  if (committed) {
    segmentCommitFlashUntilMs = nowMs + 420;
    if (segmentHistory.length > 0) {
      segmentHistory[segmentHistory.length - 1].committed = true;
    }
  }

  segmentHistory = segmentHistory.filter((item) => (nowMs - item.ts) <= (TIMELINE_WINDOW_MS + 2000));
}

function renderSegmentsTimeline(nowMs = Date.now()) {
  if (!segmentsTimelineEl) return;
  segmentsTimelineEl.innerHTML = '';

  const events = segmentHistory.filter((item) => (nowMs - item.ts) <= TIMELINE_WINDOW_MS);
  for (const event of events) {
    const ageStart = nowMs - event.ts;
    const ageEnd = Math.max(0, ageStart - 250);
    const left = Math.max(0, Math.min(1, 1 - (ageStart / TIMELINE_WINDOW_MS)));
    const right = Math.max(0, Math.min(1, 1 - (ageEnd / TIMELINE_WINDOW_MS)));
    const width = Math.max(0.008, right - left);

    const bar = document.createElement('div');
    bar.className = `tl-seg${event.committed ? ' commit' : ''}`;
    bar.style.left = `${(left * 100).toFixed(2)}%`;
    bar.style.width = `${(width * 100).toFixed(2)}%`;
    bar.title = `start=${event.start} end=${event.end} len=${event.len} score=${event.score.toFixed(3)}`;
    segmentsTimelineEl.appendChild(bar);

    if (event.committed) {
      const marker = document.createElement('div');
      marker.className = 'tl-commit-marker';
      marker.style.left = `${(right * 100).toFixed(2)}%`;
      segmentsTimelineEl.appendChild(marker);
    }
  }

  const nowLine = document.createElement('div');
  nowLine.className = 'tl-now';
  segmentsTimelineEl.appendChild(nowLine);
}

function updateDebugPanel(data) {
  const perf = data?.perf || {};
  const bio = data?.bio || data?.debug?.bio || {};
  const mode = String(data?.mode || recognitionMode || 'letters');

  if (dbgModeEl) dbgModeEl.textContent = `${mode} (${effectiveInferenceLocation})`;
  if (dbgLatencyEl) {
    const latency = Number(perf.latency_ms ?? data?.debug?.latency_ms);
    dbgLatencyEl.textContent = Number.isFinite(latency) ? `${latency.toFixed(1)} ms` : 'n/a';
  }
  if (dbgFpsCamEl) dbgFpsCamEl.textContent = Number.isFinite(cameraFps) ? cameraFps.toFixed(1) : '0.0';
  if (dbgFpsInEl) {
    const v = Number(perf.fps_in);
    dbgFpsInEl.textContent = Number.isFinite(v) ? v.toFixed(1) : '0.0';
  }
  if (dbgFpsPoseEl) {
    const v = Number(perf.fps_pose);
    dbgFpsPoseEl.textContent = Number.isFinite(v) ? v.toFixed(1) : '0.0';
  }
  if (dbgFpsTotalEl) {
    const v = Number(perf.fps_total);
    dbgFpsTotalEl.textContent = Number.isFinite(v) ? v.toFixed(1) : '0.0';
  }
  if (dbgSegEnabledEl) dbgSegEnabledEl.textContent = String(Boolean(bio.enabled));
  if (dbgWinStepEl) {
    const win = Number(bio.window);
    const step = Number(bio.step);
    dbgWinStepEl.textContent = (Number.isFinite(win) && Number.isFinite(step)) ? `${win} / ${step}` : 'n/a';
  }
  if (dbgThEl) {
    const thB = Number(bio.th_B ?? bio.th_B_sign);
    const thO = Number(bio.th_O ?? bio.th_O_sign);
    dbgThEl.textContent = (Number.isFinite(thB) && Number.isFinite(thO)) ? `${thB.toFixed(2)} / ${thO.toFixed(2)}` : 'n/a';
  }
  if (dbgMinMergeEl) {
    const minLen = Number(bio.min_len);
    const mergeGap = Number(bio.merge_gap);
    dbgMinMergeEl.textContent = (Number.isFinite(minLen) && Number.isFinite(mergeGap)) ? `${minLen} / ${mergeGap}` : 'n/a';
  }
  if (dbgLastSegEl) dbgLastSegEl.textContent = lastSegmentInfoText || 'none';
}

function drawSegmentsOverlay(data) {
  const segments = data?.segments;
  const debugBio = data?.bio || data?.debug?.bio;
  if (!segments && !debugBio) {
    return;
  }

  const signSegments = Array.isArray(segments?.sign) ? segments.sign : [];
  const phraseSegments = Array.isArray(segments?.phrase) ? segments.phrase : [];
  const bufferStart = Number(debugBio?.buffer_start);
  const bufferEnd = Number(debugBio?.buffer_end);
  const hasRange = Number.isFinite(bufferStart) && Number.isFinite(bufferEnd) && bufferEnd >= bufferStart;

  if (hasRange && (signSegments.length > 0 || phraseSegments.length > 0)) {
    const span = Math.max(1, (bufferEnd - bufferStart + 1));
    const barHeight = 16;
    const baseY = canvasEl.height - barHeight - 8;

    ctx.save();
    ctx.fillStyle = 'rgba(14, 18, 25, 0.45)';
    ctx.fillRect(4, baseY - 6, canvasEl.width - 8, barHeight + 10);

    for (const seg of phraseSegments) {
      const x1 = ((Number(seg.start) - bufferStart) / span) * canvasEl.width;
      const x2 = ((Number(seg.end) - bufferStart + 1) / span) * canvasEl.width;
      ctx.fillStyle = 'rgba(255, 194, 96, 0.35)';
      ctx.fillRect(Math.max(0, x1), baseY + 8, Math.max(2, x2 - x1), 4);
    }

    for (const seg of signSegments) {
      const x1 = ((Number(seg.start) - bufferStart) / span) * canvasEl.width;
      const x2 = ((Number(seg.end) - bufferStart + 1) / span) * canvasEl.width;
      ctx.fillStyle = 'rgba(118, 255, 176, 0.7)';
      ctx.fillRect(Math.max(0, x1), baseY, Math.max(2, x2 - x1), 6);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x1, baseY - 2);
      ctx.lineTo(x1, baseY + 8);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, baseY - 2);
      ctx.lineTo(x2, baseY + 8);
      ctx.stroke();
    }
    ctx.restore();
  }

  if (Boolean(debugBio?.active_sign)) {
    const progress = Number(debugBio?.active_sign_progress || 0);
    const text = `SEGMENT: ON ${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
    ctx.save();
    ctx.fillStyle = 'rgba(255, 126, 154, 0.92)';
    ctx.font = '600 14px "SF Mono", monospace';
    ctx.fillText(text, 14, 26);
    ctx.strokeStyle = 'rgba(255, 126, 154, 0.85)';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(8, 8, canvasEl.width - 16, canvasEl.height - 16);
    ctx.restore();
  }

  if (Date.now() < segmentCommitFlashUntilMs) {
    ctx.save();
    ctx.fillStyle = 'rgba(255, 214, 138, 0.18)';
    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.fillStyle = 'rgba(255, 214, 138, 0.95)';
    ctx.font = '700 18px "Avenir Next", sans-serif';
    ctx.fillText('SEGMENT COMMIT', 14, canvasEl.height - 18);
    ctx.restore();
  }
}

function setStatusClass(status) {
  statusEl.className = '';
  const normalized = String(status || '').toLowerCase();
  if (normalized === 'hold') {
    statusEl.classList.add('status-candidate');
    return;
  }
  if (normalized === 'commit') {
    statusEl.classList.add('status-committed');
    return;
  }
  if (normalized === 'unknown') {
    statusEl.classList.add('status-none');
    return;
  }
  statusEl.classList.add(`status-${normalized}`);
}

function pickVisibleLetter(data) {
  const rawLetter = (typeof data.letter === 'string' && data.letter.length > 0) ? data.letter : 'NONE';
  if (!data.hand_present) {
    stableVisibleLetter = 'NONE';
    return 'NONE';
  }
  if (rawLetter !== 'NONE') {
    stableVisibleLetter = rawLetter;
    return rawLetter;
  }
  return stableVisibleLetter;
}

async function fetchServerConfig() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.config) {
      sendFps = Number(data.config.frontend_fps || 12);
      jpegQuality = Number(data.config.jpeg_quality || 0.75);
      recognitionMode = String(data.config.recognition_mode || 'letters');
      tokenLabelEl.textContent = recognitionMode === 'words' ? 'Слово' : (recognitionMode === 'pose_words' ? 'Поза' : 'Буква');
    }
  } catch (err) {
    console.warn('health config unavailable:', err);
  }
}

async function startCamera() {
  if (stream) return;
  shouldRun = true;
  await fetchServerConfig();
  await initializeInferenceBackends();

  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
  });

  videoEl.srcObject = stream;
  await videoEl.play();
  applyMirrorStyles();

  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  captureCanvas.width = videoEl.videoWidth;
  captureCanvas.height = videoEl.videoHeight;

  connectWs();
  startSender();
  renderLoop();

  startBtn.disabled = true;
  stopBtn.disabled = false;
  renderSentencesPanel();
}

function stopCamera() {
  shouldRun = false;

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }

  if (renderReq) {
    cancelAnimationFrame(renderReq);
    renderReq = null;
  }

  if (ws) {
    ws.close();
    ws = null;
  }

  if (browserInferenceBackend && typeof browserInferenceBackend.dispose === 'function') {
    browserInferenceBackend.dispose().catch(() => undefined);
  }
  browserInferenceBackend = null;
  effectiveInferenceLocation = 'backend';

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  latest = null;
  lastStateAtMs = 0;
  awaitingServer = false;
  lastSendAtMs = 0;
  stableVisibleLetter = 'NONE';
  recognitionMode = 'letters';
  committedTokens = [];
  lastCommitToken = '';
  lastCommitAtMs = 0;
  cameraFps = 0;
  cameraFpsCounter = 0;
  cameraFpsWindowStartMs = 0;
  lastServerPerf = null;
  segmentCommitFlashUntilMs = 0;
  segmentHistory = [];
  segmentEventSeen = new Set();
  lastSegmentInfoText = 'none';
  statusEl.textContent = 'NONE';
  letterEl.textContent = 'NONE';
  textValueEl.textContent = 'NONE';
  setInferenceStatus(
    inferenceLocation === 'browser'
      ? 'Browser mode experimental выбран. Запустите камеру для инициализации.'
      : 'Используется backend WebSocket.',
    inferenceLocation === 'browser' ? 'warn' : 'normal',
  );
  renderSentencesPanel();
  renderSegmentsTimeline(Date.now());
  updateDebugPanel({ mode: 'letters', perf: {}, bio: { enabled: false } });
}

function connectWs() {
  if (!shouldRun) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  ws = new WebSocket(wsUrl());
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'ack') return;
      latest = data;
      lastStateAtMs = Date.now();
      awaitingServer = false;
      renderState(data);
    } catch (err) {
      console.warn('bad ws message', err);
    }
  };

  ws.onerror = () => {
    if (ws) ws.close();
  };

  ws.onclose = () => {
    awaitingServer = false;
    ws = null;
    if (shouldRun) {
      reconnectTimer = setTimeout(() => {
        connectWs();
      }, 800);
    }
  };
}

function startSender() {
  const interval = Math.max(50, Math.round(1000 / sendFps));
  sendTimer = setInterval(() => {
    if (!stream || !ws || ws.readyState !== WebSocket.OPEN || sendBusy) return;
    if (!videoEl.videoWidth || !videoEl.videoHeight) return;
    if (awaitingServer) return;

    sendBusy = true;
    drawVideoFrame(captureCtx, videoEl, captureCanvas.width, captureCanvas.height, MIRROR_STREAM);
    captureCanvas.toBlob((blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) {
        sendBusy = false;
        return;
      }
      blob.arrayBuffer().then((arr) => {
        ws.send(arr);
        awaitingServer = true;
        lastSendAtMs = Date.now();
        sendBusy = false;
      }).catch(() => {
        sendBusy = false;
      });
    }, 'image/jpeg', jpegQuality);
  }, interval);
}

function renderLoop() {
  if (!stream) return;

  const nowMs = Date.now();
  cameraFpsCounter += 1;
  if (!cameraFpsWindowStartMs) {
    cameraFpsWindowStartMs = nowMs;
  } else {
    const dt = nowMs - cameraFpsWindowStartMs;
    if (dt >= 1000) {
      cameraFps = (cameraFpsCounter * 1000) / dt;
      cameraFpsCounter = 0;
      cameraFpsWindowStartMs = nowMs;
      if (lastServerPerf) {
        updateDebugPanel(lastServerPerf);
      }
    }
  }

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const stateFresh = Date.now() - lastStateAtMs < 800;
  const wantsSkeleton = Boolean(rawSkeletonToggleEl?.checked || normSkeletonToggleEl?.checked);
  const skeletonDrawn = stateFresh ? drawSkeletonOverlay(latest) : false;

  if (!wantsSkeleton && stateFresh && latest && latest.hand_present && latest.bbox_norm) {
    const [x1, y1, x2, y2] = latest.bbox_norm;
    const px1 = x1 * canvasEl.width;
    const py1 = y1 * canvasEl.height;
    const px2 = x2 * canvasEl.width;
    const py2 = y2 * canvasEl.height;
    if (Number.isFinite(px1) && Number.isFinite(py1) && Number.isFinite(px2) && Number.isFinite(py2)) {
      ctx.strokeStyle = '#37f59a';
      ctx.lineWidth = 3;
      ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
    }
  } else if (wantsSkeleton && !skeletonDrawn) {
    // В режиме скелета без данных ничего не рисуем.
  }

  if (stateFresh && latest) {
    drawSegmentsOverlay(latest);
  }

  renderSegmentsTimeline(nowMs);

  renderReq = requestAnimationFrame(renderLoop);
}

function renderState(data) {
  const mode = String(data.mode || recognitionMode || 'letters');
  recognitionMode = mode;
  lastServerPerf = data;
  appendCommitToken(data, mode);
  captureSegmentEvents(data);
  tokenLabelEl.textContent = mode === 'words' ? 'Слово' : (mode === 'pose_words' ? 'Поза' : 'Буква');

  setStatusClass(data.status);
  statusEl.textContent = data.status;
  let visibleLetter = pickVisibleLetter(data);
  if (mode === 'words') {
    visibleLetter = String(data.word || data.top1?.label || data.letter || 'NONE');
  }
  letterEl.textContent = visibleLetter;
  scoreEl.textContent = Number(data.score || 0).toFixed(3);
  confidenceEl.textContent = Number(data.confidence || 0).toFixed(3);

  const hold = data.hold || {};
  const holdUnit = String(hold.unit || 'ms');
  const holdUnitLabel = holdUnit === 'frames' ? 'кадр.' : 'мс';
  holdEl.textContent = `${hold.elapsed_ms || 0} / ${hold.target_ms || 0} ${holdUnitLabel}`;
  remainEl.textContent = `${hold.remaining_ms || 0} ${holdUnitLabel}`;
  progressBar.style.width = `${Math.max(0, Math.min(100, (hold.progress || 0) * 100))}%`;

  textValueEl.textContent = mode === 'words'
    ? String(data.text_state?.value || visibleLetter || 'NONE')
    : visibleLetter;

  const topk = Array.isArray(data.topk) ? data.topk : [];
  topkEl.innerHTML = '';
  for (const item of topk) {
    const li = document.createElement('li');
    const label = item.letter || item.label || 'NONE';
    li.textContent = `${label}: ${Number(item.score || 0).toFixed(3)}`;
    topkEl.appendChild(li);
  }

  const dbg = data.debug || {};
  const perf = data.perf || {};
  const latencyRaw = perf.latency_ms ?? dbg.latency_ms;
  const latency = latencyRaw === null || latencyRaw === undefined ? 'n/a' : Number(latencyRaw).toFixed(1);
  const fpMin = dbg.fp_per_minute === null || dbg.fp_per_minute === undefined ? 'n/a' : Number(dbg.fp_per_minute).toFixed(2);
  const avgLat = dbg.avg_infer_latency_ms === null || dbg.avg_infer_latency_ms === undefined ? 'n/a' : Number(dbg.avg_infer_latency_ms).toFixed(1);
  const p95Lat = dbg.p95_infer_latency_ms === null || dbg.p95_infer_latency_ms === undefined ? 'n/a' : Number(dbg.p95_infer_latency_ms).toFixed(1);
  const bio = data.bio || dbg.bio || {};
  const seg = data.segments || {};
  const signSegCount = Array.isArray(seg.sign) ? seg.sign.length : 0;
  const phraseSegCount = Array.isArray(seg.phrase) ? seg.phrase.length : 0;
  const bioActive = Boolean(bio.active_sign);
  debugEl.textContent = `sim1=${Number(dbg.sim1 || 0).toFixed(3)} | sim2=${Number(dbg.sim2 || 0).toFixed(3)} | margin=${Number(dbg.margin || 0).toFixed(3)} | uncertain=${Boolean(dbg.uncertain)} | cooldown=${dbg.cooldown_left_ms || 0}${holdUnit === 'frames' ? 'fr' : 'мс'} | latency=${latency}ms | fp/min=${fpMin} | avg=${avgLat}ms p95=${p95Lat}ms | seg(sign=${signSegCount}, phrase=${phraseSegCount}, active=${bioActive})`;
  updateDebugPanel(data);
  renderSegmentsTimeline(Date.now());

  const vlm = data.vlm || {};
  vlmUsedEl.textContent = String(Boolean(vlm.used));
  vlmLetterEl.textContent = vlm.letter || 'NONE';
  vlmConfEl.textContent = Number(vlm.confidence || 0).toFixed(3);
  vlmReasonEl.textContent = vlm.reason || '';
}

startBtn.addEventListener('click', async () => {
  try {
    await startCamera();
  } catch (err) {
    alert(`Ошибка запуска камеры: ${err.message || err}`);
  }
});

stopBtn.addEventListener('click', () => {
  stopCamera();
});

window.addEventListener('beforeunload', () => {
  stopCamera();
});

renderSentencesPanel();
renderSegmentsTimeline(Date.now());
updateDebugPanel({ mode: recognitionMode, perf: {}, bio: { enabled: false } });
setupInferenceControls();
setInferenceStatus(
  inferenceLocation === 'browser'
    ? 'Browser mode experimental выбран. Запустите камеру для инициализации.'
    : 'Используется backend WebSocket.',
  inferenceLocation === 'browser' ? 'warn' : 'normal',
);
