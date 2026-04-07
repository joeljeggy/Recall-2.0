// ── Icons Helper (updated with viewBox & Theme icons) ──
const ICONS = {
  zap: '<svg viewBox="0 0 24 24"><use href="#icon-zap"></use></svg>',
  cpu: '<svg viewBox="0 0 24 24"><use href="#icon-cpu"></use></svg>',
  layers: '<svg viewBox="0 0 24 24"><use href="#icon-layers"></use></svg>',
  activity: '<svg viewBox="0 0 24 24"><use href="#icon-activity"></use></svg>',
  chevron: '<svg viewBox="0 0 24 24"><use href="#icon-chevron"></use></svg>',
  sparkles: '<svg viewBox="0 0 24 24"><use href="#icon-sparkles"></use></svg>',
  alert: '<svg viewBox="0 0 24 24"><use href="#icon-alert"></use></svg>',
  play: '<svg viewBox="0 0 24 24"><use href="#icon-play"></use></svg>',
  hash: '<svg viewBox="0 0 24 24"><use href="#icon-hash"></use></svg>',
  bot: '<svg viewBox="0 0 24 24"><use href="#icon-bot"></use></svg>'
};

// ── Theme Toggle Logic ──
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  
  const iconId = newTheme === 'light' ? 'moon' : 'sun';
  document.getElementById('theme-icon').innerHTML = `<use href="#icon-${iconId}"></use>`;
  
  // If the memory visualizer page is active, redraw the canvas with the new theme colors
  if (document.getElementById('page-memory').classList.contains('active')) {
    drawDecayCurve(currentLambdaData); 
  }
}

// Set initial toggle icon on load
window.addEventListener('DOMContentLoaded', () => {
  const initialTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const iconId = initialTheme === 'light' ? 'moon' : 'sun';
  const themeIconEl = document.getElementById('theme-icon');
  if (themeIconEl) {
    themeIconEl.innerHTML = `<use href="#icon-${iconId}"></use>`;
  }
});

// ── State ──
let activeFilter = 'all';
let runHistory = [];
let currentLambdaData = null; // Stores data to redraw canvas on theme switch/resize without API hits

// ── Nav ──
document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', () => {
    const p = el.dataset.page;
    document.querySelectorAll('.nav-item').forEach(e => e.classList.remove('active'));
    document.querySelectorAll('.page').forEach(e => e.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('page-' + p).classList.add('active');
    if (p === 'memory') loadMemory();
    if (p === 'history') loadHistory(activeFilter);
    if (p === 'runs') loadRuns();
  });
});

document.getElementById('query-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runPipeline(); }
});

// ── API (Original logic restored) ──
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Utils ──
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  let icon = ICONS.zap;
  if (type === 'err') icon = ICONS.alert;
  if (type === 'ok') icon = '<svg viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>';
  el.innerHTML = `${icon} <span>${esc(msg)}</span>`;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function esc(s) { return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function ago(ms) {
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

// ── Status ──
async function updateStatus() {
  try {
    const d = await api('GET', '/api/memory/stats');
    document.getElementById('s-total').textContent = d.total;
    document.getElementById('seg-badge').textContent = d.total;
    const runs = await api('GET', '/api/runs');
    document.getElementById('s-runs').textContent = runs.length;
    document.getElementById('run-badge').textContent = runs.length;
  } catch { }
}

// ── Pipeline ──
const AGENT_COLORS = {
  IntakeAgent: { dot: 'var(--intake-c)' },
  KnowledgeAgent: { dot: 'var(--knowledge-c)' },
  ResponseAgent: { dot: 'var(--response-c)' },
};

async function runPipeline() {
  const q = document.getElementById('query-input').value.trim();
  if (!q) return;
  if (q.length > 2000) { toast('Query too long (max 2000 chars)', 'err'); return; }
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Executing...';

  const tb = document.getElementById('trace-body');
  tb.innerHTML = '';
  document.getElementById('trace-elapsed').textContent = '';

  const resultBox = document.createElement('div');
  resultBox.className = 'result-box';
  resultBox.style.marginTop = '0';
  resultBox.style.marginBottom = '16px';
  resultBox.innerHTML = `
    <style>
      @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }
      .skel-line {
        height: 12px;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--surface) 25%, var(--border2) 50%, var(--surface) 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite linear;
        margin-bottom: 12px;
      }
    </style>
    <div class="result-label" style="display:flex; align-items:center; gap:8px; margin-bottom: 16px;">
      <div class="skel-line" style="width: 140px; height: 14px; margin: 0;"></div>
    </div>
    <div class="result-text">
      <div class="skel-line" style="width: 95%;"></div>
      <div class="skel-line" style="width: 90%;"></div>
      <div class="skel-line" style="width: 75%;"></div>
    </div>
  `;
  tb.appendChild(resultBox);

  const agentsContainer = document.createElement('div');
  agentsContainer.style.display = 'flex';
  agentsContainer.style.flexDirection = 'column';
  agentsContainer.style.gap = '16px';
  tb.appendChild(agentsContainer);

  const agentNames = ['IntakeAgent', 'KnowledgeAgent', 'ResponseAgent'];
  const stepEls = agentNames.map((name, i) => {
    const el = makePendingStep(name, i);
    agentsContainer.appendChild(el);
    return el;
  });

  const startTime = performance.now();
  const agentTraces = [];
  let finalRun = null;

  try {
    const res = await fetch('/api/pipeline/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Original SSE Streaming logic restored
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop(); 

      let eventType = 'message';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          if (eventType === 'agent_start') {
            const idx = data.index;
            if (stepEls[idx]) {
              stepEls[idx].classList.add('step-active');
            }
          } else if (eventType === 'agent_complete') {
            const idx = data.index;
            const tr = data.trace;
            agentTraces.push(tr);
            if (stepEls[idx]) {
              const completed = makeAgentStep(tr, idx);
              stepEls[idx].replaceWith(completed);
              stepEls[idx] = completed;
            }
          } else if (eventType === 'pipeline_complete') {
            finalRun = data.run;
          } else if (eventType === 'error') {
            throw new Error(data.message || 'Pipeline error');
          }
          eventType = 'message'; 
        }
      }
    }

    if (finalRun) {
      if (finalRun.response) {
        resultBox.innerHTML = `<div class="result-label">${ICONS.sparkles} Synthesis Complete</div><div class="result-text">${esc(finalRun.response)}</div>`;
      } else {
        resultBox.remove();
      }
      document.getElementById('trace-elapsed').textContent = `${finalRun.elapsed_s}s total`;
      addRecent(finalRun);
      runHistory.unshift(finalRun);
    } else {
      resultBox.remove();
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      document.getElementById('trace-elapsed').textContent = `${elapsed}s total`;
    }

    document.getElementById('query-input').value = '';
    updateStatus();
    toast('Pipeline execution completed', 'ok');
  } catch (e) {
    tb.innerHTML = `<div class="empty-state" style="color:var(--red)">${ICONS.alert} <div>Execution Failed: ${esc(e.message)}</div></div>`;
    toast('Error: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `${ICONS.play} Execute Pipeline`;
  }
}

function var_(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function makePendingStep(name, i) {
  const colors = [var_('--intake-c'), var_('--knowledge-c'), var_('--response-c')];
  const div = document.createElement('div');
  div.className = 'agent-step';
  div.style.borderColor = `rgba(${colors[i]}, 0.2)`;
  div.innerHTML = `
    <div class="step-header" style="opacity:0.6">
      <div class="step-dot" style="background:${colors[i]}"></div>
      <div class="step-name">${name}</div>
      <div class="step-meta"><span class="spinner" style="border-top-color:${colors[i]}"></span></div>
    </div>`;
  return div;
}

function makeAgentStep(tr, i) {
  const dotColors = ['var(--intake-c)', 'var(--knowledge-c)', 'var(--response-c)'];
  const dot = dotColors[i] || 'var(--muted)';
  const stepId = 'step-' + i + Math.random().toString(36).substr(2, 5);

  let memChips = '';
  if (tr.memories_used && typeof tr.memories_used === 'object' && !Array.isArray(tr.memories_used)) {
    const chips = Object.entries(tr.memories_used).map(([k, v]) => {
      return `<span class="mem-count-chip">${k}: ${v}</span>`;
    }).join('');
    if (chips) memChips = `<div class="step-section"><div class="step-section-label">${ICONS.layers} Semantic Recalls</div><div class="mem-count-row">${chips}</div></div>`;
  }

  let usedPills = '';
  if (tr.used_memory_ids && tr.used_memory_ids.length) {
    const pills = tr.used_memory_ids.map((id, j) => {
      const txt = tr.used_memory_texts && tr.used_memory_texts[j] ? esc(tr.used_memory_texts[j]) : id;
      return `<div class="mem-pill-row"><span class="mem-pill-id">${id}</span><span class="mem-pill-text">${txt}</span></div>`;
    }).join('');
    usedPills = `<div class="step-section"><div class="step-section-label">${ICONS.cpu} Segments Utilized</div><div>${pills}</div></div>`;
  }

  let outputHtml = '';
  if (tr.output) {
    const out = typeof tr.output === 'object' ? JSON.stringify(tr.output, null, 2) : String(tr.output);
    outputHtml = `<div class="step-section"><div class="step-section-label">${ICONS.activity} Output Data</div><div class="step-output"><pre>${esc(out)}</pre></div></div>`;
  }

  const hasBody = memChips || usedPills || outputHtml;

  const div = document.createElement('div');
  div.className = 'agent-step';

  let hoverTimeout; 

  div.onmouseenter = () => {
    clearTimeout(hoverTimeout); 
    const body = document.getElementById(stepId);
    const chev = document.getElementById('chev-' + stepId);
    if (body && body.dataset.pinned !== 'true') {
      body.classList.add('open');
      if (chev) chev.classList.add('open');
    }
  };
  
  div.onmouseleave = () => {
    hoverTimeout = setTimeout(() => {
      const body = document.getElementById(stepId);
      const chev = document.getElementById('chev-' + stepId);
      if (body && body.dataset.pinned !== 'true') {
        body.classList.remove('open');
        if (chev) chev.classList.remove('open');
      }
    }, 250); 
  };

  div.innerHTML = `
    <div class="step-header" onclick="togglePin('${stepId}')">
      <div class="step-dot" style="background:${dot}"></div>
      <div class="step-name">${esc(tr.agent)}</div>
      <div class="step-meta">
        <span>${tr.elapsed_s}s</span>
        <span>+${tr.mem_delta} mem</span>
        ${hasBody ? `<span class="step-chevron" id="chev-${stepId}">${ICONS.chevron}</span>` : ''}
      </div>
    </div>
    ${hasBody ? `
      <div class="step-body" id="${stepId}" data-pinned="false">
        <div class="step-body-inner">
          <div class="step-body-content">
            ${memChips}${usedPills}${outputHtml}
          </div>
        </div>
      </div>` : ''}`;
  return div;
}

window.togglePin = function(id) {
  const body = document.getElementById(id);
  const chev = document.getElementById('chev-' + id);
  if (!body) return;

  if (body.dataset.pinned === 'true') {
    body.dataset.pinned = 'false';
    body.classList.remove('open');
    if (chev) chev.classList.remove('open');
  } else {
    body.dataset.pinned = 'true';
    body.classList.add('open');
    if (chev) chev.classList.add('open');
  }
}

function addRecent(run) {
  const list = document.getElementById('recent-list');
  const placeholder = list.querySelector('[data-placeholder]');
  if (placeholder) placeholder.remove();

  const intent = run.agent_traces?.[0]?.output?.intent || 'general';
  const div = document.createElement('div');
  div.className = 'recent-item';
  div.innerHTML = `
    <span class="recent-intent intent-${intent}">${intent}</span>
    <div style="flex:1;min-width:0">
      <div class="recent-text">${esc(run.task)}</div>
      <div style="font-size:11px;color:var(--muted);font-family:var(--mono)">${ago(run.timestamp * 1000)} · ${run.elapsed_s}s</div>
    </div>
    <span style="color:var(--muted); opacity: 0.5;">${ICONS.play}</span>`;
  div.onclick = () => replayRun(run);
  list.prepend(div);
}

function replayRun(run) {
  document.getElementById('query-input').value = run.task;
  const tb = document.getElementById('trace-body');
  tb.innerHTML = '';
  document.getElementById('trace-elapsed').textContent = `${run.elapsed_s}s total`;

  if (run.response) {
    const rb = document.createElement('div');
    rb.className = 'result-box';
    rb.style.marginTop = '0';
    rb.style.marginBottom = '16px';
    rb.innerHTML = `<div class="result-label">${ICONS.sparkles} Synthesis Complete</div><div class="result-text">${esc(run.response)}</div>`;
    tb.appendChild(rb);
  }

  const agentsContainer = document.createElement('div');
  agentsContainer.style.display = 'flex';
  agentsContainer.style.flexDirection = 'column';
  agentsContainer.style.gap = '16px';
  tb.appendChild(agentsContainer);

  run.agent_traces.forEach((tr, i) => agentsContainer.appendChild(makeAgentStep(tr, i)));
}

// ── Seed / Memory ──
async function seedKnowledge() {
  try {
    const d = await api('POST', '/api/memory/seed');
    toast(`Seeded ${d.seeded} knowledge vectors`, 'ok');
    updateStatus();
  } catch (e) { toast('Error: ' + e.message, 'err'); }
}

async function pruneMemory() {
  try {
    const d = await api('POST', '/api/memory/prune');
    toast(`Purged ${d.pruned} degraded segments. ${d.remaining} intact.`, 'ok');
    loadMemory(); updateStatus();
  } catch (e) { toast('Error: ' + e.message, 'err'); }
}

async function loadMemory() {
  try {
    const d = await api('GET', '/api/memory/stats');
    currentLambdaData = d.lambda_by_type; 
    
    document.getElementById('v-total').textContent = d.total;
    document.getElementById('v-knowledge').textContent = d.by_type.knowledge || 0;
    document.getElementById('v-dialog').textContent = d.by_type.dialog || 0;
    document.getElementById('v-task').textContent = d.by_type.task || 0;
    document.getElementById('v-recalls').textContent = `${d.total_recalls || 0} recalls across pipeline`;
    
    document.getElementById('r-high').textContent = d.retention_buckets['high (>0.8)'] || 0;
    document.getElementById('r-mid').textContent = d.retention_buckets['mid (0.4–0.8)'] || 0;
    document.getElementById('r-low').textContent = d.retention_buckets['low (<0.4)'] || 0;

    const agents = Object.entries(d.by_agent || {}).sort((a, b) => b[1] - a[1]);
    const max = agents[0]?.[1] || 1;
    const palette = ['var(--accent)', 'var(--green)', 'var(--amber)', 'var(--muted)'];
    
    document.getElementById('agent-bars').innerHTML = agents.length
      ? agents.map(([a, n], i) => `
        <div class="bar-row">
          <div class="bar-label" title="${a}">${a.replace('Agent', '')}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${(n / max * 100)}%;background:${palette[i % palette.length]}"></div></div>
          <div class="bar-val">${n}</div>
        </div>`).join('')
      : '<div style="color:var(--muted);font-size:13px">Insufficient data</div>';

    drawDecayCurve(currentLambdaData);
  } catch (e) { toast('Failed to fetch memory state', 'err'); }
}

// ── Enhanced Canvas Drawing for Light/Dark Mode ──
function drawDecayCurve(lambdaByType) {
  const canvas = document.getElementById('decay-canvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return; // Prevent 0 size errors when hidden

  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  // Determine Theme variables for Canvas Drawing
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const cGrid = isLight ? '0,0,0' : '255,255,255'; 
  const cAxisL = isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.22)';
  const cLabel = isLight ? 'rgba(0,0,0,0.65)' : 'rgba(255,255,255,0.65)';
  const cFootnote = isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.18)';

  const padL = 36, padR = 16, padT = 28, padB = 22;
  const W2 = W - padL - padR, H2 = H - padT - padB;
  const maxHrs = 48;

const TYPE_META = [
    { key: 'knowledge', color: isLight ? '#4f46e5' : '#3b82f6', label: 'knowledge', dflt: 2.0 },
    { key: 'task',      color: isLight ? '#059669' : '#22c55e', label: 'task',      dflt: 1.5 },
    { key: 'dialog',    color: isLight ? '#d97706' : '#f97316', label: 'dialog',    dflt: 1.0 },
  ];

  const curves = TYPE_META.map(({ key, color, label, dflt }) => {
    const lambdas = lambdaByType && lambdaByType[key] && lambdaByType[key].length ? lambdaByType[key] : null;
    const top = lambdas ? lambdas[0] : dflt;
    const bot = lambdas && lambdas.length > 1 ? lambdas[lambdas.length - 1] : null;
    return { top, bot, color, label, hasReal: !!lambdas };
  });

  // ── Background subtle gradient ──────────────────────────────
  const bgGrad = ctx.createLinearGradient(0, padT, 0, padT + H2);
  bgGrad.addColorStop(0, isLight ? 'rgba(99,102,241,0.05)' : 'rgba(99,102,241,0.03)');
  bgGrad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(padL, padT, W2, H2);

  // ── Grid lines ──────────────────────────────────────────────
  ctx.save();
  [0.25, 0.5, 0.75, 1.0].forEach(frac => {
    const y = padT + H2 * (1 - frac);
    const grad = ctx.createLinearGradient(padL, y, padL + W2, y);
    grad.addColorStop(0,   `rgba(${cGrid},0.0)`);
    grad.addColorStop(0.1, frac === 1.0 ? `rgba(${cGrid},0.15)` : `rgba(${cGrid},0.06)`);
    grad.addColorStop(0.9, frac === 1.0 ? `rgba(${cGrid},0.15)` : `rgba(${cGrid},0.06)`);
    grad.addColorStop(1,   `rgba(${cGrid},0.0)`);
    ctx.strokeStyle = grad;
    ctx.lineWidth = frac === 1.0 ? 1 : 0.5;
    ctx.setLineDash(frac === 1.0 ? [] : [4, 6]);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W2, y); ctx.stroke();
  });
  ctx.setLineDash([]);
  [12, 24, 36].forEach(h => {
    const x = padL + (h / maxHrs) * W2;
    ctx.strokeStyle = `rgba(${cGrid},0.06)`;
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + H2); ctx.stroke();
  });
  ctx.restore();

  // ── Draw each curve ──────────────────────────────────────────
  const points = (lambda) => {
    const pts = [];
    for (let x = 0; x <= W2; x += 0.5) {
      const t = (x / W2) * maxHrs;
      pts.push([padL + x, padT + H2 * (1 - Math.exp(-t / lambda))]);
    }
    return pts;
  };

  curves.forEach(({ top, bot, color }) => {
    const pTop = points(top);
    const pBot = bot !== null && bot !== top ? points(bot) : null;

    // Filled area under top curve with gradient
    const areaGrad = ctx.createLinearGradient(0, padT, 0, padT + H2);
    areaGrad.addColorStop(0, color + '22');
    areaGrad.addColorStop(1, color + '02');
    ctx.beginPath();
    pTop.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.lineTo(padL + W2, padT + H2);
    ctx.lineTo(padL, padT + H2);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // Band fill between top and bot
    if (pBot) {
      ctx.beginPath();
      pTop.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      for (let i = pBot.length - 1; i >= 0; i--) ctx.lineTo(pBot[i][0], pBot[i][1]);
      ctx.closePath();
      ctx.fillStyle = color + '14';
      ctx.fill();
    }

    // Main line
    ctx.beginPath();
    pTop.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Dashed weaker line (no glow)
    if (pBot) {
      ctx.save();
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      pBot.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.strokeStyle = color + '55';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.restore();
    }

    // Dot at t=0
    ctx.beginPath();
    ctx.arc(padL, padT, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  });

  // ── Y axis labels ────────────────────────────────────────────
  ctx.fillStyle = cAxisL;
  ctx.font = '500 9px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  ctx.fillText('1.0', padL - 6, padT + 4);
  ctx.fillText('0.5', padL - 6, padT + H2 * 0.5 + 4);
  ctx.fillText('0',   padL - 6, padT + H2 + 4);

  // ── X axis labels ─────────────────────────────────────────────
  ctx.textAlign = 'center';
  ctx.fillStyle = cAxisL;
  ctx.font = '500 9px JetBrains Mono, monospace';
  [[0,'0h'],[12,'12h'],[24,'24h'],[36,'36h'],[48,'48h']].forEach(([h, lbl]) => {
    ctx.fillText(lbl, padL + (h / maxHrs) * W2, H - 4);
  });

  // ── Legend ────────────────────────────────────────────────────
  let lx = padL + 4;
  curves.forEach(({ color, label, top, hasReal }) => {
    // Pill background
    const tw = 84;
    ctx.fillStyle = color + '15';
    ctx.beginPath();
    ctx.roundRect(lx, 6, tw, 14, 4);
    ctx.fill();
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 0.5;
    ctx.stroke();
    // Colour swatch
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(lx + 5, 11, 6, 3, 1);
    ctx.fill();
    // Label text
    ctx.fillStyle = cLabel;
    ctx.font = '500 8.5px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label + ' \u03bb=' + top.toFixed(1) + (hasReal ? '' : '*'), lx + 14, 19);
    lx += tw + 6;
  });

  // Footnote
  if (curves.some(c => !c.hasReal)) {
    ctx.fillStyle = cFootnote;
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('* default \u03bb\u2080 \u2014 run pipeline to see real curves', padL, H - 6);
  }
}

// Handle resize to redraw canvas
window.addEventListener('resize', () => {
  if (document.getElementById('page-memory').classList.contains('active')) {
    drawDecayCurve(currentLambdaData);
  }
});

// ── History ──
async function loadHistory(filter = 'all') {
  try {
    const url = filter === 'all' ? '/api/memory/history?limit=100' : `/api/memory/history?type=${filter}&limit=100`;
    const segs = await api('GET', url);
    const list = document.getElementById('history-list');
    if (!segs.length) {
      list.innerHTML = `<div class="empty-state">${ICONS.layers}<div>No segments initialized</div></div>`;
      return;
    }
    list.innerHTML = segs.map(s => {
      const r = s.retention;
      const retColor = r > 0.8 ? 'var(--green)' : r > 0.4 ? 'var(--amber)' : 'var(--red)';
      return `
        <div class="hi-card">
          <div class="hi-top">
            <span class="type-badge ${s.memory_type}">${s.memory_type}</span>
            <div style="flex:1">
              <div class="hi-text">${esc(s.text)}</div>
              <div class="hi-meta">
                <span style="font-family:var(--mono); color:var(--text)">${esc(s.source_agent)}</span>
                <span>&bull;</span>
                <span>${ago(s.created_at * 1000)}</span>
                <span>&bull;</span>
                <span>${s.recall_count} Recalls</span>
                <span class="ret-badge" style="color:${retColor}">Ret: ${r.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>`;
    }).join('');
  } catch (e) { toast('Failed to load segments', 'err'); }
}

window.filterHistory = function(btn, filter) {
  activeFilter = filter;
  document.querySelectorAll('.fb').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadHistory(filter);
}

// ── Runs ──
async function loadRuns() {
  try {
    const runs = await api('GET', '/api/runs');
    const list = document.getElementById('runs-list');
    if (!runs.length) {
      list.innerHTML = `<div class="empty-state">${ICONS.activity}<div>No execution history found</div></div>`;
      return;
    }
    
    list.innerHTML = '';

    runs.forEach((run, ri) => {
      const intent = run.agent_traces?.[0]?.output?.intent || 'general';
      const runId = 'run-' + ri;
      
      const div = document.createElement('div');
      div.className = 'run-card';

      let hoverTimeout; 

      div.onmouseenter = () => {
        clearTimeout(hoverTimeout);
        const body = document.getElementById(runId);
        const chev = document.getElementById('chev-' + runId);
        if (body && body.dataset.pinned !== 'true') {
          body.classList.add('open');
          if (chev) chev.classList.add('open');
        }
      };
      
      div.onmouseleave = () => {
        hoverTimeout = setTimeout(() => {
          const body = document.getElementById(runId);
          const chev = document.getElementById('chev-' + runId);
          if (body && body.dataset.pinned !== 'true') {
            body.classList.remove('open');
            if (chev) chev.classList.remove('open');
          }
        }, 250);
      };

      div.innerHTML = `
        <div class="run-head" onclick="togglePin('${runId}')">
          <span class="run-id">${run.run_id.substring(0,8)}</span>
          <span class="run-query">${esc(run.task)}</span>
          <div class="run-meta">
            <span class="type-badge intent-${intent}">${intent}</span>
            <span style="font-family:var(--mono)">${run.elapsed_s}s</span>
            <span>${ago(run.timestamp * 1000)}</span>
            <span class="step-chevron" id="chev-${runId}">${ICONS.chevron}</span>
          </div>
        </div>
        <div class="run-detail" id="${runId}" data-pinned="false">
          <div class="run-detail-inner">
            <div class="run-detail-content">
              <div style="background:var(--bg); border:1px solid var(--border); border-radius: 6px; padding: 16px; font-family:var(--mono); font-size: 12px; color:var(--muted); overflow-x:auto; white-space: pre-wrap; word-break: break-word;">
                ${esc(JSON.stringify(run.agent_traces, null, 2))}
              </div>
            </div>
          </div>
        </div>
      `;
      list.appendChild(div);
    });
  } catch (e) { toast('Failed to load runs', 'err'); }
}

// ── Init ──
async function initRecent() {
  try {
    const runs = await api('GET', '/api/runs');
    [...runs].reverse().forEach(run => addRecent(run));
  } catch { }
}

updateStatus();
setInterval(updateStatus, 10000);
initRecent();