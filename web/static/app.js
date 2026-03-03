// ── Icons Helper (updated with viewBox) ──
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

// ── State ──
let activeFilter = 'all';
let runHistory = [];

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

// ── API ──
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

  const agentNames = ['IntakeAgent', 'KnowledgeAgent', 'ResponseAgent'];
  const stepEls = agentNames.map((name, i) => {
    const el = makePendingStep(name, i);
    tb.appendChild(el);
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

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line in buffer

      let eventType = 'message';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          if (eventType === 'agent_start') {
            // Agent is starting — show spinner (already shown from pending)
            const idx = data.index;
            if (stepEls[idx]) {
              stepEls[idx].classList.add('step-active');
            }
          } else if (eventType === 'agent_complete') {
            const idx = data.index;
            const tr = data.trace;
            agentTraces.push(tr);
            // Replace pending step with completed step
            if (stepEls[idx]) {
              const completed = makeAgentStep(tr, idx);
              stepEls[idx].replaceWith(completed);
              stepEls[idx] = completed;
              // Auto-expand the latest completed step briefly
              const stepId = completed.querySelector('.step-body')?.id;
              if (stepId) {
                const body = document.getElementById(stepId);
                const chev = document.getElementById('chev-' + stepId);
                if (body) body.classList.add('open');
                if (chev) chev.classList.add('open');
              }
            }
          } else if (eventType === 'pipeline_complete') {
            finalRun = data.run;
          } else if (eventType === 'error') {
            throw new Error(data.message || 'Pipeline error');
          }
          eventType = 'message'; // reset for next event
        }
      }
    }

    if (finalRun) {
      if (finalRun.response) {
        const rb = document.createElement('div');
        rb.className = 'result-box';
        rb.innerHTML = `<div class="result-label">${ICONS.sparkles} Synthesis Complete</div><div class="result-text">${esc(finalRun.response)}</div>`;
        tb.appendChild(rb);
      }
      document.getElementById('trace-elapsed').textContent = `${finalRun.elapsed_s}s total`;
      addRecent(finalRun);
      runHistory.unshift(finalRun);
    } else {
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
  div.innerHTML = `
    <div class="step-header" onclick="toggleStep('${stepId}')">
      <div class="step-dot" style="background:${dot}"></div>
      <div class="step-name">${esc(tr.agent)}</div>
      <div class="step-meta">
        <span>${tr.elapsed_s}s</span>
        <span>+${tr.mem_delta} mem</span>
        ${hasBody ? `<span class="step-chevron" id="chev-${stepId}">${ICONS.chevron}</span>` : ''}
      </div>
    </div>
    ${hasBody ? `<div class="step-body" id="${stepId}">${memChips}${usedPills}${outputHtml}</div>` : ''}`;
  return div;
}

window.toggleStep = function(id) {
  const body = document.getElementById(id);
  const chev = document.getElementById('chev-' + id);
  if (body) body.classList.toggle('open');
  if (chev) chev.classList.toggle('open');
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

  run.agent_traces.forEach((tr, i) => tb.appendChild(makeAgentStep(tr, i)));
  if (run.response) {
    const rb = document.createElement('div');
    rb.className = 'result-box';
    rb.innerHTML = `<div class="result-label">${ICONS.sparkles} Synthesis Complete</div><div class="result-text">${esc(run.response)}</div>`;
    tb.appendChild(rb);
  }
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
    document.getElementById('v-total').textContent = d.total;
    document.getElementById('v-knowledge').textContent = d.by_type.knowledge || 0;
    document.getElementById('v-dialog').textContent = d.by_type.dialog || 0;
    document.getElementById('v-task').textContent = d.by_type.task || 0;
    document.getElementById('v-recalls').textContent = `${d.total_recalls || 0} recalls across pipeline`;
    
    const eb = document.getElementById('embedder-badge');
    if (eb) {
      const isST = d.embedder && d.embedder.includes('Sentence');
      eb.innerHTML = isST ? `${ICONS.bot} Semantic (Sentence-Transformers)` : `${ICONS.hash} Fast Hash (Bag-of-Words)`;
      eb.style.color = isST ? 'var(--green)' : 'var(--text)';
    }

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

    drawDecayCurve();
  } catch (e) { toast('Failed to fetch memory state', 'err'); }
}

function drawDecayCurve() {
  const canvas = document.getElementById('decay-canvas');
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const lambdas = [
    { l: 0.5, color: '#6366f1' },
    { l: 2, color: '#10b981' },
    { l: 5, color: '#f59e0b' },
  ];

  const pad = 12;
  const W2 = W - pad*2, H2 = H - pad*2;

  lambdas.forEach(({ l, color }) => {
    ctx.beginPath();
    for (let x = 0; x <= W2; x++) {
      const R = Math.exp(-((x / W2) * 24) / l);
      const px = pad + x;
      const py = pad + H2 * (1 - R);
      x === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  });

  ctx.fillStyle = 'var(--muted)';
  ctx.font = '10px JetBrains Mono,monospace';
  ctx.fillText('0h', pad, H - 4);
  ctx.fillText('24h', W - 30, H - 4);
}

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
    list.innerHTML = runs.map((run, ri) => {
      const intent = run.agent_traces?.[0]?.output?.intent || 'general';
      const runId = 'run-' + ri;
      return `
        <div class="run-card">
          <div class="run-head" onclick="toggleStep('${runId}')">
            <span class="run-id">${run.run_id.substring(0,8)}</span>
            <span class="run-query">${esc(run.task)}</span>
            <div class="run-meta">
              <span class="type-badge intent-${intent}">${intent}</span>
              <span style="font-family:var(--mono)">${run.elapsed_s}s</span>
              <span>${ago(run.timestamp * 1000)}</span>
              <span class="step-chevron open" id="chev-${runId}">${ICONS.chevron}</span>
            </div>
          </div>
          <div class="run-detail open" id="${runId}">
            <div style="background:var(--bg); border:1px solid var(--border); border-radius: 6px; padding: 16px; font-family:var(--mono); font-size: 12px; color:var(--muted); overflow-x:auto; white-space: pre-wrap; word-break: break-word;">
              ${esc(JSON.stringify(run.agent_traces, null, 2))}
            </div>
          </div>
        </div>`;
    }).join('');
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
