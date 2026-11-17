// PayI shared runtime: API, live event stream, formatting, Jarvis voice, sound, orb.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (n, d = 2) => Number(n || 0).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  return r.json();
}

// Live events: server-sent events, plus a JSON poll every 1.5 s. Some proxies (VPN/corporate PAC,
// extensions) buffer event streams; the poll guarantees every screen updates without a refresh.
function stream(onEvent, since = 0) {
  let last = since, es;
  const deliver = e => { if (e.seq <= last) return; last = e.seq; try { onEvent(e); } catch (err) { console.error(err); } };
  const open = () => {
    try {
      es = new EventSource(`/events?since=${last}`);
      es.onmessage = m => deliver(JSON.parse(m.data));
      es.onerror = () => { es.close(); setTimeout(open, 3000); };
    } catch (_) {}
  };
  const poll = async () => {
    try { const r = await fetch(`/api/events?since=${last}`, {cache: "no-store"}).then(r => r.json()); (r.events || []).forEach(deliver); } catch (_) {}
    setTimeout(poll, 1500);
  };
  open(); setTimeout(poll, 1500);
  return {get last() { return last; }};
}

const ICON = {
  check: '<svg viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5l3 3 6-7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  x: '<svg viewBox="0 0 16 16" fill="none"><path d="M4.5 4.5l7 7m0-7l-7 7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  mic: '<svg viewBox="0 0 24 24" fill="none"><rect x="9" y="3" width="6" height="11" rx="3" stroke="currentColor" stroke-width="1.7"/><path d="M5.5 11a6.5 6.5 0 0013 0M12 17.5V21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none"><path d="M5 12h13M13 6l6 6-6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  palm: '<svg viewBox="0 0 24 24" fill="none"><path d="M8 12V5.5a1.5 1.5 0 013 0V11m0-6.5a1.5 1.5 0 013 0V11m0-5a1.5 1.5 0 013 0v7.5c0 4-2.5 7-6.5 7-3 0-4.6-1.6-6-4l-2-3.4a1.5 1.5 0 012.5-1.6L8 13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  wave: '<svg viewBox="0 0 24 24" fill="none"><path d="M3 12h2m3-5v10m4-13v16m4-11v6m4-3h1" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
  key: '<svg viewBox="0 0 24 24" fill="none"><circle cx="8" cy="15" r="4" stroke="currentColor" stroke-width="1.6"/><path d="M11 12l8-8m-3 3l2 2m-5 1l2 2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  finger: '<svg viewBox="0 0 24 24" fill="none"><path d="M7.5 20c1-2 1.5-4.5 1.5-7.5a3 3 0 016 0c0 1 0 2-.2 3M12 12.5c0 3.5-.8 6.3-2.3 8.5M16.8 18.5c.5-1.6.7-3.6.7-6a5.5 5.5 0 00-11 0c0 1.5-.2 3-.7 4.3M5 8.2A8 8 0 0119 8.5M15 21c.3-.6.6-1.2.8-1.9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  reset: '<svg viewBox="0 0 24 24" fill="none"><path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4h4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};
const check = (ok, yes = "verified", no = "failed") =>
  `<span class="check ${ok ? "y" : "n"}">${ok ? ICON.check : ICON.x}${ok ? yes : no}</span>`;

function hexHTML(h, max = 0) {
  h = String(h); const s = max && h.length > max ? h.slice(0, max) + "…" : h;
  return `<span class="hex"><span class="h0">${esc(s.slice(0, 8))}</span>${esc(s.slice(8))}</span>`;
}

function toast(msg) {
  let t = $(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.append(t); }
  t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2600);
}

// ── Jarvis voice: ElevenLabs through the server (streams), browser voice only as a fallback ──
const Voice = {
  enabled: true, audio: null, onstart: null, onend: null, _seq: 0,
  say(text) {
    if (!this.enabled || !text) { this.onend?.(); return; }
    const seq = ++this._seq;
    this.stop(false);
    const a = new Audio(`/api/tts?text=${encodeURIComponent(text)}`);
    this.audio = a; let started = false;
    a.onplaying = () => { started = true; this.onstart?.(); };
    a.onended = () => { if (seq === this._seq) this.onend?.(); };
    a.onerror = () => { if (seq === this._seq && !started) this._browser(text); };
    a.play().catch(() => { if (seq === this._seq && !started) this._browser(text); });
  },
  stop(fire = true) { if (this.audio) { this.audio.onended = this.audio.onerror = null; this.audio.pause(); this.audio = null; }
    if ("speechSynthesis" in window) speechSynthesis.cancel(); if (fire) this.onend?.(); },
  _browser(text) {
    if (!("speechSynthesis" in window)) { this.onend?.(); return; }
    const u = new SpeechSynthesisUtterance(text.replace(/AED/g, "dirhams"));
    const v = speechSynthesis.getVoices().find(v => /en-GB/.test(v.lang)) || null; if (v) u.voice = v;
    u.onstart = () => this.onstart?.(); u.onend = () => this.onend?.(); u.onerror = () => this.onend?.();
    speechSynthesis.speak(u);
  },
};

// ── Jarvis ears: record in any browser, transcribe on the laptop with whisper.cpp ──
const Ears = {
  stream: null, ctx: null, node: null, chunks: [], rate: 48000, on: false,
  async start({onLevel, onDone, maxMs = 9000} = {}) {
    if (this.on) return this.stop();
    this.stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, channelCount: 1}});
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.rate = this.ctx.sampleRate; this.chunks = []; this.on = true; this.onDone = onDone;
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    let spoke = false, quietSince = 0; const t0 = performance.now();
    this.node.onaudioprocess = e => {
      if (!this.on) return;
      const d = e.inputBuffer.getChannelData(0); this.chunks.push(new Float32Array(d));
      let rms = 0; for (let i = 0; i < d.length; i += 4) rms += d[i] * d[i]; rms = Math.sqrt(rms / (d.length / 4));
      onLevel?.(Math.min(1, rms * 8));
      const now = performance.now();
      if (rms > 0.02) { spoke = true; quietSince = 0; } else if (spoke) { quietSince ||= now; }
      if ((spoke && quietSince && now - quietSince > 1100) || now - t0 > maxMs || (!spoke && now - t0 > 6000)) this.stop();
    };
    src.connect(this.node); this.node.connect(this.ctx.destination);
  },
  stop() {
    if (!this.on) return; this.on = false;
    this.node?.disconnect(); this.stream?.getTracks().forEach(t => t.stop()); this.ctx?.close();
    const wav = this._wav(); this.onDone?.(wav);
  },
  _wav() {
    const len = this.chunks.reduce((a, c) => a + c.length, 0), all = new Float32Array(len); let o = 0;
    for (const c of this.chunks) { all.set(c, o); o += c.length; }
    const ratio = this.rate / 16000, n = Math.floor(len / ratio), pcm = new Int16Array(n);
    for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, all[Math.floor(i * ratio)])); pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff; }
    const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
    const w = (p, s) => [...s].forEach((c, i) => v.setUint8(p + i, c.charCodeAt(0)));
    w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, pcm.length * 2, true);
    new Int16Array(buf, 44).set(pcm);
    return new Blob([buf], {type: "audio/wav"});
  },
};

// ── sound (synthesised, no files) ─────────────────────────────────────────
const Sound = {
  ctx: null, on: true,
  _c() { return this.ctx ||= new (window.AudioContext || window.webkitAudioContext)(); },
  tone(freqs, dur = .18, type = "sine", gain = .06) {
    if (!this.on) return; const c = this._c(); const t = c.currentTime;
    freqs.forEach((f, i) => {
      const o = c.createOscillator(), g = c.createGain(); o.type = type; o.frequency.value = f;
      g.gain.setValueAtTime(0, t + i * .09); g.gain.linearRampToValueAtTime(gain, t + i * .09 + .015);
      g.gain.exponentialRampToValueAtTime(.0001, t + i * .09 + dur);
      o.connect(g).connect(c.destination); o.start(t + i * .09); o.stop(t + i * .09 + dur + .05);
    });
  },
  tick() { this.tone([1760], .05, "sine", .02); },
  ok() { this.tone([660, 880, 1320], .35, "sine", .05); },
  no() { this.tone([300, 220], .3, "triangle", .05); },
  credit() { this.tone([523, 659, 784, 1047], .6, "sine", .06); },
};

// ── the orb ───────────────────────────────────────────────────────────────
function Orb(canvas) {
  const ctx = canvas.getContext("2d"); let state = "idle", flash = 0, flashColor = "#3fe7ab", t0 = performance.now(), level = 0;
  const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();
  function size() { const r = canvas.getBoundingClientRect(), d = devicePixelRatio || 1; canvas.width = r.width * d; canvas.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0); }
  size(); addEventListener("resize", size);
  function frame(now) {
    const w = canvas.clientWidth, h = canvas.clientHeight, cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 6, t = (now - t0) / 1000;
    const acc = state === "stepup" ? css("--amber") : state === "error" ? css("--red") : css("--accent");
    const target = {idle: .12, listening: .75, thinking: .45, speaking: .9, stepup: .5, error: .3}[state] ?? .12;
    level += (target - level) * .06;
    ctx.clearRect(0, 0, w, h);
    // core glow
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * .62);
    g.addColorStop(0, acc + "cc"); g.addColorStop(.35, acc + "33"); g.addColorStop(1, "transparent");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R * (.55 + level * .06 * Math.sin(t * 6)), 0, 7); ctx.fill();
    // waveform ring
    ctx.strokeStyle = acc; ctx.lineWidth = 1.4; ctx.globalAlpha = .9; ctx.beginPath();
    for (let i = 0; i <= 180; i++) {
      const a = i / 180 * Math.PI * 2;
      const n = Math.sin(a * 6 + t * 3.1) * .5 + Math.sin(a * 11 - t * 4.3) * .3 + Math.sin(a * 3 + t * 1.7) * .2;
      const r = R * .42 + n * R * .07 * level * (state === "idle" ? .4 : 1);
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath(); ctx.stroke();
    // rotating arcs
    ctx.globalAlpha = .7; ctx.lineWidth = 1.2;
    [[.66, .9, 1], [.74, -.55, 2.1], [.84, .35, .7]].forEach(([k, sp, len], j) => {
      ctx.beginPath(); ctx.arc(cx, cy, R * k, t * sp + j, t * sp + j + len); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, R * k, t * sp + j + Math.PI, t * sp + j + Math.PI + len * .5); ctx.stroke();
    });
    // tick ring
    ctx.globalAlpha = .35; ctx.lineWidth = 1;
    for (let i = 0; i < 72; i++) {
      const a = i / 72 * Math.PI * 2 + t * .05, l = i % 6 === 0 ? 7 : 3;
      ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * R * .95, cy + Math.sin(a) * R * .95);
      ctx.lineTo(cx + Math.cos(a) * (R * .95 - l), cy + Math.sin(a) * (R * .95 - l)); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (flash > 0) {
      ctx.strokeStyle = flashColor; ctx.globalAlpha = flash; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, R * (1 - flash * .35), 0, 7); ctx.stroke(); ctx.globalAlpha = 1; flash -= .018;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return {set(s) { state = s; }, get state() { return state; },
          pulse(color) { flashColor = color || css("--ok"); flash = 1; }};
}

// animated number
function countTo(el, to, d = 2) {
  const from = parseFloat(el.dataset.v || "0"), t0 = performance.now(), dur = 900;
  el.dataset.v = to;
  const step = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e, d);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
