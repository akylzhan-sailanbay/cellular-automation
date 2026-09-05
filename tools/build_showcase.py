"""Inject a showcase bundle into the replay page. No simulation logic here."""
import json
import sys
from pathlib import Path

TEMPLATE = r"""<title>Evolution Showcase — Predation vs No Predation</title>
<style>
:root{
  --void:#070B0D; --panel:#0D1417; --panel2:#111A1E; --line:#1E2B30;
  --ink:#DEE7E5; --dim:#7C8F8D; --faint:#4A5A5A;
  --herb:#4FB79C; --carn:#E2914F; --hot:#E2914F; --dead:#C4574A;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--void);color:var(--ink);font-family:var(--sans);
  padding:clamp(1rem,3vw,2.25rem);line-height:1.5;-webkit-font-smoothing:antialiased}
.shell{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}

header{display:flex;flex-direction:column;gap:.55rem}
.kicker{font-family:var(--mono);font-size:.66rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--dim)}
h1{font-family:var(--mono);font-size:clamp(1.35rem,3.4vw,2rem);margin:0;
  font-weight:600;letter-spacing:-.02em;text-wrap:balance}
.setup{color:var(--dim);font-size:.92rem;max-width:70ch;margin:0}
.setup b{color:var(--ink);font-weight:600}

.arena{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:760px){.arena{grid-template-columns:1fr}}
.world{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  overflow:hidden;display:flex;flex-direction:column}
.wtitle{display:flex;justify-content:space-between;align-items:baseline;
  padding:.6rem .8rem;border-bottom:1px solid var(--line);font-family:var(--mono);
  font-size:.75rem;letter-spacing:.04em}
.wtitle .tag{font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim)}
canvas.field{display:block;width:100%;height:auto;background:#04080A}
.hud{padding:.7rem .8rem .85rem;display:flex;flex-direction:column;gap:.5rem;
  font-family:var(--mono);font-variant-numeric:tabular-nums}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;font-size:.72rem}
.stat{display:flex;flex-direction:column;gap:.1rem}
.stat .k{color:var(--faint);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase}
.stat .v{font-size:1.02rem;font-weight:600}
.meter{display:flex;flex-direction:column;gap:.25rem}
.meter .row{display:flex;justify-content:space-between;font-size:.62rem;
  letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.meter .row b{color:var(--ink);font-size:.78rem;letter-spacing:0;text-transform:none}
.track{height:.5rem;background:var(--panel2);border-radius:2px;overflow:hidden}
.fill{height:100%;width:0;background:var(--herb);transition:width .12s linear}
.extinct{color:var(--dead);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase}

.transport{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  padding:.85rem 1rem;display:flex;flex-direction:column;gap:.75rem}
.tl{position:relative;height:2.4rem}
.tl input{position:absolute;inset:auto 0 0;width:100%;margin:0;accent-color:var(--hot)}
.marks{position:absolute;inset:0 0 auto;height:1.5rem}
.mark{position:absolute;top:0;transform:translateX(-50%);display:flex;
  flex-direction:column;align-items:center;gap:.15rem;cursor:pointer;
  background:none;border:0;padding:0;color:var(--dim);font-family:var(--mono);
  font-size:.56rem;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
.mark:hover,.mark:focus-visible{color:var(--hot);outline:none}
.mark i{display:block;width:1px;height:.6rem;background:currentColor}
.controls{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;
  font-family:var(--mono);font-size:.72rem}
button{font-family:var(--mono);font-size:.72rem;background:var(--panel2);
  color:var(--ink);border:1px solid var(--line);border-radius:3px;
  padding:.35rem .7rem;cursor:pointer}
button:hover{border-color:var(--dim)}
button[aria-pressed="true"]{background:var(--hot);color:#0A0F10;border-color:var(--hot);
  font-weight:600}
button:focus-visible{outline:2px solid var(--hot);outline-offset:2px}
.clock{margin-left:auto;color:var(--dim);font-variant-numeric:tabular-nums}
.clock b{color:var(--ink)}

.brains{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:760px){.brains{grid-template-columns:1fr}}
.brain{background:var(--panel);border:1px solid var(--line);border-radius:5px;
  padding:.7rem .8rem;display:flex;flex-direction:column;gap:.5rem}
.brain h3{margin:0;font-family:var(--mono);font-size:.66rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);font-weight:600}
canvas.net{display:block;width:100%;height:auto;background:#04080A;border-radius:3px}
.bmeta{font-family:var(--mono);font-size:.68rem;color:var(--dim);
  font-variant-numeric:tabular-nums}
.bmeta b{color:var(--ink)}

.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;font-family:var(--mono);
  font-size:.66rem;color:var(--dim);align-items:center}
.sw{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;
  margin-right:.35rem;vertical-align:-1px}
.verdict{border-top:1px solid var(--line);padding-top:1rem;color:var(--dim);
  font-size:.85rem;max-width:78ch}
.verdict b{color:var(--ink)}
</style>

<div class="shell">
<header>
  <span class="kicker">Replay · real output from the tested engine</span>
  <h1>Does being hunted make you smarter?</h1>
  <p class="setup">Two worlds, <b>same seed</b>, <b>same constants</b>, one difference:
  on the left, agents can kill each other. Nothing here is simulated in your browser —
  every frame is recorded output from the Python engine that passes 89 tests. Watch the
  <b>hidden neurons</b> bar.</p>
</header>

<div class="arena" id="arena"></div>

<div class="transport">
  <div class="tl">
    <div class="marks" id="marks"></div>
    <input id="scrub" type="range" min="0" value="0" step="1" aria-label="Timeline">
  </div>
  <div class="controls">
    <button id="play" aria-pressed="false">▶ Play</button>
    <button class="sp" data-s="1" aria-pressed="true">1×</button>
    <button class="sp" data-s="4" aria-pressed="false">4×</button>
    <button class="sp" data-s="16" aria-pressed="false">16×</button>
    <button id="restart">↺ Restart</button>
    <span class="clock">tick <b id="tick">0</b> · generation <b id="gen">0</b></span>
  </div>
  <div class="legend">
    <span><span class="sw" style="background:#4FB79C"></span>herbivore (diet 0)</span>
    <span><span class="sw" style="background:#E2914F"></span>carnivore (diet 1)</span>
    <span>dot size = brain connections</span>
    <span>green field = plant density</span>
  </div>
</div>

<div class="brains" id="brains"></div>

<p class="verdict"><b>What you are watching.</b> The left world ends with roughly 5–9×
more hidden neurons than the right. Both start identical: 10 flat input-to-output
connections, zero hidden. A world of plants alone does not reward cognition; a world
containing things that kill you does. Timeline markers are computed from the recorded
data, not authored — first structure, peak complexity, and population crash.</p>
</div>

<script>
const DATA = __BUNDLE__;
const AB = DATA.meta.alphabet, BINS = DATA.meta.plant_bins, GRID = DATA.meta.grid;
const IDX = {}; for (let i=0;i<AB.length;i++) IDX[AB[i]] = i;
const W = DATA.worlds;
const LEN = Math.max(...W.map(w => w.frames.length));
const PX = 8, SIZE = GRID * PX;

function lerp(a,b,t){return a+(b-a)*t}
function dietColor(d){ // teal -> amber, the simulation's own axis
  const t = d/9;
  return `rgb(${Math.round(lerp(79,226,t))},${Math.round(lerp(183,145,t))},${Math.round(lerp(156,79,t))})`;
}

// ---- build world panels -------------------------------------------------
const arena = document.getElementById('arena');
const panels = W.map((w,i) => {
  const el = document.createElement('div'); el.className = 'world';
  el.innerHTML = `
    <div class="wtitle"><span>${w.name}</span>
      <span class="tag">${w.attack ? 'attack enabled' : 'attack disabled'}</span></div>
    <canvas class="field" width="${SIZE}" height="${SIZE}"></canvas>
    <div class="hud">
      <div class="stats">
        <div class="stat"><span class="k">Population</span><span class="v" data-f="pop">—</span></div>
        <div class="stat"><span class="k">Mean links</span><span class="v" data-f="links">—</span></div>
        <div class="stat"><span class="k">Diet</span><span class="v" data-f="diet">—</span></div>
      </div>
      <div class="meter">
        <div class="row"><span>Hidden neurons</span><b data-f="hidden">—</b></div>
        <div class="track"><div class="fill"></div></div>
      </div>
      <div class="extinct" hidden></div>
    </div>`;
  arena.appendChild(el);
  return {
    w, ctx: el.querySelector('canvas').getContext('2d'),
    f: Object.fromEntries([...el.querySelectorAll('[data-f]')].map(n=>[n.dataset.f,n])),
    fill: el.querySelector('.fill'), ext: el.querySelector('.extinct'),
  };
});

// hidden-neuron bar shares one scale across worlds or the comparison lies
const HMAX = Math.max(0.5, ...W.flatMap(w => w.frames.map(f => f.s[2])));

// ---- champion brain panels ---------------------------------------------
const brains = document.getElementById('brains');
const bpanels = W.map(w => {
  const el = document.createElement('div'); el.className = 'brain';
  el.innerHTML = `<h3>${w.name} — best brain so far</h3>
    <canvas class="net" width="520" height="200"></canvas>
    <div class="bmeta">—</div>`;
  brains.appendChild(el);
  return {w, ctx: el.querySelector('canvas').getContext('2d'),
          meta: el.querySelector('.bmeta')};
});

function drawNet(p, tick){
  const champs = p.w.champions.filter(c => c.tick <= tick);
  const c = champs.length ? champs[champs.length-1] : null;
  const x = p.ctx, Wd = x.canvas.width, H = x.canvas.height;
  x.fillStyle = '#04080A'; x.fillRect(0,0,Wd,H);
  if (!c){ p.meta.textContent = 'no snapshot yet'; return; }
  const cols = {i:[], h:[], o:[]};
  c.nodes.forEach(n => cols[n[1]].push(n));
  const pos = {};
  const place = (arr, cx) => arr.forEach((n,k) => {
    pos[n[0]] = [cx, 16 + (H-32) * (arr.length===1?0.5:k/(arr.length-1))];
  });
  place(cols.i, 34); place(cols.h, Wd/2); place(cols.o, Wd-34);
  x.lineWidth = 1;
  c.conns.forEach(([s,d,w0]) => {
    if (!pos[s] || !pos[d]) return;
    const mag = Math.min(1, Math.abs(w0)/2.5);
    x.strokeStyle = w0 >= 0 ? `rgba(79,183,156,${0.15+0.55*mag})`
                            : `rgba(226,145,79,${0.15+0.55*mag})`;
    x.beginPath(); x.moveTo(pos[s][0],pos[s][1]); x.lineTo(pos[d][0],pos[d][1]); x.stroke();
  });
  c.nodes.forEach(n => {
    const [px,py] = pos[n[0]];
    x.beginPath(); x.arc(px,py, n[1]==='h'?4.5:3, 0, 6.2832);
    x.fillStyle = n[1]==='h' ? '#E2914F' : (n[1]==='i' ? '#4FB79C' : '#DEE7E5');
    x.fill();
  });
  p.meta.innerHTML = `tick <b>${c.tick.toLocaleString()}</b> · <b>${c.links}</b> connections · <b>${c.hidden}</b> hidden`;
}

// ---- frame rendering ----------------------------------------------------
function frameAt(w, i){ return w.frames[Math.min(i, w.frames.length-1)] || null; }

function draw(i){
  panels.forEach(p => {
    const f = frameAt(p.w, i);
    const x = p.ctx;
    x.fillStyle = '#04080A'; x.fillRect(0,0,SIZE,SIZE);
    if (!f) return;
    const cell = SIZE / BINS;
    for (let k=0;k<f.p.length;k++){
      const v = IDX[f.p[k]] / 9;
      if (v <= 0) continue;
      x.fillStyle = `rgba(46,${Math.round(70+110*v)},72,${0.18+0.62*v})`;
      x.fillRect((k % BINS)*cell, Math.floor(k/BINS)*cell, cell, cell);
    }
    for (let k=0;k<f.a.length;k+=4){
      const ax=IDX[f.a[k]], ay=IDX[f.a[k+1]], d=IDX[f.a[k+2]], b=IDX[f.a[k+3]];
      x.beginPath();
      x.arc(ax*PX+PX/2, ay*PX+PX/2, 1.9 + b*0.42, 0, 6.2832);
      x.fillStyle = dietColor(d); x.fill();
    }
    const past = p.w.extinct !== null && p.w.frames.length && i >= p.w.frames.length-1;
    p.ext.hidden = !past;
    if (past) p.ext.textContent = `Extinct at tick ${p.w.extinct.toLocaleString()}`;
    p.f.pop.textContent = f.s[0];
    p.f.links.textContent = f.s[1].toFixed(2);
    p.f.diet.textContent = f.s[4].toFixed(3);
    p.f.hidden.textContent = f.s[2].toFixed(2);
    p.fill.style.width = (100*Math.min(1, f.s[2]/HMAX)) + '%';
    p.fill.style.background = f.s[2] > 0.5 ? '#E2914F' : '#4FB79C';
  });
  const lead = frameAt(W[0], i);
  document.getElementById('tick').textContent = lead ? lead.t.toLocaleString() : '0';
  document.getElementById('gen').textContent = lead ? Math.round(lead.s[3]) : '0';
  bpanels.forEach(p => drawNet(p, lead ? lead.t : 0));
}

// ---- timeline markers (computed upstream, not authored) -----------------
const LABELS = {first_structure:'first structure', peak_complexity:'peak complexity',
                crash:'crash'};
const marks = document.getElementById('marks');
const seen = new Set();
W.forEach(w => Object.entries(w.milestones).forEach(([k,tick]) => {
  if (seen.has(k)) return; seen.add(k);
  const idx = w.frames.findIndex(f => f.t >= tick);
  if (idx < 0) return;
  const b = document.createElement('button');
  b.className='mark'; b.style.left = (100*idx/Math.max(1,LEN-1))+'%';
  b.innerHTML = `<i></i>${LABELS[k]||k}`;
  b.title = `${LABELS[k]||k} — tick ${tick.toLocaleString()}`;
  b.onclick = () => { setFrame(idx); };
  marks.appendChild(b);
}));

// ---- transport ----------------------------------------------------------
const scrub = document.getElementById('scrub');
scrub.max = LEN - 1;
let cur = 0, playing = false, speed = 1, acc = 0, last = 0;
function setFrame(i){ cur = Math.max(0, Math.min(LEN-1, i)); scrub.value = cur; draw(cur); }
scrub.oninput = () => setFrame(+scrub.value);

const playBtn = document.getElementById('play');
function setPlaying(v){
  playing = v;
  playBtn.setAttribute('aria-pressed', String(v));
  playBtn.textContent = v ? '❚❚ Pause' : '▶ Play';
  if (v){ last = performance.now(); requestAnimationFrame(step); }
}
playBtn.onclick = () => setPlaying(!playing);
document.getElementById('restart').onclick = () => { setFrame(0); setPlaying(true); };
document.querySelectorAll('.sp').forEach(b => b.onclick = () => {
  speed = +b.dataset.s;
  document.querySelectorAll('.sp').forEach(o =>
    o.setAttribute('aria-pressed', String(o === b)));
});
function step(now){
  if (!playing) return;
  acc += (now - last) / 1000 * 12 * speed; last = now;
  if (acc >= 1){
    const adv = Math.floor(acc); acc -= adv;
    if (cur + adv >= LEN - 1){ setFrame(LEN-1); setPlaying(false); return; }
    setFrame(cur + adv);
  }
  requestAnimationFrame(step);
}
addEventListener('keydown', e => {
  if (e.key === ' '){ e.preventDefault(); setPlaying(!playing); }
  if (e.key === 'ArrowRight') setFrame(cur+1);
  if (e.key === 'ArrowLeft') setFrame(cur-1);
});

setFrame(0);
if (!matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(true);
</script>
"""


def build(bundle_path: str | Path, out_path: str | Path) -> int:
    bundle = json.loads(Path(bundle_path).read_text())
    page = TEMPLATE.replace("__BUNDLE__", json.dumps(bundle, separators=(",", ":")))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(page)
    return len(page)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/showcase.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "docs/results/showcase.html"
    n = build(src, dst)
    print(f"{dst}: {n/1024/1024:.2f} MB")
