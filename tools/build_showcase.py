"""Inject a showcase bundle into the replay page. No simulation logic here."""
import json
import sys
from pathlib import Path

TEMPLATE = r"""<title>Do bullies make you smarter?</title>
<style>
:root{
  --void:#090E11; --panel:#131E22; --panel2:#1B282E; --line:#28383F;
  --ink:#EAF2F0; --dim:#9BAFAD; --safe:#5FC9A9; --danger:#F58F52;
  --gold:#FFD277; --dead:#E56F5B;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--void);color:var(--ink);font-family:var(--sans);
  padding:clamp(1rem,3vw,2rem);line-height:1.55;-webkit-font-smoothing:antialiased}
.shell{max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}
h1{font-size:clamp(1.75rem,5vw,2.9rem);margin:0;line-height:1.07;font-weight:750;
  letter-spacing:-.025em;text-wrap:balance}
.sub{font-size:clamp(1.02rem,2.2vw,1.22rem);color:var(--dim);margin:.7rem 0 0;max-width:62ch}
.sub b{color:var(--ink)}

.score{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:1.15rem 1.25rem 1.35rem;display:flex;flex-direction:column;gap:1rem}
.score h2{margin:0;font-size:.82rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--dim);font-weight:700}
.bar{display:grid;grid-template-columns:13rem 1fr 4.4rem;gap:1rem;align-items:center}
@media(max-width:660px){.bar{grid-template-columns:1fr;gap:.35rem}}
.who{font-weight:680;font-size:1.02rem;display:flex;align-items:center;gap:.6rem}
.rail{height:2.4rem;background:var(--panel2);border-radius:7px;overflow:hidden}
.grow{height:100%;width:0;border-radius:7px;transition:width .35s ease-out}
.num{font-family:var(--mono);font-size:1.65rem;font-weight:750;text-align:right;
  font-variant-numeric:tabular-nums}
.callout{font-size:1.06rem;color:var(--dim);margin:0}
.callout b{color:var(--gold)}

.say{background:var(--panel2);border:1px solid var(--line);
  border-left:4px solid var(--gold);border-radius:9px;padding:1rem 1.15rem;
  font-size:1.1rem;min-height:4.3rem;display:flex;align-items:center}
.say b{color:var(--gold)}

.arena{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}
@media(max-width:860px){.arena{grid-template-columns:1fr}}
.world{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;display:flex;flex-direction:column}
.world.danger{border-color:#5E3928}
.wtop{padding:.8rem .95rem;display:flex;align-items:center;gap:.6rem;
  border-bottom:1px solid var(--line)}
.wtop .name{font-weight:700;font-size:1.06rem}
.wtop .note{margin-left:auto;font-size:.8rem;color:var(--dim)}
canvas.field{display:block;width:100%;height:auto;background:#07100F}
.wfoot{padding:.7rem .95rem .85rem;display:flex;gap:1rem;align-items:center;
  font-size:.9rem;color:var(--dim);flex-wrap:wrap}
.wfoot b{color:var(--ink);font-family:var(--mono);font-variant-numeric:tabular-nums}
canvas.mini{width:74px;height:74px;border:1px solid var(--line);border-radius:5px;
  background:#07100F;margin-left:auto;flex:none}
.gone{color:var(--dead);font-weight:700}

.bay{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.8rem}
.tl{position:relative;height:2.7rem}
.tl input{position:absolute;inset:auto 0 .1rem;width:100%;margin:0;
  accent-color:var(--gold);height:1.2rem}
.marks{position:absolute;inset:0 0 auto;height:1.6rem;pointer-events:none}
.mark{position:absolute;top:0;transform:translateX(-50%);pointer-events:auto;
  background:none;border:0;padding:.1rem .25rem;cursor:pointer;color:var(--dim);
  font-size:.75rem;white-space:nowrap;display:flex;flex-direction:column;
  align-items:center;gap:.12rem;font-family:var(--sans)}
.mark i{width:2px;height:.6rem;background:currentColor;border-radius:1px}
.mark:hover,.mark:focus-visible{color:var(--gold);outline:none}
.row{display:flex;gap:.55rem;align-items:center;flex-wrap:wrap}
button{font-family:var(--sans);font-size:.92rem;font-weight:620;
  background:var(--panel2);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:.55rem 1rem;cursor:pointer}
button:hover{border-color:var(--dim)}
button[aria-pressed="true"]{background:var(--gold);color:#14191B;border-color:var(--gold)}
button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
#play{min-width:7rem}
.time{margin-left:auto;color:var(--dim);font-size:.9rem}
.time b{color:var(--ink);font-family:var(--mono)}
.key{display:flex;flex-wrap:wrap;gap:.6rem 1.5rem;font-size:.9rem;color:var(--dim);
  align-items:center;border-top:1px solid var(--line);padding-top:.8rem}
.key canvas{vertical-align:-6px;margin-right:.45rem}
details{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:.9rem 1.1rem;color:var(--dim);font-size:.95rem}
summary{cursor:pointer;color:var(--ink);font-weight:660;font-size:1rem}
summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
details p{margin:.75rem 0 0} details b{color:var(--ink)}
.foot{color:var(--dim);font-size:.95rem;max-width:68ch;margin:0}
.foot b{color:var(--ink)}
</style>

<div class="shell">
<header>
  <h1>Do bullies make you smarter?</h1>
  <p class="sub">Two worlds of little creatures. Both start with <b>exactly the same
  dumb creatures</b>. On the left they can attack each other. On the right they can't.
  Nobody taught them anything. Watch which side grows bigger brains.</p>
</header>

<div class="score">
  <h2>Brain size right now</h2>
  <div class="bar">
    <span class="who"><canvas class="ic" width="60" height="52" data-diet="8"></canvas>Left: bullies allowed</span>
    <span class="rail"><span class="grow" id="g0" style="background:var(--danger)"></span></span>
    <span class="num" id="n0" style="color:var(--danger)">0.0</span>
  </div>
  <div class="bar">
    <span class="who"><canvas class="ic" width="60" height="52" data-diet="1"></canvas>Right: everyone's nice</span>
    <span class="rail"><span class="grow" id="g1" style="background:var(--safe)"></span></span>
    <span class="num" id="n1" style="color:var(--safe)">0.0</span>
  </div>
  <p class="callout" id="ratio">Both sides start with completely empty brains.</p>
</div>

<div class="say"><span id="narr">Hit play. The camera follows the crowd — the little map in each corner shows where you are.</span></div>

<div class="arena" id="arena"></div>

<div class="bay">
  <div class="tl">
    <div class="marks" id="marks"></div>
    <input id="scrub" type="range" min="0" value="0" step="1" aria-label="Timeline">
  </div>
  <div class="row">
    <button id="play" aria-pressed="false">▶ Play</button>
    <button class="sp" data-s="0.5" aria-pressed="false">Slow</button>
    <button class="sp" data-s="1" aria-pressed="true">Normal</button>
    <button class="sp" data-s="4" aria-pressed="false">Fast</button>
    <button class="sp" data-s="12" aria-pressed="false">Skip ahead</button>
    <button id="restart">↺ Start over</button>
    <span class="time">family tree: <b id="gen">0</b> generations deep</span>
  </div>
  <div class="key">
    <span><canvas class="ic" width="56" height="48" data-diet="1"></canvas>eats plants</span>
    <span><canvas class="ic" width="56" height="48" data-diet="8"></canvas>eats meat — spiky, orange</span>
    <span>bigger body = bigger brain</span>
    <span>green blobs = food growing</span>
  </div>
</div>

<details>
  <summary>What am I actually looking at?</summary>
  <p>Each creature has a tiny brain — a network that looks at what's nearby and decides:
  move, eat, attack, or have a baby. Babies get a slightly mutated copy of their parent's
  brain. There's no score and no goal. Creatures that can't find food just die, and their
  family line ends there.</p>
  <p><b>The only difference between the two sides is whether attacking is switched on.</b>
  Same starting creatures, same food, same everything. So whatever difference you see,
  evolution made it — we didn't.</p>
  <p>The world is bigger than what you can see. The camera stays with the crowd, and the
  little map in the bottom corner of each panel shows the whole thing, with a box marking
  where you're looking.</p>
  <p>The bars up top count <b>extra brain cells</b>: brain bits beyond the bare wiring
  every creature is born with. Zero means pure reflex. Higher means evolution built
  something in the middle that can actually think a bit.</p>
</details>

<p class="foot"><b>One honest catch.</b> The nice world does occasionally produce a single
clever creature. The difference is whether smart <i>spreads</i>. By the end, the average
creature on the bully side has about <b>9 times</b> more extra brain cells than the average
on the nice side. One world makes rare geniuses. The other makes brains normal.</p>
</div>

<script>
const DATA = __BUNDLE__;
const AB = DATA.meta.alphabet, BINS = DATA.meta.plant_bins, GRID = DATA.meta.grid;
const IDX = {}; for (let i=0;i<AB.length;i++) IDX[AB[i]] = i;
const W = DATA.worlds;
const LEN = Math.max(...W.map(w => w.frames.length));
const HMAX = Math.max(0.5, ...W.flatMap(w => w.frames.map(f => f.s[2])));

/* Camera: 64x64 cells across one panel leaves ~10px per creature, far too small
   to read. Show a 22-cell window instead and let a minimap carry the context. */
const VIEW = 22, CELL = 42, PAD = 26;
const SIZE = VIEW * CELL + PAD * 2;

function lerp(a,b,t){return a+(b-a)*t}

/* Precompute a smoothed camera path so scrubbing lands in the same place
   playback would -- a camera smoothed live would drift depending on how you
   got to a frame. */
function cameraPath(w){
  const path = []; let cx = GRID/2, cy = GRID/2;
  for (const f of w.frames){
    let sx = 0, sy = 0, n = f.a.length/4;
    for (let j=0;j<f.a.length;j+=4){ sx += IDX[f.a[j]]; sy += IDX[f.a[j+1]]; }
    const tx = n ? sx/n : GRID/2, ty = n ? sy/n : GRID/2;
    cx = lerp(cx, tx, 0.08); cy = lerp(cy, ty, 0.08);
    const half = VIEW/2, lo = half, hi = GRID - half;
    path.push([Math.min(hi,Math.max(lo,cx)), Math.min(hi,Math.max(lo,cy))]);
  }
  return path;
}

/* ---------------- creature ---------------- */
function creature(x, cx, cy, r, diet, phase){
  const meat = diet >= 5;
  const skin = meat ? '#F58F52' : '#5FC9A9';
  const lite = meat ? '#FFB47E' : '#8BE3C6';
  const edge = meat ? '#8E3A18' : '#1E6A56';
  const bob  = Math.sin(phase) * r * 0.11;
  const y = cy + bob;

  x.save();
  x.beginPath(); x.ellipse(cx, cy + r*0.92, r*0.78, r*0.22, 0, 0, 6.2832);
  x.fillStyle = 'rgba(0,0,0,.28)'; x.fill();                     // ground shadow

  if (meat){                                                     // spiky back
    x.fillStyle = edge;
    for (let k=-1;k<=1;k++){
      const sx = cx + k*r*0.54;
      x.beginPath();
      x.moveTo(sx - r*0.19, y - r*0.70);
      x.lineTo(sx,          y - r*1.42);
      x.lineTo(sx + r*0.19, y - r*0.70);
      x.closePath(); x.fill();
    }
  }
  const g = x.createLinearGradient(cx, y - r, cx, y + r);
  g.addColorStop(0, lite); g.addColorStop(1, skin);
  x.beginPath(); x.ellipse(cx, y, r, r*0.88, 0, 0, 6.2832);
  x.fillStyle = g; x.fill();
  x.lineWidth = Math.max(1.4, r*0.13); x.strokeStyle = edge; x.stroke();

  const ex = r*0.37, ey = -r*0.12, er = Math.max(2, r*0.29);
  for (const s of [-1,1]){
    x.beginPath(); x.arc(cx+s*ex, y+ey, er, 0, 6.2832);
    x.fillStyle = '#FFFFFF'; x.fill();
    x.beginPath(); x.arc(cx+s*ex+er*0.2, y+ey+er*0.12, er*0.5, 0, 6.2832);
    x.fillStyle = '#0E1C1A'; x.fill();
    x.beginPath(); x.arc(cx+s*ex-er*0.28, y+ey-er*0.3, er*0.2, 0, 6.2832);
    x.fillStyle = 'rgba(255,255,255,.9)'; x.fill();              // catchlight
  }
  x.beginPath();
  if (meat){ x.moveTo(cx-r*0.3, y+r*0.36); x.lineTo(cx, y+r*0.56); x.lineTo(cx+r*0.3, y+r*0.36); }
  else { x.arc(cx, y+r*0.34, r*0.22, 0.15*Math.PI, 0.85*Math.PI); }
  x.lineWidth = Math.max(1.2, r*0.10); x.strokeStyle = edge; x.stroke();
  x.restore();
}

/* icons in the legend and scoreboard use the same drawing routine */
document.querySelectorAll('canvas.ic').forEach(c => {
  const x = c.getContext('2d');
  creature(x, c.width/2, c.height/2 + 2, 17, +c.dataset.diet, 0);
});

/* ---------------- panels ---------------- */
const LABEL = [
  {name:'Bullies allowed', note:'they can attack each other', cls:'danger'},
  {name:"Everyone's nice", note:'attacking switched off',     cls:''},
];
const arena = document.getElementById('arena');
const panels = W.map((w,i) => {
  const L = LABEL[i], el = document.createElement('div');
  el.className = 'world ' + L.cls;
  el.innerHTML = `
    <div class="wtop"><canvas class="ic" width="52" height="46" data-diet="${i?1:8}"></canvas>
      <span class="name">${L.name}</span><span class="note">${L.note}</span></div>
    <canvas class="field" width="${SIZE}" height="${SIZE}"></canvas>
    <div class="wfoot">
      <span>alive <b data-f="pop">—</b></span>
      <span>extra brain cells <b data-f="hidden">—</b></span>
      <span class="gone" hidden></span>
      <canvas class="mini" width="148" height="148"></canvas>
    </div>`;
  arena.appendChild(el);
  const ic = el.querySelector('canvas.ic');
  creature(ic.getContext('2d'), ic.width/2, ic.height/2+2, 15, i?1:8, 0);
  return {w, cam: cameraPath(w),
    ctx: el.querySelector('canvas.field').getContext('2d'),
    mini: el.querySelector('canvas.mini').getContext('2d'),
    f: Object.fromEntries([...el.querySelectorAll('[data-f]')].map(n=>[n.dataset.f,n])),
    gone: el.querySelector('.gone')};
});

function frameAt(w,i){ return w.frames[Math.min(i, w.frames.length-1)] || null; }

function draw(i){
  panels.forEach(p => {
    const f = frameAt(p.w, i), x = p.ctx;
    x.fillStyle = '#07100F'; x.fillRect(0,0,SIZE,SIZE);
    if (!f) return;
    const [ccx, ccy] = p.cam[Math.min(i, p.cam.length-1)];
    const ox = PAD - (ccx - VIEW/2) * CELL, oy = PAD - (ccy - VIEW/2) * CELL;
    const bin = GRID / BINS, bcell = CELL * bin;

    // food as soft overlapping blobs, not grid squares
    for (let j=0;j<f.p.length;j++){
      const v = IDX[f.p[j]]/9;
      if (v <= 0.05) continue;
      const bx = (j%BINS)*bcell + ox + bcell/2, by = Math.floor(j/BINS)*bcell + oy + bcell/2;
      if (bx < -bcell || by < -bcell || bx > SIZE+bcell || by > SIZE+bcell) continue;
      const rad = bcell * (0.42 + 0.36*v);
      const gr = x.createRadialGradient(bx, by, 0, bx, by, rad);
      gr.addColorStop(0, `rgba(52,${Math.round(120+90*v)},86,${0.36+0.3*v})`);
      gr.addColorStop(1, 'rgba(52,150,86,0)');
      x.fillStyle = gr; x.beginPath(); x.arc(bx, by, rad, 0, 6.2832); x.fill();
    }
    for (let j=0;j<f.a.length;j+=4){
      const ax=IDX[f.a[j]], ay=IDX[f.a[j+1]], d=IDX[f.a[j+2]], b=IDX[f.a[j+3]];
      const cx = ax*CELL + ox + CELL/2, cy = ay*CELL + oy + CELL/2;
      if (cx < -60 || cy < -60 || cx > SIZE+60 || cy > SIZE+60) continue;
      creature(x, cx, cy, 13 + b*2.0, d, (ax*7+ay*13+i)*0.32);
    }
    // vignette so the crowd reads as the subject
    const vg = x.createRadialGradient(SIZE/2,SIZE/2,SIZE*0.34,SIZE/2,SIZE/2,SIZE*0.72);
    vg.addColorStop(0,'rgba(7,16,15,0)'); vg.addColorStop(1,'rgba(7,16,15,.55)');
    x.fillStyle = vg; x.fillRect(0,0,SIZE,SIZE);

    // minimap: whole world, plus where the camera is
    const m = p.mini, M = 148, s = M/GRID;
    m.fillStyle = '#07100F'; m.fillRect(0,0,M,M);
    for (let j=0;j<f.p.length;j++){
      const v = IDX[f.p[j]]/9; if (v <= 0.15) continue;
      m.fillStyle = `rgba(52,140,86,${0.2+0.5*v})`;
      m.fillRect((j%BINS)*(M/BINS), Math.floor(j/BINS)*(M/BINS), M/BINS, M/BINS);
    }
    for (let j=0;j<f.a.length;j+=4){
      m.fillStyle = IDX[f.a[j+2]] >= 5 ? '#F58F52' : '#5FC9A9';
      m.fillRect(IDX[f.a[j]]*s-1, IDX[f.a[j+1]]*s-1, 2.4, 2.4);
    }
    m.strokeStyle = '#FFD277'; m.lineWidth = 1.5;
    m.strokeRect((ccx-VIEW/2)*s, (ccy-VIEW/2)*s, VIEW*s, VIEW*s);

    p.f.pop.textContent = f.s[0];
    p.f.hidden.textContent = f.s[2].toFixed(2);
    const dead = p.w.extinct !== null && i >= p.w.frames.length-1;
    p.gone.hidden = !dead;
    if (dead) p.gone.textContent = 'everyone died';
  });

  const a = frameAt(W[0], i), b = frameAt(W[1], i);
  const ha = a ? a.s[2] : 0, hb = b ? b.s[2] : 0;
  g0.style.width = (100*Math.min(1,ha/HMAX))+'%';
  g1.style.width = (100*Math.min(1,hb/HMAX))+'%';
  n0.textContent = ha.toFixed(1); n1.textContent = hb.toFixed(1);
  gen.textContent = a ? Math.round(a.s[3]) : 0;
  if (ha < 0.15 && hb < 0.15) ratio.innerHTML = 'Both sides start with completely empty brains.';
  else if (hb < 0.05) ratio.innerHTML = 'The bully side is growing brains. The nice side still has <b>none at all</b>.';
  else ratio.innerHTML = `Bully side is <b>${(ha/Math.max(hb,0.01)).toFixed(1)}× smarter</b> on average.`;
  narr.innerHTML = narrate(a ? a.t : 0, ha, hb);
}

const MS = W[0].milestones;
function narrate(t, ha, hb){
  if (t < 1500) return 'Generation one. Random creatures with no brains — they twitch around and mostly die.';
  if (MS.first_structure && t < MS.first_structure)
    return 'The hopeless ones are dying off fast. Still nobody has a real brain yet.';
  if (MS.peak_complexity && t < MS.peak_complexity)
    return 'Look at the left bar. Getting attacked is pushing them to grow <b>actual brain cells</b>. The nice side is not bothering.';
  if (MS.crash && t < MS.crash)
    return 'Peak smart. But brains burn energy every second, so they only stick around if they earn their keep.';
  if (MS.crash && t >= MS.crash)
    return 'The plants ran out and loads of them starved. Brains shrink back — being clever is rented, not owned.';
  return `Bully side: <b>${ha.toFixed(1)}</b> extra brain cells. Nice side: <b>${hb.toFixed(1)}</b>.`;
}

const FRIENDLY = {first_structure:'brains appear', peak_complexity:'peak smart',
                  crash:'food runs out'};
Object.entries(MS).sort((a,b)=>a[1]-b[1]).forEach(([k,tick]) => {
  const idx = W[0].frames.findIndex(f => f.t >= tick);
  if (idx < 0) return;
  const btn = document.createElement('button');
  btn.className='mark'; btn.style.left = (100*idx/Math.max(1,LEN-1))+'%';
  btn.innerHTML = `<i></i>${FRIENDLY[k]||k}`;
  btn.title = `Jump to "${FRIENDLY[k]||k}"`;
  btn.onclick = () => setFrame(idx);
  marks.appendChild(btn);
});

scrub.max = LEN - 1;
let cur = 0, playing = false, speed = 1, acc = 0, last = 0;
const BASE_FPS = 3.4;
function setFrame(i){ cur = Math.max(0, Math.min(LEN-1, i)); scrub.value = cur; draw(cur); }
scrub.oninput = () => setFrame(+scrub.value);
function setPlaying(v){
  playing = v;
  play.setAttribute('aria-pressed', String(v));
  play.textContent = v ? '❚❚ Pause' : '▶ Play';
  if (v){ last = performance.now(); requestAnimationFrame(step); }
}
play.onclick = () => setPlaying(!playing);
restart.onclick = () => { setFrame(0); setPlaying(true); };
document.querySelectorAll('.sp').forEach(bn => bn.onclick = () => {
  speed = +bn.dataset.s;
  document.querySelectorAll('.sp').forEach(o =>
    o.setAttribute('aria-pressed', String(o === bn)));
});
function step(now){
  if (!playing) return;
  acc += (now - last)/1000 * BASE_FPS * speed; last = now;
  if (acc >= 1){
    const adv = Math.floor(acc); acc -= adv;
    if (cur + adv >= LEN-1){ setFrame(LEN-1); setPlaying(false); return; }
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
