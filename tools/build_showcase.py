"""Inject a showcase bundle into the replay page. No simulation logic here."""
import json
import sys
from pathlib import Path

TEMPLATE = r"""<title>Does danger make you smarter?</title>
<style>
:root{
  --void:#0A0F12; --panel:#121B1F; --panel2:#172328; --line:#243238;
  --ink:#E8F0EE; --dim:#93A6A4; --faint:#5B6E6D;
  --safe:#4FB79C; --danger:#F0864A; --gold:#FFC961; --dead:#D9604F;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--void);color:var(--ink);font-family:var(--sans);
  padding:clamp(1rem,3vw,2rem);line-height:1.55;-webkit-font-smoothing:antialiased}
.shell{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:1.5rem}

h1{font-size:clamp(1.6rem,5vw,2.6rem);margin:0;line-height:1.1;font-weight:700;
  letter-spacing:-.02em;text-wrap:balance}
.sub{font-size:clamp(1rem,2.4vw,1.15rem);color:var(--dim);margin:.6rem 0 0;max-width:62ch}
.sub b{color:var(--ink)}

/* ---------- the scoreboard: the whole point, up top and huge ---------- */
.score{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:1.1rem 1.2rem 1.3rem;display:flex;flex-direction:column;gap:.9rem}
.score h2{margin:0;font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim);font-weight:700}
.bars{display:flex;flex-direction:column;gap:.85rem}
.bar{display:grid;grid-template-columns:11rem 1fr auto;gap:.9rem;align-items:center}
@media(max-width:620px){.bar{grid-template-columns:1fr;gap:.3rem}}
.bar .who{font-weight:650;font-size:.98rem;display:flex;align-items:center;gap:.5rem}
.dot{width:.7rem;height:.7rem;border-radius:50%;flex:none}
.rail{height:2.1rem;background:var(--panel2);border-radius:6px;overflow:hidden;
  position:relative}
.grow{height:100%;width:0;border-radius:6px;transition:width .15s linear}
.bar .num{font-family:var(--mono);font-size:1.5rem;font-weight:700;
  font-variant-numeric:tabular-nums;min-width:3.6rem;text-align:right}
.verdictline{font-size:1rem;color:var(--dim);margin:0}
.verdictline b{color:var(--gold)}

/* ---------- narration ---------- */
.say{background:linear-gradient(180deg,var(--panel2),var(--panel));
  border:1px solid var(--line);border-left:3px solid var(--gold);
  border-radius:8px;padding:.9rem 1.1rem;font-size:1.02rem;min-height:3.9rem;
  display:flex;align-items:center}
.say span{margin:0}
.say b{color:var(--gold)}

/* ---------- worlds ---------- */
.arena{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:700px){.arena{grid-template-columns:1fr}}
.world{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  overflow:hidden;display:flex;flex-direction:column}
.world.danger{border-color:#4A2E22}
.wtop{padding:.75rem .9rem;display:flex;align-items:center;gap:.55rem;
  border-bottom:1px solid var(--line)}
.wtop .name{font-weight:700;font-size:1rem}
.wtop .note{margin-left:auto;font-size:.75rem;color:var(--dim)}
canvas.field{display:block;width:100%;height:auto;background:#050A0C}
.wfoot{padding:.7rem .9rem .85rem;display:flex;gap:1.4rem;font-size:.85rem;
  color:var(--dim);flex-wrap:wrap}
.wfoot b{color:var(--ink);font-family:var(--mono);font-variant-numeric:tabular-nums}
.gone{color:var(--dead);font-weight:700}

/* ---------- controls ---------- */
.bay{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:.9rem 1rem;display:flex;flex-direction:column;gap:.7rem}
.tl{position:relative;height:2.6rem}
.tl input{position:absolute;inset:auto 0 .1rem;width:100%;margin:0;
  accent-color:var(--gold);height:1.1rem}
.marks{position:absolute;inset:0 0 auto;height:1.5rem;pointer-events:none}
.mark{position:absolute;top:0;transform:translateX(-50%);pointer-events:auto;
  background:none;border:0;padding:.1rem .2rem;cursor:pointer;color:var(--dim);
  font-size:.7rem;white-space:nowrap;display:flex;flex-direction:column;
  align-items:center;gap:.1rem;font-family:var(--sans)}
.mark i{width:2px;height:.55rem;background:currentColor;border-radius:1px}
.mark:hover,.mark:focus-visible{color:var(--gold);outline:none}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
button{font-family:var(--sans);font-size:.88rem;font-weight:600;
  background:var(--panel2);color:var(--ink);border:1px solid var(--line);
  border-radius:7px;padding:.5rem .9rem;cursor:pointer}
button:hover{border-color:var(--dim)}
button[aria-pressed="true"]{background:var(--gold);color:#12191B;border-color:var(--gold)}
button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
#play{min-width:6.5rem}
.time{margin-left:auto;color:var(--dim);font-size:.85rem}
.time b{color:var(--ink);font-family:var(--mono);font-variant-numeric:tabular-nums}

.key{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;font-size:.85rem;color:var(--dim);
  align-items:center;border-top:1px solid var(--line);padding-top:.75rem}
.key .sw{display:inline-block;border-radius:50%;margin-right:.4rem;vertical-align:-2px}
.foot{color:var(--dim);font-size:.9rem;max-width:70ch;margin:0}
.foot b{color:var(--ink)}
details{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:.8rem 1rem;color:var(--dim);font-size:.9rem}
summary{cursor:pointer;color:var(--ink);font-weight:600;font-size:.95rem}
summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
details p{margin:.7rem 0 0}
details b{color:var(--ink)}
</style>

<div class="shell">
<header>
  <h1>Does danger make you smarter?</h1>
  <p class="sub">Two worlds of little creatures, started from <b>exactly the same
  random creatures</b>. On the left they can eat each other. On the right they cannot.
  Nobody programmed them to be clever — watch which side grows bigger brains.</p>
</header>

<div class="score">
  <h2>Brain size right now</h2>
  <div class="bars">
    <div class="bar">
      <span class="who"><span class="dot" style="background:var(--danger)"></span>Dangerous world</span>
      <span class="rail"><span class="grow" id="g0" style="background:var(--danger)"></span></span>
      <span class="num" id="n0" style="color:var(--danger)">0.0</span>
    </div>
    <div class="bar">
      <span class="who"><span class="dot" style="background:var(--safe)"></span>Safe world</span>
      <span class="rail"><span class="grow" id="g1" style="background:var(--safe)"></span></span>
      <span class="num" id="n1" style="color:var(--safe)">0.0</span>
    </div>
  </div>
  <p class="verdictline" id="ratio">Both start with identical, completely empty brains.</p>
</div>

<div class="say"><span id="narr">Press play. It takes about 30 seconds.</span></div>

<div class="arena" id="arena"></div>

<div class="bay">
  <div class="tl">
    <div class="marks" id="marks"></div>
    <input id="scrub" type="range" min="0" value="0" step="1" aria-label="Timeline">
  </div>
  <div class="row">
    <button id="play" aria-pressed="false">▶ Play</button>
    <button class="sp" data-s="1" aria-pressed="true">Normal</button>
    <button class="sp" data-s="4" aria-pressed="false">Fast</button>
    <button class="sp" data-s="16" aria-pressed="false">Very fast</button>
    <button id="restart">↺ Start over</button>
    <span class="time">generation <b id="gen">0</b></span>
  </div>
  <div class="key">
    <span><span class="sw" style="width:.55rem;height:.55rem;background:#4FB79C"></span>plant-eater</span>
    <span><span class="sw" style="width:.55rem;height:.55rem;background:#F0864A"></span>meat-eater</span>
    <span><span class="sw" style="width:1rem;height:1rem;background:#8FA8A0"></span>bigger dot = bigger brain</span>
    <span>green background = food growing</span>
  </div>
</div>

<details>
  <summary>Wait, what am I actually looking at?</summary>
  <p>Each dot is a creature with a tiny brain — a little network that takes in what it
  can see and decides whether to move, eat, attack, or have a baby. When a creature
  breeds, its baby gets a slightly mutated copy of the parent brain. There is no score
  and no goal. Creatures that fail to eat simply die, and their line ends.</p>
  <p><b>The only difference between the two worlds is whether attacking is switched
  on.</b> Same starting creatures, same food, same everything else. So any difference
  you see was produced by evolution, not by us.</p>
  <p>The bars at the top count <b>extra brain cells</b> — neurons beyond the bare
  input-to-output wiring every creature starts with. Zero means the brain is a simple
  reflex. Higher means evolution has built something in the middle that can actually
  process information.</p>
  <p>This is a recording of real runs, not a live simulation. Every frame came out of
  the simulator, which has 89 tests covering it.</p>
</details>

<p class="foot"><b>The catch.</b> Occasionally the safe world produces one clever
creature too. The difference is whether cleverness <i>spreads</i>: by the end, the
average creature in the dangerous world has around <b>9 times</b> more extra brain
cells than the average creature in the safe world. One world makes rare geniuses;
the other makes brains standard.</p>
</div>

<script>
const DATA = __BUNDLE__;
const AB = DATA.meta.alphabet, BINS = DATA.meta.plant_bins, GRID = DATA.meta.grid;
const IDX = {}; for (let i=0;i<AB.length;i++) IDX[AB[i]] = i;
const W = DATA.worlds;
const LEN = Math.max(...W.map(w => w.frames.length));
const PX = 11, SIZE = GRID * PX;
const HMAX = Math.max(0.5, ...W.flatMap(w => w.frames.map(f => f.s[2])));

function lerp(a,b,t){return a+(b-a)*t}
function dietColor(d,a){
  const t = d/9;
  return `rgba(${Math.round(lerp(79,240,t))},${Math.round(lerp(183,134,t))},${Math.round(lerp(156,74,t))},${a})`;
}

const LABEL = [
  {name:'Dangerous world', note:'they can eat each other', cls:'danger'},
  {name:'Safe world',      note:'no attacking allowed',   cls:''},
];

const arena = document.getElementById('arena');
const panels = W.map((w,i) => {
  const L = LABEL[i];
  const el = document.createElement('div');
  el.className = 'world ' + L.cls;
  el.innerHTML = `
    <div class="wtop">
      <span class="dot" style="background:${i?'var(--safe)':'var(--danger)'}"></span>
      <span class="name">${L.name}</span><span class="note">${L.note}</span></div>
    <canvas class="field" width="${SIZE}" height="${SIZE}"></canvas>
    <div class="wfoot">
      <span>creatures <b data-f="pop">—</b></span>
      <span>extra brain cells <b data-f="hidden">—</b></span>
      <span class="gone" hidden></span>
    </div>`;
  arena.appendChild(el);
  return {w, ctx: el.querySelector('canvas').getContext('2d'),
          f: Object.fromEntries([...el.querySelectorAll('[data-f]')].map(n=>[n.dataset.f,n])),
          gone: el.querySelector('.gone')};
});

function frameAt(w,i){ return w.frames[Math.min(i, w.frames.length-1)] || null; }

function draw(i){
  panels.forEach((p,k) => {
    const f = frameAt(p.w, i), x = p.ctx;
    x.fillStyle = '#050A0C'; x.fillRect(0,0,SIZE,SIZE);
    if (!f) return;
    const cell = SIZE / BINS;
    for (let j=0;j<f.p.length;j++){
      const v = IDX[f.p[j]] / 9;
      if (v <= 0.02) continue;
      x.fillStyle = `rgba(38,${Math.round(78+120*v)},76,${0.2+0.6*v})`;
      x.fillRect((j % BINS)*cell, Math.floor(j/BINS)*cell, cell, cell);
    }
    for (let j=0;j<f.a.length;j+=4){
      const ax=IDX[f.a[j]], ay=IDX[f.a[j+1]], d=IDX[f.a[j+2]], b=IDX[f.a[j+3]];
      const cx = ax*PX+PX/2, cy = ay*PX+PX/2, r = 2.8 + b*0.62;
      x.beginPath(); x.arc(cx,cy,r+2.2,0,6.2832);
      x.fillStyle = dietColor(d,0.16); x.fill();
      x.beginPath(); x.arc(cx,cy,r,0,6.2832);
      x.fillStyle = dietColor(d,1); x.fill();
    }
    p.f.pop.textContent = f.s[0];
    p.f.hidden.textContent = f.s[2].toFixed(2);
    const dead = p.w.extinct !== null && i >= p.w.frames.length-1;
    p.gone.hidden = !dead;
    if (dead) p.gone.textContent = 'everyone died';
  });

  const a = frameAt(W[0], i), b = frameAt(W[1], i);
  const ha = a ? a.s[2] : 0, hb = b ? b.s[2] : 0;
  document.getElementById('g0').style.width = (100*Math.min(1,ha/HMAX))+'%';
  document.getElementById('g1').style.width = (100*Math.min(1,hb/HMAX))+'%';
  document.getElementById('n0').textContent = ha.toFixed(1);
  document.getElementById('n1').textContent = hb.toFixed(1);
  document.getElementById('gen').textContent = a ? Math.round(a.s[3]) : 0;

  const rl = document.getElementById('ratio');
  if (ha < 0.15 && hb < 0.15) rl.innerHTML = 'Both start with identical, completely empty brains.';
  else if (hb < 0.05) rl.innerHTML = `The dangerous world is building brains. The safe world still has <b>none</b>.`;
  else rl.innerHTML = `Dangerous world is <b>${(ha/Math.max(hb,0.01)).toFixed(1)}×</b> ahead.`;
  document.getElementById('narr').innerHTML = narrate(a ? a.t : 0, ha, hb);
}

const M = W[0].milestones;
function narrate(t, ha, hb){
  if (t < 1200) return 'Generation one. Every creature is a random reflex — no thinking, just twitching toward food.';
  if (M.first_structure && t < M.first_structure)
    return 'The clumsy ones are dying off. Brains are still empty on both sides.';
  if (M.peak_complexity && t < M.peak_complexity)
    return `Something is happening on the left. Hunted creatures are growing <b>real brain cells</b> — the safe ones are not bothering.`;
  if (M.crash && t < M.crash)
    return `Peak cleverness on the left. Bigger brains cost energy every second, so they only survive if they pay for themselves.`;
  if (M.crash && t >= M.crash)
    return `The plants ran out and the population crashed. Brains shrink back — cleverness is rented, not owned.`;
  return `Dangerous world: <b>${ha.toFixed(1)}</b> extra brain cells. Safe world: <b>${hb.toFixed(1)}</b>.`;
}

const FRIENDLY = {first_structure:'brains appear', peak_complexity:'peak cleverness',
                  crash:'food runs out'};
const marks = document.getElementById('marks');
Object.entries(M).sort((a,b)=>a[1]-b[1]).forEach(([k,tick]) => {
  const idx = W[0].frames.findIndex(f => f.t >= tick);
  if (idx < 0) return;
  const btn = document.createElement('button');
  btn.className = 'mark'; btn.style.left = (100*idx/Math.max(1,LEN-1))+'%';
  btn.innerHTML = `<i></i>${FRIENDLY[k]||k}`;
  btn.title = `Jump to "${FRIENDLY[k]||k}" (dangerous world)`;
  btn.onclick = () => setFrame(idx);
  marks.appendChild(btn);
});

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
document.querySelectorAll('.sp').forEach(bn => bn.onclick = () => {
  speed = +bn.dataset.s;
  document.querySelectorAll('.sp').forEach(o =>
    o.setAttribute('aria-pressed', String(o === bn)));
});
function step(now){
  if (!playing) return;
  acc += (now - last)/1000 * 13 * speed; last = now;
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
