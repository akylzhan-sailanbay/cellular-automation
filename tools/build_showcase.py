"""Inject a showcase bundle into the replay page. No simulation logic here."""
import json
import sys
from pathlib import Path

TEMPLATE = r"""<title>Do bullies make you smarter?</title>
<style>
:root{
  --void:#0A0F12; --panel:#131D21; --panel2:#1A272C; --line:#26363C;
  --ink:#EAF2F0; --dim:#9AADAB; --safe:#56BEA0; --danger:#F0864A;
  --gold:#FFCE6B; --dead:#E06A57;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--void);color:var(--ink);font-family:var(--sans);
  padding:clamp(1rem,3vw,2rem);line-height:1.55;-webkit-font-smoothing:antialiased}
.shell{max-width:1340px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}
h1{font-size:clamp(1.7rem,5vw,2.8rem);margin:0;line-height:1.08;font-weight:750;
  letter-spacing:-.025em;text-wrap:balance}
.sub{font-size:clamp(1.02rem,2.2vw,1.2rem);color:var(--dim);margin:.7rem 0 0;max-width:60ch}
.sub b{color:var(--ink)}

.score{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:1.15rem 1.25rem 1.35rem;display:flex;flex-direction:column;gap:1rem}
.score h2{margin:0;font-size:.82rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--dim);font-weight:700}
.bar{display:grid;grid-template-columns:12rem 1fr 4.2rem;gap:1rem;align-items:center}
@media(max-width:640px){.bar{grid-template-columns:1fr;gap:.35rem}}
.who{font-weight:680;font-size:1rem;display:flex;align-items:center;gap:.55rem}
.blob{width:1.05rem;height:.9rem;border-radius:50%;flex:none;position:relative}
.blob::after{content:"";position:absolute;top:.24rem;left:.22rem;width:.16rem;
  height:.16rem;border-radius:50%;background:#0C1416;
  box-shadow:.32rem 0 0 #0C1416}
.rail{height:2.3rem;background:var(--panel2);border-radius:7px;overflow:hidden}
.grow{height:100%;width:0;border-radius:7px;transition:width .35s ease-out}
.num{font-family:var(--mono);font-size:1.6rem;font-weight:750;text-align:right;
  font-variant-numeric:tabular-nums}
.callout{font-size:1.05rem;color:var(--dim);margin:0}
.callout b{color:var(--gold)}

.say{background:var(--panel2);border:1px solid var(--line);
  border-left:4px solid var(--gold);border-radius:9px;padding:1rem 1.15rem;
  font-size:1.08rem;min-height:4.2rem;display:flex;align-items:center}
.say b{color:var(--gold)}

.arena{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}
@media(max-width:820px){.arena{grid-template-columns:1fr}}
.world{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;display:flex;flex-direction:column}
.world.danger{border-color:#5A3626}
.wtop{padding:.8rem .95rem;display:flex;align-items:center;gap:.6rem;
  border-bottom:1px solid var(--line)}
.wtop .name{font-weight:700;font-size:1.05rem}
.wtop .note{margin-left:auto;font-size:.8rem;color:var(--dim)}
canvas.field{display:block;width:100%;height:auto;background:#060B0D}
.wfoot{padding:.75rem .95rem .9rem;display:flex;gap:1.5rem;font-size:.9rem;
  color:var(--dim);flex-wrap:wrap}
.wfoot b{color:var(--ink);font-family:var(--mono);font-variant-numeric:tabular-nums}
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
details{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:.9rem 1.1rem;color:var(--dim);font-size:.95rem}
summary{cursor:pointer;color:var(--ink);font-weight:660;font-size:1rem}
summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
details p{margin:.75rem 0 0}
details b{color:var(--ink)}
.foot{color:var(--dim);font-size:.95rem;max-width:68ch;margin:0}
.foot b{color:var(--ink)}
</style>

<div class="shell">
<header>
  <h1>Do bullies make you smarter?</h1>
  <p class="sub">Two worlds of little creatures. Both start with <b>exactly the same
  dumb creatures</b>. On the left, they're allowed to attack each other. On the right,
  they aren't. Nobody taught them anything. Watch which side grows bigger brains.</p>
</header>

<div class="score">
  <h2>Brain size right now</h2>
  <div class="bar">
    <span class="who"><span class="blob" style="background:var(--danger)"></span>Left: bullies allowed</span>
    <span class="rail"><span class="grow" id="g0" style="background:var(--danger)"></span></span>
    <span class="num" id="n0" style="color:var(--danger)">0.0</span>
  </div>
  <div class="bar">
    <span class="who"><span class="blob" style="background:var(--safe)"></span>Right: everyone's nice</span>
    <span class="rail"><span class="grow" id="g1" style="background:var(--safe)"></span></span>
    <span class="num" id="n1" style="color:var(--safe)">0.0</span>
  </div>
  <p class="callout" id="ratio">Both sides start with completely empty brains.</p>
</div>

<div class="say"><span id="narr">Hit play. It runs for about two minutes — you can speed it up.</span></div>

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
    <span><span class="blob" style="display:inline-block;background:#56BEA0;vertical-align:-2px;margin-right:.4rem"></span>eats plants</span>
    <span><span class="blob" style="display:inline-block;background:#F0864A;vertical-align:-2px;margin-right:.4rem"></span>eats meat (spiky)</span>
    <span>bigger creature = bigger brain</span>
    <span>green patches = food</span>
  </div>
</div>

<details>
  <summary>What am I actually looking at?</summary>
  <p>Each little guy has a tiny brain — a network that looks at what's nearby and
  decides: move, eat, attack, or have a baby. Babies get a slightly mutated copy of
  their parent's brain. There's no score and no goal. Creatures that can't find food
  just die, and their family line ends there.</p>
  <p><b>The only difference between the two sides is whether attacking is switched
  on.</b> Same starting creatures, same food, same everything. So whatever difference
  you see, evolution made it — we didn't.</p>
  <p>The bars up top count <b>extra brain cells</b>: brain bits beyond the bare
  wiring every creature is born with. Zero means it's a pure reflex, like a knee-jerk.
  Higher means evolution has built something in the middle that can actually think a
  little.</p>
  <p>This is a recording of real runs, not a live simulation.</p>
</details>

<p class="foot"><b>One honest catch.</b> The nice world does occasionally produce a
single clever creature. The difference is whether smart <i>spreads</i>. By the end,
the average creature on the bully side has about <b>9 times</b> more extra brain cells
than the average creature on the nice side. One world makes rare geniuses. The other
makes brains normal.</p>
</div>

<script>
const DATA = __BUNDLE__;
const AB = DATA.meta.alphabet, BINS = DATA.meta.plant_bins, GRID = DATA.meta.grid;
const IDX = {}; for (let i=0;i<AB.length;i++) IDX[AB[i]] = i;
const W = DATA.worlds;
const LEN = Math.max(...W.map(w => w.frames.length));
const PX = 20, PAD = 24;                      // PAD: biggest creatures have a
const SIZE = GRID * PX + PAD * 2;             // radius of 20 and would be sliced
const world2px = v => PAD + v * PX + PX / 2;  // in half at the world edge
const HMAX = Math.max(0.5, ...W.flatMap(w => w.frames.map(f => f.s[2])));

const LABEL = [
  {name:'Bullies allowed', note:'they can attack each other', cls:'danger'},
  {name:"Everyone's nice", note:'attacking switched off',     cls:''},
];

const arena = document.getElementById('arena');
const panels = W.map((w,i) => {
  const L = LABEL[i], el = document.createElement('div');
  el.className = 'world ' + L.cls;
  el.innerHTML = `
    <div class="wtop">
      <span class="blob" style="background:${i?'var(--safe)':'var(--danger)'}"></span>
      <span class="name">${L.name}</span><span class="note">${L.note}</span></div>
    <canvas class="field" width="${SIZE}" height="${SIZE}"></canvas>
    <div class="wfoot">
      <span>creatures alive <b data-f="pop">—</b></span>
      <span>extra brain cells <b data-f="hidden">—</b></span>
      <span class="gone" hidden></span>
    </div>`;
  arena.appendChild(el);
  return {w, ctx: el.querySelector('canvas').getContext('2d'),
    f: Object.fromEntries([...el.querySelectorAll('[data-f]')].map(n=>[n.dataset.f,n])),
    gone: el.querySelector('.gone')};
});

/* ---- draw an actual little creature, not a dot ---- */
function creature(x, cx, cy, r, diet, phase){
  const meat = diet >= 5;
  const skin = meat ? 'rgb(240,134,74)'  : 'rgb(86,190,160)';
  const edge = meat ? 'rgb(148,62,30)'   : 'rgb(32,104,86)';
  const glow = meat ? 'rgba(240,134,74,.13)' : 'rgba(86,190,160,.13)';
  const bob  = Math.sin(phase) * r * 0.10;

  x.beginPath(); x.arc(cx, cy, r*1.5, 0, 6.2832);
  x.fillStyle = glow; x.fill();

  if (meat && r > 6){                       // spikes only read at larger sizes
    x.fillStyle = edge;
    for (let k=-1;k<=1;k++){
      const sx = cx + k*r*0.52;
      x.beginPath();
      x.moveTo(sx - r*0.17, cy+bob - r*0.72);
      x.lineTo(sx,          cy+bob - r*1.34);
      x.lineTo(sx + r*0.17, cy+bob - r*0.72);
      x.closePath(); x.fill();
    }
  }
  x.beginPath(); x.ellipse(cx, cy+bob, r, r*0.87, 0, 0, 6.2832);
  x.fillStyle = skin; x.fill();
  x.lineWidth = Math.max(1.2, r*0.15); x.strokeStyle = edge; x.stroke();

  if (r > 5){                                // eyes
    const ex = r*0.36, ey = -r*0.10, er = Math.max(1.5, r*0.27);
    for (const s of [-1,1]){
      x.beginPath(); x.arc(cx+s*ex, cy+bob+ey, er, 0, 6.2832);
      x.fillStyle = '#FFFFFF'; x.fill();
      x.beginPath(); x.arc(cx+s*ex+er*0.22, cy+bob+ey+er*0.1, er*0.5, 0, 6.2832);
      x.fillStyle = '#101C1A'; x.fill();
    }
  }
}

function frameAt(w,i){ return w.frames[Math.min(i, w.frames.length-1)] || null; }

function draw(i){
  panels.forEach(p => {
    const f = frameAt(p.w, i), x = p.ctx;
    x.fillStyle = '#060B0D'; x.fillRect(0,0,SIZE,SIZE);
    if (!f) return;
    const cell = (GRID * PX) / BINS;
    for (let j=0;j<f.p.length;j++){           // food, kept quiet so creatures pop
      const v = IDX[f.p[j]]/9;
      if (v <= 0.04) continue;
      x.fillStyle = `rgba(30,${Math.round(66+86*v)},62,${0.16+0.4*v})`;
      x.fillRect(PAD + (j%BINS)*cell, PAD + Math.floor(j/BINS)*cell, cell, cell);
    }
    for (let j=0;j<f.a.length;j+=4){
      const ax=IDX[f.a[j]], ay=IDX[f.a[j+1]], d=IDX[f.a[j+2]], b=IDX[f.a[j+3]];
      creature(x, world2px(ax), world2px(ay), 6.5 + b*1.5, d, (ax*7+ay*13+i)*0.35);
    }
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

const M = W[0].milestones;
function narrate(t, ha, hb){
  if (t < 1500) return 'Generation one. These are random creatures with no brains — they just twitch around and mostly die.';
  if (M.first_structure && t < M.first_structure)
    return 'The hopeless ones are dying off fast. Still nobody has a real brain yet.';
  if (M.peak_complexity && t < M.peak_complexity)
    return 'Look at the left bar. Getting attacked is pushing them to grow <b>actual brain cells</b>. The nice side is not bothering.';
  if (M.crash && t < M.crash)
    return 'Peak smart. But brains burn energy every single second, so they only stick around if they earn their keep.';
  if (M.crash && t >= M.crash)
    return 'The plants ran out and loads of them starved. Brains shrink back — being clever is rented, not owned.';
  return `Bully side: <b>${ha.toFixed(1)}</b> extra brain cells. Nice side: <b>${hb.toFixed(1)}</b>.`;
}

const FRIENDLY = {first_structure:'brains appear', peak_complexity:'peak smart',
                  crash:'food runs out'};
Object.entries(M).sort((a,b)=>a[1]-b[1]).forEach(([k,tick]) => {
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
const BASE_FPS = 3.4;                        // deliberately unhurried
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
