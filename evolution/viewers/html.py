import json
import sys
from pathlib import Path

TEMPLATE = """<!doctype html><meta charset=utf-8>
<title>Evolution replay</title>
<style>
 body{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:16px}
 canvas{image-rendering:pixelated;width:100%;max-width:900px;
        border:1px solid #333;background:#000}
 input{width:100%;max-width:900px}
</style>
<h1 style="font-size:16px">Evolution replay</h1>
<canvas id=c></canvas>
<input id=s type=range min=0 value=0>
<div id=info></div>
<script>
const FRAMES = __DATA__;
const c = document.getElementById('c'), x = c.getContext('2d');
const s = document.getElementById('s'), info = document.getElementById('info');
s.max = FRAMES.length - 1;
function draw(i){
  const f = FRAMES[i];
  c.width = f.width; c.height = f.height;
  const img = x.createImageData(f.width, f.height);
  for (let y = 0; y < f.height; y++)
    for (let X = 0; X < f.width; X++){
      const v = f.plant[y][X], p = 4 * (y * f.width + X);
      img.data[p] = 20; img.data[p+1] = 40 + 180 * v;
      img.data[p+2] = 30; img.data[p+3] = 255;
    }
  x.putImageData(img, 0, 0);
  for (const a of f.agents){
    const ax = a[0], ay = a[1], diet = a[2];
    x.fillStyle = `rgb(${Math.round(60+195*diet)},${Math.round(200-160*diet)},240)`;
    x.fillRect(ax, ay, 1, 1);
  }
  info.textContent = `tick ${f.tick} \\u2014 ${f.agents.length} agents`;
}
s.oninput = () => draw(+s.value);
draw(0);
let i = 0;
setInterval(() => { if (document.hasFocus()){ i = (i+1) % FRAMES.length;
  s.value = i; draw(i);} }, 80);
</script>
"""


def build_html(jsonl_path: str | Path, html_path: str | Path) -> None:
    frames = [
        json.loads(line)
        for line in Path(jsonl_path).read_text().splitlines()
        if line
    ]
    if not frames:
        raise ValueError(f"{jsonl_path} contains no frames")
    Path(html_path).write_text(
        TEMPLATE.replace("__DATA__", json.dumps(frames, separators=(",", ":")))
    )


if __name__ == "__main__":
    build_html(sys.argv[1], sys.argv[2])
