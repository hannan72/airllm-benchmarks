#!/usr/bin/env python3
"""article/figures/*.png: the figures of the write-up.

Each figure is an HTML/SVG page (numbers read from results/ through scripts/report.py), rendered at
2x with headless Chrome and cropped to its content. Run: python3 article/make_figures.py
"""

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report import ORDER, _best, load, load_direct  # noqa: E402

OUT = ROOT / "article" / "figures"
SHORT = {"gemma-4-E2B-it-qat-mobile-ct": "Gemma-4-E2B (QAT)", "gemma-4-E4B-it": "Gemma-4-E4B",
         "Qwen3.5-9B": "Qwen3.5-9B", "gemma-4-12B-it": "Gemma-4-12B",
         "gemma-4-26B-A4B-it": "Gemma-4-26B-A4B", "Qwen3.6-27B": "Qwen3.6-27B", "Qwen3.8-27B": "Qwen3.8-27B"}
SIZE = {"gemma-4-E2B-it-qat-mobile-ct": "2B eff.", "gemma-4-E4B-it": "4B eff.", "Qwen3.5-9B": "9.7B",
        "gemma-4-12B-it": "12B", "gemma-4-26B-A4B-it": "MoE, 4B active", "Qwen3.6-27B": "27.8B",
        "Qwen3.8-27B": "27.8B"}

# --- data -------------------------------------------------------------------------------------


def airllm(host, tuned=True):
    rows = load(ROOT / "results" / host)
    out = {}
    for m in ORDER:
        r = _best(rows, m, tuned)
        if r:
            out[m] = r["runs"]["warm"]["decode_tok_per_s"]
    return out


def direct(host):
    out = {}
    for r in load_direct(ROOT / "results" / host):
        if r.get("status") == "ok":
            v = r["runs"]["single"]["decode_tok_per_s"]
            if v > out.get(r["model"], (0, ""))[0]:
                out[r["model"]] = (v, r.get("weights", "bf16"))
    return out


def fmt(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"


# --- shared style -----------------------------------------------------------------------------

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
         "&family=IBM+Plex+Sans:wght@400;500;600;700&display=block")
# dataviz reference palette: chrome + categorical slots 1-3, sequential blue ramp
CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--rule:#ecebe6;--panel:#f3f3f0;--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a;--stock:#a8a69e;
--b100:#cde2fb;--b250:#86b6ef;--b450:#2a78d6;--b600:#184f95;--b700:#0d366b}
html,body{margin:0;background:var(--bg)}
body{font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink);-webkit-font-smoothing:antialiased}
.fig{width:1000px;box-sizing:border-box;padding:40px 48px 34px}
.eyebrow{font:600 13px/1.2 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font-size:30px;font-weight:600;line-height:1.2;margin:10px 0 8px;letter-spacing:-.01em}
.sub{font-size:17px;line-height:1.45;color:var(--ink2);max-width:880px;margin:0}
.foot{font:13px/1.5 'IBM Plex Mono',monospace;color:var(--muted);margin-top:16px}
.mono{font-family:'IBM Plex Mono',monospace}
svg text{font-family:'IBM Plex Sans',sans-serif}
svg .mono{font-family:'IBM Plex Mono',monospace}
"""


def page(body, extra_css=""):
    return (f"<!doctype html><html><head><meta charset='utf-8'><link rel='stylesheet' href='{FONTS}'>"
            f"<style>{CSS}{extra_css}</style></head><body>{body}</body></html>")


# --- figures ----------------------------------------------------------------------------------


def cover():
    rows = {h: load(ROOT / "results" / h) for h in ("local-rtx2070", "kaggle-t4")}
    q = _best(rows["local-rtx2070"], "Qwen3.8-27B", True)
    vram = q["runs"]["warm"]["peak_gpu_process_gb"]
    ups = [_best(r, m, True)["runs"]["warm"]["decode_tok_per_s"] / _best(r, m, False)["runs"]["warm"]["decode_tok_per_s"]
           for r in rows.values() for m in ORDER if _best(r, m, False) and _best(r, m, True)]
    n = q.get("n_streamed_modules", 67)
    bars, bh, gap, hi = [], 2.4, 1.4, 27
    for i in range(n):
        y = i * (bh + gap)
        bars.append(f"<rect x='0' y='{y:.1f}' width='150' height='{bh}' rx='1' "
                    f"fill='{'#ffffff' if i == hi else '#2f6db5'}'/>")
    yh = hi * (bh + gap) + bh / 2
    art = (f"<svg width='330' height='300' viewBox='0 -26 330 300'>"
           f"<text x='0' y='-10' class='mono' font-size='12' fill='#86b6ef'>Qwen3.8-27B · {n} modules · 15.1 GB</text>"
           f"{''.join(bars)}"
           f"<path d='M158,{yh:.1f} C200,{yh:.1f} 196,128 232,128' fill='none' stroke='#ffffff' stroke-width='1.6'/>"
           f"<path d='M226,123 L234,128 L226,133' fill='none' stroke='#ffffff' stroke-width='1.6'/>"
           f"<rect x='238' y='84' width='88' height='88' rx='8' fill='none' stroke='#86b6ef' stroke-width='1.6'/>"
           f"<text x='282' y='106' text-anchor='middle' class='mono' font-size='12' fill='#cde2fb'>GPU</text>"
           f"<text x='282' y='122' text-anchor='middle' class='mono' font-size='12' fill='#86b6ef'>8 GB</text>"
           f"<rect x='252' y='140' width='60' height='{bh + 1}' rx='1' fill='#ffffff'/>"
           f"<text x='282' y='196' text-anchor='middle' font-size='12.5' fill='#86b6ef'>one layer</text>"
           f"<text x='282' y='212' text-anchor='middle' font-size='12.5' fill='#86b6ef'>at a time</text></svg>")
    css = """
body{background:#0d366b}
.cover{width:960px;height:540px;box-sizing:border-box;padding:44px 48px 30px;background:#0d366b;color:#fff;
display:grid;grid-template-columns:1fr 330px;grid-template-rows:auto 1fr auto;column-gap:40px}
.cover .eyebrow{color:#86b6ef}
.cover h1{font-size:46px;line-height:1.1;margin:14px 0 14px;font-weight:600;letter-spacing:-.018em}
.cover .lede{font-size:18px;line-height:1.45;color:#cde2fb;margin:0;max-width:510px}
.art{grid-row:1 / 3;grid-column:2;align-self:start;margin-top:4px}
.stats{grid-column:1 / 3;display:grid;grid-template-columns:repeat(3,1fr);gap:28px;align-self:end;
border-top:1px solid #2f6db5;padding-top:20px}
.stat b{display:block;font:600 52px/1 'IBM Plex Mono',monospace;letter-spacing:-.03em}
.stat span{display:block;font-size:15px;line-height:1.35;color:#86b6ef;margin-top:8px}
.cover .foot{grid-column:1 / 3;color:#86b6ef;margin-top:16px}
"""
    body = f"""<div class="cover">
<div><div class="eyebrow">AirLLM 4.0 · Gemma 4 and Qwen 3.x · laptop GPUs</div>
<h1>Running a 27B LLM on an 8&nbsp;GB laptop GPU</h1>
<p class="lede">Seven open models, from 2B to 27B parameters, streamed layer by layer on two laptops and a free Kaggle T4.</p></div>
<div class="art">{art}</div>
<div class="stats">
 <div class="stat"><b>27.8B</b><span>parameters in Qwen3.8-27B</span></div>
 <div class="stat"><b>{vram:.1f} GB</b><span>peak VRAM while it runs on the 8 GB laptop GPU</span></div>
 <div class="stat"><b>{min(ups):.0f}–{max(ups):.0f}×</b><span>faster after two small fixes to AirLLM</span></div>
</div>
<div class="foot">RTX 2070 Max-Q laptop (Linux) · GTX 1070 Max-Q laptop (Windows 11) · Tesla T4 on Kaggle</div>
</div>"""
    return page(body, css), (960, 540)


def machines():
    cards = [
        ("Own laptop · Linux", "RTX 2070 Max-Q", [("GPU", "8 GB · Turing, sm_75"), ("CPU", "Core i7-9750H, 6 cores"),
         ("RAM", "32 GB"), ("Storage", "Samsung 970 EVO Plus NVMe, 3.1 GB/s"), ("Runs", "stock and tuned AirLLM")]),
        ("Own laptop · Windows 11", "GTX 1070 Max-Q", [("GPU", "8 GB · Pascal, sm_61"), ("CPU", "Core i7-8750H, 6 cores"),
         ("RAM", "32 GB"), ("Storage", "local disk"), ("Runs", "tuned AirLLM")]),
        ("Kaggle · free notebook", "Tesla T4", [("GPU", "16 GB · Turing, sm_75"), ("CPU", "Xeon 2.0 GHz, 4 vCPU"),
         ("RAM", "32 GB"), ("Storage", "VM scratch disk"), ("Runs", "stock and tuned AirLLM")]),
    ]

    def card(c):
        rows = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in c[2])
        return f"<div class='card'><div class='where'>{c[0]}</div><div class='name'>{c[1]}</div><dl>{rows}</dl></div>"

    css = """
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}
.card{border:1.5px solid var(--grid);border-radius:10px;padding:18px 18px 16px;background:#fff}
.where{font:600 12px/1.2 'IBM Plex Mono',monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--blue)}
.name{font-size:24px;font-weight:600;margin:8px 0 14px;letter-spacing:-.01em}
dl{display:grid;grid-template-columns:76px 1fr;row-gap:9px;margin:0;font-size:15.5px;line-height:1.35}
dt{font:500 12.5px/1.6 'IBM Plex Mono',monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
dd{margin:0}
.sw{margin-top:16px;font-size:15px;color:var(--ink2)}
.sw b{font-weight:600;color:var(--ink)}
"""
    body = f"""<div class="fig">
<div class="eyebrow">The test machines</div>
<h1>Two laptops and a free cloud notebook</h1>
<p class="sub">None of them can hold the larger models in VRAM, so AirLLM streams every model layer by layer.
All three have 32 GB of RAM, which turns out to matter more than the GPU.</p>
<div class="cards">{''.join(card(c) for c in cards)}</div>
<p class="sw"><b>Software:</b> AirLLM 4.0.0, bitsandbytes 0.50.2, transformers 5.20–5.23, PyTorch 2.11–2.15.
<b>Models:</b> Gemma-4-E2B (QAT), Gemma-4-E4B, Qwen3.5-9B, Gemma-4-12B, Gemma-4-26B-A4B, Qwen3.6-27B, Qwen3.8-27B.</p>
</div>"""
    return page(body, css), (1000, None)


def how_it_works():
    css = """
.lane{margin-top:22px;border:1.5px solid var(--grid);border-radius:10px;padding:18px 22px 20px;background:#fff}
.flow{display:flex;align-items:stretch}
.box{flex:1;border-radius:8px;padding:12px 14px;background:var(--panel)}
.box b{display:block;font-size:17px;font-weight:600}
.box span{display:block;font-size:14.5px;line-height:1.35;color:var(--ink2);margin-top:3px}
.box.gpu{background:var(--b100)}
.arrow{width:46px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:22px}
.loop{font:500 14px/1.3 'IBM Plex Mono',monospace;color:var(--blue);margin-top:10px;text-align:right}
table{width:100%;border-collapse:collapse;margin-top:16px;font:15px/1.4 'IBM Plex Mono',monospace}
th{font:600 11.5px/1.2 'IBM Plex Mono',monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
text-align:left;padding:0 10px 8px 0;border-bottom:1.5px solid var(--axis)}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
td.who{font-family:'IBM Plex Sans',sans-serif;font-size:15px;white-space:normal}
td.who span{display:block;color:var(--muted);font-size:13.5px}
td b{color:var(--blue);font-weight:600}
"""
    rows = [("Qwen3.8-27B", "RTX 2070 laptop, from NVMe", "15.1 GB ÷ 1.28 GB/s", "11.8 s", "12.3 s"),
            ("Qwen3.8-27B", "Kaggle T4, from page cache", "15.1 GB ÷ 3.65 GB/s", "4.1 s", "4.8 s"),
            ("Gemma-4-12B", "RTX 2070 laptop, from page cache", "6.1 GB ÷ 6.68 GB/s", "0.92 s", "1.17 s")]
    tr = "".join(f"<tr><td class='who'>{a}<span>{b}</span></td><td>{c}</td><td>= {d}</td><td><b>{e}</b></td></tr>"
                 for a, b, c, d, e in rows)
    body = f"""<div class="fig">
<div class="eyebrow">How AirLLM runs a model that does not fit</div>
<h1>Every generated token streams the whole model</h1>
<p class="sub">Only one layer is on the GPU at a time, so VRAM stays small. The price: the speed is the bytes read per token
divided by how fast the machine delivers them, from the SSD or, if the layer files fit, from the RAM page cache.</p>
<div class="lane">
 <div class="flow">
  <div class="box"><b>SSD</b><span>one file per layer, 15.1 GB for a 4‑bit 27B</span></div><div class="arrow">→</div>
  <div class="box"><b>RAM page cache</b><span>keeps the files if enough RAM is free</span></div><div class="arrow">→</div>
  <div class="box gpu"><b>GPU, one layer at a time</b><span>load, run, drop · 3.7 GB VRAM peak</span></div>
 </div>
 <div class="loop">↻ all 67 modules of Qwen3.8-27B, again for every generated token</div>
 <table><thead><tr><th>Model · machine</th><th>Bytes per token ÷ read speed</th><th>Reading</th><th>Measured per token</th></tr></thead>
 <tbody>{tr}</tbody></table>
</div>
</div>"""
    return page(body, css), (1000, None)


def per_model():
    rows = {h: load(ROOT / "results" / h) for h in ("local-rtx2070", "local-gtx1070", "kaggle-t4")}
    from report import CKPT_GB
    tr = []
    for m in ORDER:
        r = _best(rows["local-rtx2070"], m, True)
        g = _best(rows["local-gtx1070"], m, True)
        k = _best(rows["kaggle-t4"], m, True)
        p, w = r["phases"], r["runs"]["warm"]
        v, vg, vk = w["decode_tok_per_s"], g["runs"]["warm"]["decode_tok_per_s"], k["runs"]["warm"]["decode_tok_per_s"]
        mins = 200 / v / 60
        best = max(v, vg, vk)
        note = f"<span>{200 / best / 60:.0f} min with the files cached (T4)</span>" if best > 1.5 * v else ""
        files = "native QAT" if m.endswith("-ct") else "4-bit"
        tr.append(f"<tr><th>{SHORT[m]}<span>{SIZE[m]}</span></th><td>{CKPT_GB[m]:.1f} GB</td>"
                  f"<td>{p['split_size_gb']:.1f} GB<span>{files}</span></td><td>{p['convert_s'] / 60:.1f} min</td>"
                  f"<td>{w['peak_gpu_process_gb']:.1f} GB</td><td><b>{v:.2f}</b></td><td>{vg:.2f}</td>"
                  f"<td class='ans'>{mins:.0f} min{note}</td></tr>")
    css = """
table{width:100%;border-collapse:collapse;margin-top:22px;font:16px/1.3 'IBM Plex Mono',monospace;table-layout:fixed}
col.c1{width:196px}col.c8{width:136px}
thead th{font:600 11.5px/1.3 'IBM Plex Mono',monospace;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
text-align:left;padding:0 10px 10px 0;border-bottom:1.5px solid var(--axis);vertical-align:bottom}
tbody th{text-align:left;font:600 16px/1.3 'IBM Plex Sans',sans-serif;padding:10px 10px 10px 0;vertical-align:top}
tbody th span,td span{display:block;font:400 12.5px/1.3 'IBM Plex Sans',sans-serif;color:var(--muted)}
td{padding:10px 10px 10px 0;border-bottom:1px solid var(--rule);vertical-align:top}
tbody th{border-bottom:1px solid var(--rule)}
td b{font-weight:600;color:var(--blue)}
td.ans{font-weight:600}
"""
    body = f"""<div class="fig">
<div class="eyebrow">Practical numbers · tuned AirLLM</div>
<h1>What each model takes on an 8 GB laptop</h1>
<p class="sub">Download = the original checkpoint. Layer files = what AirLLM converts it into, once. To read them at
page-cache speed, about that much RAM has to be free. Speed and answer time: warm runs on the laptops.</p>
<table><colgroup><col class="c1"><col><col><col><col><col><col><col class="c8"></colgroup>
<thead><tr><th>Model</th><th>Download</th><th>Layer files</th><th>One-time convert</th><th>Peak VRAM</th>
<th>Tok/s RTX 2070</th><th>Tok/s GTX 1070</th><th>200-token answer, RTX 2070</th></tr></thead>
<tbody>{''.join(tr)}</tbody></table>
<div class="foot">Convert times on the RTX 2070 laptop from a local checkpoint. Gemma-4-26B-A4B uses the lazy splitter (its
checkpoint is a single 49.9 GB file).</div>
</div>"""
    return page(body, css), (1000, None)


def tuning():
    host = "local-rtx2070"
    stock, tuned = airllm(host, tuned=False), airllm(host, tuned=True)
    W, left, right, xmax = 904, 196, 712, 1.75
    X = lambda v: left + v / xmax * (right - left)
    rh, top = 50, 34
    H = top + rh * len(ORDER) + 30
    g = []
    for t in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        g.append(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 8}' y2='{H - 26}' stroke='var(--grid)' "
                 f"stroke-width='{1.4 if t == 0 else 1}'/>"
                 f"<text x='{X(t):.1f}' y='{H - 6}' text-anchor='middle' class='mono' font-size='13' "
                 f"fill='var(--muted)'>{t:g}</text>")
    g.append(f"<text x='{right + 18}' y='{top - 14}' class='mono' font-size='12' fill='var(--muted)' "
             f"letter-spacing='.06em'>STOCK → TUNED</text>")
    for i, m in enumerate(ORDER):
        y = top + rh * i + rh / 2
        s, t = stock[m], tuned[m]
        g.append(f"<text x='0' y='{y - 2:.1f}' font-size='16' font-weight='600' fill='var(--ink)'>{SHORT[m]}</text>"
                 f"<text x='0' y='{y + 16:.1f}' font-size='13' fill='var(--muted)'>{SIZE[m]}</text>")
        g.append(f"<line x1='{X(s):.1f}' x2='{X(t):.1f}' y1='{y:.1f}' y2='{y:.1f}' stroke='var(--axis)' stroke-width='2.5'/>"
                 f"<circle cx='{X(s):.1f}' cy='{y:.1f}' r='6.5' fill='var(--stock)' stroke='var(--bg)' stroke-width='2'/>"
                 f"<circle cx='{X(t):.1f}' cy='{y:.1f}' r='8' fill='var(--blue)' stroke='var(--bg)' stroke-width='2'/>")
        g.append(f"<text x='{right + 18}' y='{y + 5:.1f}' class='mono' font-size='14' fill='var(--ink2)'>"
                 f"{s:.2f} → {t:.2f}</text>"
                 f"<text x='{W}' y='{y + 6:.1f}' text-anchor='end' class='mono' font-size='19' font-weight='600' "
                 f"fill='var(--ink)'>×{t / s:.0f}</text>")
    q = top + rh * 5
    g.append(f"<text x='{X(0.16):.1f}' y='{q + 22:.1f}' font-size='14' fill='var(--ink2)'>"
             f"<tspan x='{X(0.16):.1f}'>15 GB of layer files did not fit in free RAM,</tspan>"
             f"<tspan x='{X(0.16):.1f}' dy='19'>so every token came from the NVMe at 1.3 GB/s</tspan></text>")
    svg = f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>{''.join(g)}</svg>"
    legend = ("<div class='leg'><span><i style='background:var(--stock)'></i>Stock AirLLM 4.0, fastest config with "
              "correct output</span><span><i style='background:var(--blue)'></i>Tuned: NF4 on weight matrices only "
              "+ no per-layer gc.collect()</span></div>")
    css = """.leg{display:flex;gap:26px;flex-wrap:wrap;font-size:15px;color:var(--ink2);margin:20px 0 6px}
.leg i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;vertical-align:-1px}"""
    body = f"""<div class="fig">
<div class="eyebrow">AirLLM tuning · RTX 2070 Max-Q laptop, 8 GB</div>
<h1>Two small changes made every model 2–15× faster</h1>
<p class="sub">Decode tokens per second, one request, warm run.</p>
{legend}{svg}
<div class="foot">x axis: decode tokens / second · Gemma-4-E2B runs its shipped QAT weights, so only the gc change applies.</div>
</div>"""
    return page(body, css), (1000, None)


def small_gpus():
    series = [("local-gtx1070", "GTX 1070 Max-Q · own laptop, Windows", "var(--aqua)"),
              ("local-rtx2070", "RTX 2070 Max-Q · own laptop, Linux", "var(--blue)"),
              ("kaggle-t4", "Tesla T4 · Kaggle", "var(--orange)")]
    data = {h: airllm(h) for h, _, _ in series}
    W, left, right, xmax = 904, 196, 800, 1.6
    X = lambda v: left + v / xmax * (right - left)
    bh, gap, rh, top = 12, 2, 62, 14
    H = top + rh * len(ORDER) + 28
    g = []
    for t in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        g.append(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 6}' y2='{H - 26}' stroke='var(--grid)' "
                 f"stroke-width='{1.4 if t == 0 else 1}'/>"
                 f"<text x='{X(t):.1f}' y='{H - 6}' text-anchor='middle' class='mono' font-size='13' "
                 f"fill='var(--muted)'>{t:g}</text>")
    for i, m in enumerate(ORDER):
        y0 = top + rh * i + (rh - 3 * bh - 2 * gap) / 2
        g.append(f"<text x='0' y='{y0 + 17:.1f}' font-size='16' font-weight='600' fill='var(--ink)'>{SHORT[m]}</text>"
                 f"<text x='0' y='{y0 + 35:.1f}' font-size='13' fill='var(--muted)'>{SIZE[m]}</text>")
        for j, (h, _, c) in enumerate(series):
            v = data[h][m]
            y = y0 + j * (bh + gap)
            w = max(X(v) - X(0), 2)
            g.append(f"<path d='M{X(0):.1f},{y:.1f} h{w - 3:.1f} a3,3 0 0 1 3,3 v{bh - 6} a3,3 0 0 1 -3,3 h{-(w - 3):.1f} z' "
                     f"fill='{c}'/>"
                     f"<text x='{X(v) + 6:.1f}' y='{y + bh - 1.5:.1f}' class='mono' font-size='12.5' "
                     f"fill='var(--ink2)'>{v:.2f}</text>")
    q = top + rh * 5
    g.append(f"<text font-size='14' fill='var(--ink2)'>"
             f"<tspan x='{X(0.42):.1f}' y='{q + 22:.1f}'>27B: the order flips. Their 15 GB of 4-bit layer files fit in</tspan>"
             f"<tspan x='{X(0.42):.1f}' dy='19'>the free RAM of Kaggle's VM. On the RTX 2070 laptop they did</tspan>"
             f"<tspan x='{X(0.42):.1f}' dy='19'>not, so every token reads them from the NVMe at 1.3 GB/s.</tspan></text>")
    svg = f"<svg width='{W}' height='{H}' viewBox='0 0 {W} {H}'>{''.join(g)}</svg>"
    legend = "<div class='leg'>" + "".join(
        f"<span><i style='background:{c}'></i>{lab}</span>" for _, lab, c in series) + "</div>"
    css = """.leg{display:flex;gap:24px;flex-wrap:wrap;font-size:15px;color:var(--ink2);margin:20px 0 10px}
.leg i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:8px;vertical-align:-1px}"""
    body = f"""<div class="fig">
<div class="eyebrow">Tuned AirLLM on three small GPUs</div>
<h1>The fastest laptop falls behind on the 27B models</h1>
<p class="sub">Decode tokens per second, one request, warm run, tuned configuration (NF4 on matrices + no per-layer gc).</p>
{legend}{svg}
<div class="foot">x axis: decode tokens / second</div>
</div>"""
    return page(body, css), (1000, None)


def matrix():
    cols = [("local-gtx1070", "GTX 1070", "own laptop", "AirLLM", "a"),
            ("local-rtx2070", "RTX 2070", "own laptop", "AirLLM", "a"),
            ("kaggle-t4", "T4", "Kaggle", "AirLLM", "a"),
            ("zerogpu-rtxpro6000", "RTX PRO 6000", "Hugging Face", "transformers", "d"),
            ("modal-a100-40", "A100 40 GB", "Modal", "vLLM", "d"),
            ("modal-a100-80", "A100 80 GB", "Modal", "vLLM", "d"),
            ("modal-h100", "H100", "Modal", "vLLM", "d")]
    AIR = {h for h, *_, kind in cols if kind == "a"}
    vals = {}
    for h, *_, kind in cols:
        if kind == "a":
            for m, v in airllm(h).items():
                vals[(m, h)] = (v, "")
        else:
            for m, (v, w) in direct(h).items():
                vals[(m, h)] = (v, w if w != "bf16" else "")
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
    lo, hi = math.log10(0.05), math.log10(300)

    def color(v):
        f = (math.log10(v) - lo) / (hi - lo)
        k = min(len(ramp) - 1, max(0, round(f * (len(ramp) - 1))))
        return ramp[k], ("#ffffff" if k >= 6 else "#0b0b0b")

    head = "".join(("<th class='sp'></th>" if i == 3 else "") + f"<th><b>{n}</b><span>{p}</span><em>{e}</em></th>"
                   for i, (_, n, p, e, _) in enumerate(cols))
    body_rows = []
    for m in ORDER:
        cells = []
        for i, (h, *_rest) in enumerate(cols):
            cls = ""
            if i == 3:
                cells.append("<td class='sp'></td>")
            if (m, h) not in vals:
                cells.append(f"<td class='{cls} na'>fails to load</td>")
                continue
            v, tag = vals[(m, h)]
            bg, fg = color(v)
            t = f"<small>{tag}</small>" if tag else ""
            cells.append(f"<td class='{cls}' style='background:{bg};color:{fg}'>{fmt(v)}{t}</td>")
        body_rows.append(f"<tr><th class='row'>{SHORT[m]}<span>{SIZE[m]}</span></th>{''.join(cells)}</tr>")
    best_a = {m: max(v for (mm, h), (v, _) in vals.items() if mm == m and h in AIR) for m in ORDER}
    ratios = [v / best_a[m] for (m, h), (v, _) in vals.items() if h not in AIR]
    lo_r, hi_r = min(ratios), max(ratios)
    stops = ",".join(f"{c} {i / (len(ramp) - 1) * 100:.0f}%" for i, c in enumerate(ramp))
    ticks = "".join(f"<span style='left:{(math.log10(t) - lo) / (hi - lo) * 100:.1f}%'>{t:g}</span>"
                    for t in (0.1, 1, 10, 100))
    css = f"""
table{{border-collapse:separate;border-spacing:3px;margin:18px -3px 0;width:calc(100% + 6px);table-layout:fixed}}
th.grp{{font:600 12px/1.2 'IBM Plex Mono',monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--blue);
text-align:left;padding:0 0 7px;border-bottom:1.5px solid var(--b250)}}
.sp{{background:none!important}}
thead th{{font-weight:400;text-align:center;vertical-align:bottom;padding:10px 0 8px;line-height:1.25}}
thead th b{{display:block;font-size:14px;font-weight:600;white-space:nowrap}}
thead th span{{display:block;font-size:13.5px;color:var(--ink2)}}
thead th em{{display:block;font:500 12px/1.3 'IBM Plex Mono',monospace;color:var(--muted);font-style:normal}}
th.corner{{width:166px}}
th.row{{text-align:left;font-size:15px;font-weight:600;padding-right:6px}}
th.row span{{display:block;font-size:12.5px;font-weight:400;color:var(--muted)}}
td{{height:46px;text-align:center;font:600 17px/1 'IBM Plex Mono',monospace;border-radius:5px}}
td small{{display:block;font:500 11px/1.2 'IBM Plex Mono',monospace;opacity:.85;margin-top:2px}}
td.na{{background:var(--panel);color:var(--muted);font:500 12px/1.2 'IBM Plex Sans',sans-serif}}
.scale{{display:flex;align-items:center;gap:14px;margin-top:20px;font-size:14px;color:var(--ink2)}}
.bar{{position:relative;flex:0 0 360px;height:12px;border-radius:6px;background:linear-gradient(90deg,{stops})}}
.bar span{{position:absolute;top:16px;transform:translateX(-50%);font:12px 'IBM Plex Mono',monospace;color:var(--muted)}}
"""
    body = f"""<div class="fig">
<div class="eyebrow">For perspective · decode tokens per second, one request</div>
<h1>A GPU that holds the model is {lo_r:.0f}–{hi_r:.0f}× faster</h1>
<p class="sub">Small GPUs: tuned AirLLM, warm run. Big GPUs: whole model on one GPU; vLLM is the median of 3 runs of
256 tokens, transformers of 2 runs of 64.</p>
<table><colgroup><col style="width:166px"><col><col><col><col style="width:10px"><col><col><col><col></colgroup>
<thead><tr><th></th><th class="grp" colspan="3">Streamed · AirLLM</th><th class="sp"></th>
<th class="grp" colspan="4">Whole model on the GPU</th></tr>
<tr><th class="corner"></th>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>
<div class="scale"><span>tokens / second</span><div class="bar">{ticks}</div><span style="margin-left:8px">log scale</span></div>
<div class="foot" style="margin-top:30px">AirLLM runs 4-bit weights; the big GPUs run 16-bit (bf16), or 8-bit (fp8) where marked, so the 26–27B models
fit the 40 GB A100. Gemma-4-E2B's QAT checkpoint does not load in transformers 5.23.</div>
</div>"""
    return page(body, css), (1000, None)


# --- render -----------------------------------------------------------------------------------

FIGS = {"cover": cover, "fig1-machines": machines, "fig2-how-it-works": how_it_works, "fig3-tuning": tuning,
        "fig4-small-gpus": small_gpus, "fig5-per-model": per_model, "fig6-perspective": matrix}


def render(name, html, size, tmp):
    src = tmp / f"{name}.html"
    src.write_text(html)
    w, h = size
    png = OUT / f"{name}.png"
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                    f"--user-data-dir={tmp / 'profile'}", "--force-device-scale-factor=2",
                    f"--window-size={w},{h or 1600}", "--virtual-time-budget=15000",
                    f"--screenshot={png}", src.as_uri()], check=True, capture_output=True)
    if h is None:  # crop the empty bottom
        im = Image.open(png).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((2, im.height - 2)))
        box = ImageChops.difference(im, bg).getbbox()
        if box:
            im.crop((0, 0, im.width, min(im.height, box[3] + 64))).save(png, optimize=True)
    return png


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, fn in FIGS.items():
            if only and name not in only:
                continue
            html, size = fn()
            (OUT / "src").mkdir(exist_ok=True)
            (OUT / "src" / f"{name}.html").write_text(html)
            print("wrote", render(name, html, size, tmp))


if __name__ == "__main__":
    main()
