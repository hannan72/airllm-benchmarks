#!/usr/bin/env python3
"""article/carousel.pdf: a LinkedIn document post (carousel) with the article's findings.

Eight 1080x1350 slides with large type, so they stay readable on a phone: cover, six findings
(each a simplified version of one article figure, numbers read from results/), closing links.
Run: python article/make_carousel.py
"""

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ART))
from make_figures import airllm, direct  # noqa: E402

OUT = ART / "carousel.pdf"
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
         "&family=IBM+Plex+Sans:wght@400;500;600;700&display=block")
AUTHOR = "Hannan Komari · AirLLM on laptop GPUs"
N = 8
SHORT = {"gemma-4-E2B-it-qat-mobile-ct": "Gemma-4-E2B", "gemma-4-E4B-it": "Gemma-4-E4B", "Qwen3.5-9B": "Qwen3.5-9B",
         "gemma-4-12B-it": "Gemma-4-12B", "gemma-4-26B-A4B-it": "Gemma-4-26B-A4B", "Qwen3.6-27B": "Qwen3.6-27B",
         "Qwen3.8-27B": "Qwen3.8-27B"}
BLUE, ORANGE, AQUA, INK2, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#898781"

CSS = f"""
@page {{ size: 1080px 1350px; margin: 0 }}
* {{ box-sizing: border-box }}
html, body {{ margin: 0; background: #fcfcfb; -webkit-print-color-adjust: exact; print-color-adjust: exact }}
body {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; color: #0b0b0b }}
.slide {{ width: 1080px; height: 1350px; padding: 76px 72px 52px; display: flex; flex-direction: column;
  page-break-after: always; break-after: page; overflow: hidden }}
.eyebrow {{ font: 600 24px/1.2 'IBM Plex Mono', monospace; letter-spacing: .06em; text-transform: uppercase; color: {BLUE} }}
h1 {{ font-size: 64px; line-height: 1.07; font-weight: 600; letter-spacing: -.02em; margin: 18px 0 0; text-wrap: balance }}
.body {{ flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 26px; margin: 40px 0 }}
.take {{ font-size: 36px; line-height: 1.3; font-weight: 500; margin: 0; text-wrap: pretty }}
.take code, .fix code {{ font: 500 0.88em 'IBM Plex Mono', monospace; background: #f0efec; padding: 0 8px; border-radius: 6px }}
.foot {{ display: flex; justify-content: space-between; margin-top: 34px; padding-top: 18px; border-top: 2px solid #ecebe6;
  font: 500 20px 'IBM Plex Mono', monospace; color: {MUTED} }}
.label {{ font: 600 22px/1.2 'IBM Plex Mono', monospace; letter-spacing: .05em; text-transform: uppercase; color: {MUTED} }}
.mono {{ font-family: 'IBM Plex Mono', monospace }}
/* machine cards */
.card {{ border: 2px solid #e1e0d9; border-radius: 16px; padding: 26px 30px; background: #fff }}
.card .who {{ font: 600 22px 'IBM Plex Mono', monospace; letter-spacing: .05em; text-transform: uppercase; color: {BLUE} }}
.card .name {{ font-size: 50px; font-weight: 600; margin: 6px 0 14px; letter-spacing: -.01em }}
.chips {{ display: flex; flex-wrap: wrap; gap: 12px }}
.chips span {{ font-size: 27px; background: #f3f3f0; border-radius: 999px; padding: 6px 18px }}
.chips span.hi {{ background: #cde2fb; font-weight: 600 }}
/* flow */
.flow {{ display: grid; grid-template-columns: 1fr 44px 1fr 44px 1fr; align-items: stretch }}
.box {{ border-radius: 14px; background: #f3f3f0; padding: 20px 20px; font-size: 27px; line-height: 1.25 }}
.box b {{ display: block; font-size: 32px; margin-bottom: 6px }}
.box.gpu {{ background: #cde2fb }}
.arrow {{ display: flex; align-items: center; justify-content: center; font-size: 34px; color: {MUTED} }}
.loop {{ font: 500 26px 'IBM Plex Mono', monospace; color: {BLUE}; text-align: right }}
.eq {{ border: 2px solid #e1e0d9; border-radius: 16px; background: #fff; padding: 26px 30px; display: grid; gap: 18px }}
.eq .row {{ display: grid; grid-template-columns: 1fr auto; align-items: baseline; gap: 20px; font-size: 30px }}
.eq .row b {{ font: 600 52px 'IBM Plex Mono', monospace; color: {BLUE}; letter-spacing: -.02em }}
.eq .row span small {{ display: block; font-size: 24px; color: {MUTED} }}
/* bars */
.bars {{ display: grid; gap: 16px }}
.bar {{ display: grid; grid-template-columns: 290px 1fr; align-items: center; gap: 18px }}
.bar .name {{ font-size: 28px; font-weight: 500 }}
.bar .track {{ display: flex; align-items: center; gap: 14px }}
.bar .fill {{ height: 34px; border-radius: 0 8px 8px 0; min-width: 6px }}
.bar .val {{ font: 600 30px 'IBM Plex Mono', monospace; white-space: nowrap }}
.group {{ display: grid; gap: 12px }}
.group h3 {{ margin: 0 0 4px; font-size: 34px; font-weight: 600 }}
.legend {{ display: flex; flex-wrap: wrap; gap: 26px; font-size: 26px; color: {INK2} }}
.legend i {{ display: inline-block; width: 22px; height: 22px; border-radius: 5px; margin-right: 10px; vertical-align: -3px }}
.fix {{ display: grid; grid-template-columns: 64px 1fr; gap: 10px; font-size: 34px; line-height: 1.25; font-weight: 500 }}
.fix span {{ font: 600 34px 'IBM Plex Mono', monospace; color: {BLUE} }}
/* table */
table {{ width: 100%; border-collapse: collapse; font-size: 30px }}
th {{ font: 600 21px/1.25 'IBM Plex Mono', monospace; letter-spacing: .04em; text-transform: uppercase; color: {MUTED};
  text-align: left; padding: 0 12px 14px 0; border-bottom: 2px solid #c3c2b7; vertical-align: bottom }}
td {{ padding: 18px 12px 18px 0; border-bottom: 2px solid #ecebe6 }}
td.m {{ font-weight: 600 }}
td.n {{ font-family: 'IBM Plex Mono', monospace }}
td.hi {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: {BLUE} }}
/* dark slides */
.dark {{ background: #0d366b; color: #fff }}
.dark .eyebrow {{ color: #86b6ef }}
.dark .foot {{ border-color: #2f6db5; color: #86b6ef }}
.cover h1 {{ font-size: 92px; margin-top: 30px }}
.lede {{ font-size: 34px; line-height: 1.35; color: #cde2fb; margin: 32px 0 0; max-width: 880px }}
.stats {{ margin-top: auto; display: grid; gap: 34px }}
.stat {{ display: grid; grid-template-columns: 330px 1fr; align-items: baseline; gap: 24px; border-top: 2px solid #2f6db5; padding-top: 22px }}
.stat b {{ font: 600 76px/1 'IBM Plex Mono', monospace; letter-spacing: -.03em }}
.stat span {{ font-size: 28px; line-height: 1.3; color: #86b6ef }}
.swipe {{ margin-top: 44px; font: 600 28px 'IBM Plex Mono', monospace }}
.links {{ margin-top: 56px; display: grid; gap: 34px }}
.link small {{ display: block; font: 600 22px 'IBM Plex Mono', monospace; letter-spacing: .05em; text-transform: uppercase; color: #86b6ef }}
.link div {{ font-size: 36px; line-height: 1.3; margin-top: 8px }}
.ask {{ margin-top: auto; font-size: 50px; line-height: 1.2; font-weight: 600 }}
"""


def foot(i):
    return f'<div class="foot"><span>{AUTHOR}</span><span>{i} / {N}</span></div>'


def slide(i, eyebrow, head, body, take):
    return (f'<section class="slide"><div class="eyebrow">{eyebrow}</div><h1>{head}</h1>'
            f'<div class="body">{body}</div><p class="take">{take}</p>{foot(i)}</section>')


def bar(name, frac, val, color, maxw=520):
    return (f'<div class="bar"><div class="name">{name}</div><div class="track">'
            f'<div class="fill" style="width:{max(frac, 0.008) * maxw:.0f}px;background:{color}"></div>'
            f'<div class="val">{val}</div></div></div>')


def build():
    rtx, t4, gtx = airllm("local-rtx2070"), airllm("kaggle-t4"), airllm("local-gtx1070")
    rtx_s, t4_s = airllm("local-rtx2070", tuned=False), airllm("kaggle-t4", tuned=False)
    zg, a100, h100 = direct("zerogpu-rtxpro6000"), direct("modal-a100-80"), direct("modal-h100")
    ups_rtx = {m: rtx[m] / rtx_s[m] for m in rtx}
    best_t4 = max(t4[m] / t4_s[m] for m in t4_s)
    lo_all = min(min(rtx[m] / rtx_s[m] for m in rtx_s), min(t4[m] / t4_s[m] for m in t4_s))
    q = "Qwen3.8-27B"

    s = [f"""<section class="slide dark cover">
<div class="eyebrow">Benchmark · Gemma 4 and Qwen 3.x</div>
<h1>Running a 27B LLM on an 8&nbsp;GB laptop GPU</h1>
<p class="lede">AirLLM on two laptops and a free Kaggle T4, compared with vLLM on A100 and H100 GPUs.</p>
<div class="stats">
 <div class="stat"><b>27.8B</b><span>parameters in Qwen3.8-27B</span></div>
 <div class="stat"><b>3.7 GB</b><span>peak VRAM while it runs on an 8&nbsp;GB laptop GPU</span></div>
 <div class="stat"><b>{lo_all:.0f}–{best_t4:.0f}×</b><span>faster after two small fixes to AirLLM</span></div>
</div>
<div class="swipe">Swipe →</div>{foot(1)}</section>"""]

    cards = [("Own laptop", "RTX 2070 Max-Q", ["8 GB VRAM", "32 GB RAM", "Linux", "NVMe SSD"]),
             ("Own laptop", "GTX 1070 Max-Q", ["8 GB VRAM", "32 GB RAM", "Windows 11", "Pascal GPU"]),
             ("Kaggle · free notebook", "Tesla T4", ["16 GB VRAM", "32 GB RAM", "4 vCPU"])]
    body = "".join(f'<div class="card"><div class="who">{w}</div><div class="name">{n}</div><div class="chips">'
                   + "".join(f'<span class="{"hi" if "RAM" in c and "V" not in c else ""}">{c}</span>' for c in chips)
                   + "</div></div>" for w, n, chips in cards)
    s.append(slide(2, "01 · The setup", "Three small GPUs, all with 32&nbsp;GB of RAM", body,
                   "None can hold the larger models in VRAM, so AirLLM streams every model layer by layer."))

    body = f"""<div class="flow"><div class="box"><b>SSD</b>one file per layer</div><div class="arrow">→</div>
<div class="box"><b>RAM cache</b>if enough is free</div><div class="arrow">→</div>
<div class="box gpu"><b>GPU</b>one layer at a time</div></div>
<div class="loop">↻ every layer, again for every token</div>
<div class="eq"><div class="label">{q}, 4-bit: 15.1 GB read per token · time per token</div>
<div class="row"><span>from the SSD<small>1.3 GB/s · RTX 2070 laptop</small></span><b>{1 / rtx[q]:.1f} s</b></div>
<div class="row"><span>from the RAM cache<small>3.6 GB/s · Kaggle T4</small></span><b>{1 / t4[q]:.1f} s</b></div>
</div>"""
    s.append(slide(3, "02 · How AirLLM works", "Every generated token streams the whole model", body,
                   "Time per token ≈ bytes read ÷ read speed. The GPU is never the bottleneck."))

    top = max(ups_rtx.values())
    body = ('<div class="fix"><span>1</span><div>Quantize only the weight matrices, not norms and scalars</div></div>'
            '<div class="fix"><span>2</span><div>Drop the <code>gc.collect()</code> after every layer</div></div>'
            '<div class="label" style="margin-top:14px">Speed-up on the RTX 2070 laptop</div><div class="bars">'
            + "".join(bar(SHORT[m], v / top, f"×{v:.0f}", BLUE) for m, v in ups_rtx.items()) + "</div>")
    s.append(slide(4, "03 · Two small fixes", f"Two changes made AirLLM {lo_all:.0f}–{best_t4:.0f}× faster", body,
                   f"Up to ×{best_t4:.0f} on Kaggle's T4. The 27B models gain least: on the laptop they are limited by the SSD."))

    series = [("RTX 2070 laptop", rtx, BLUE), ("Kaggle T4", t4, ORANGE), ("GTX 1070 laptop", gtx, AQUA)]
    groups = ""
    for m in ("gemma-4-12B-it", q):
        mx = max(d[m] for _, d, _ in series)
        groups += (f'<div class="group"><h3>{SHORT[m]}</h3>'
                   + "".join(bar(n, d[m] / mx, f"{d[m]:.2f} tok/s", c, maxw=400) for n, d, c in series) + "</div>")
    body = groups + '<div class="label">bars scaled per model</div>'
    s.append(slide(5, "04 · The surprise", "Free RAM matters more than the GPU", body,
                   f"On {q} the fastest laptop came last: its 15&nbsp;GB of layer files did not fit in free RAM."))

    rows = [("Gemma-4-E4B", "1.8 GB", "8.3 GB", "2 min"), ("Qwen3.5-9B", "2.9 GB", "5.0 GB", "3 min"),
            ("Gemma-4-12B", "2.7 GB", "6.7 GB", "4 min"), ("Gemma-4-26B-A4B", "4.4 GB", "14.2 GB", "9 min"),
            ("27B models", "3.7 GB", "15.1 GB", "16–41 min")]
    body = ('<table><thead><tr><th>Model</th><th>Peak<br>VRAM</th><th>Free RAM<br>to cache</th>'
            '<th>200-token<br>answer</th></tr></thead><tbody>'
            + "".join(f'<tr><td class="m">{a}</td><td class="n">{b}</td><td class="hi">{c}</td><td class="n">{d}</td></tr>'
                      for a, b, c, d in rows) + "</tbody></table>")
    s.append(slide(6, "05 · In practice", "VRAM is never the limit. RAM&nbsp;is.", body,
                   "Close memory-hungry apps before a 27B run, and plan for minutes per answer."))

    lo, hi = 0.05, 100
    W = lambda v: math.log10(v / lo) / math.log10(hi / lo)
    ref = [("AirLLM · RTX 2070 laptop", rtx[q], ORANGE), ("AirLLM · Kaggle T4", t4[q], ORANGE),
           ("transformers · HF ZeroGPU", zg[q][0], BLUE), ("vLLM · A100 80 GB", a100[q][0], BLUE),
           ("vLLM · H100 80 GB", h100[q][0], BLUE)]
    body = (f'<div class="label">{q}, one request, tokens per second (log scale)</div><div class="bars">'
            + "".join(bar(n, W(v), f"{v:.2f}" if v < 1 else f"{v:.1f}", c) for n, v, c in ref) + "</div>"
            + '<div class="legend"><span><i style="background:#eb6834"></i>streamed, model does not fit</span>'
              '<span><i style="background:#2a78d6"></i>whole model on the GPU</span></div>')
    s.append(slide(7, "06 · The reference", f"An H100 runs the 27B model {h100[q][0] / t4[q]:.0f}× faster", body,
                   "AirLLM runs models that don't fit your GPU. For chat or agents, serve the model with vLLM."))

    s.append(f"""<section class="slide dark"><div class="eyebrow">Read more</div>
<h1>The full write-up, code and raw results</h1>
<div class="links">
 <div class="link"><small>Article on Medium</small><div>medium.com/@komari.hannan</div></div>
 <div class="link"><small>Code and every raw result</small><div>github.com/hannan72/airllm-benchmarks</div></div>
 <div class="link"><small>Fixes proposed upstream</small><div>lyogavin/airllm pull requests #410, #411, #412</div></div>
</div>
<div class="ask">What would you run on an 8&nbsp;GB GPU?</div>{foot(N)}</section>""")
    assert len(s) == N
    return (f"<!doctype html><html><head><meta charset='utf-8'><link rel='stylesheet' href='{FONTS}'>"
            f"<style>{CSS}</style></head><body>{''.join(s)}</body></html>")


def main():
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "carousel.html"
        src.write_text(build())
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run",
                        f"--user-data-dir={Path(t) / 'profile'}", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--print-to-pdf={OUT}", src.as_uri()],
                       check=True, capture_output=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
