#!/usr/bin/env python
"""results/gpus.png: single-stream decode speed of every model on every GPU (log scale).

Small GPUs (RTX 2070, T4, GTX 1070): AirLLM tuned (NF4 matrices only + no gc), because the models do not fit.
Big GPUs: the model deployed directly; ZeroGPU with plain transformers (vLLM cannot run there),
A100 / H100 on Modal with vLLM. One dot per GPU, one row per model.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report import AIRLLM_HOSTS, ORDER, _best, load, load_direct  # noqa: E402

LABEL = {"gemma-4-E2B-it-qat-mobile-ct": "Gemma-4-E2B-qat", "gemma-4-E4B-it": "Gemma-4-E4B",
         "Qwen3.5-9B": "Qwen3.5-9B", "gemma-4-12B-it": "Gemma-4-12B", "gemma-4-26B-A4B-it": "Gemma-4-26B-A4B",
         "Qwen3.6-27B": "Qwen3.6-27B", "Qwen3.8-27B": "Qwen3.8-27B"}
# reference palette (dataviz skill), categorical slots 1-7 in fixed order, light mode
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [
    ("local-rtx2070", "RTX 2070 8 GB · AirLLM tuned", "#2a78d6", "o"),
    ("kaggle-t4", "T4 16 GB · AirLLM tuned", "#eb6834", "s"),
    ("zerogpu-rtxpro6000", "ZeroGPU RTX PRO 6000 · transformers", "#1baf7a", "D"),
    ("modal-a100-40", "A100 40 GB · vLLM (fp8 for 26–27B)", "#eda100", "^"),
    ("modal-a100-80", "A100 80 GB · vLLM", "#e87ba4", "v"),
    ("modal-h100", "H100 80 GB · vLLM", "#008300", "h"),
    ("local-gtx1070", "GTX 1070 8 GB · AirLLM tuned", "#6250d6", "X"),
]


def speeds(host):
    d = ROOT / "results" / host
    if not d.is_dir():
        return {}
    if host in AIRLLM_HOSTS:
        rows = load(d)
        out = {}
        for m in ORDER:
            r = _best(rows, m, True)
            if r:
                out[m] = r["runs"]["warm"]["decode_tok_per_s"]
        return out
    out = {}
    for r in load_direct(d):
        if r.get("status") == "ok":
            out[r["model"]] = max(out.get(r["model"], 0), r["runs"]["single"]["decode_tok_per_s"])
    return out


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(ORDER)
    present = [(h, lab, c, mk, speeds(h)) for h, lab, c, mk in SERIES]
    present = [p for p in present if p[4]]
    for i, m in enumerate(ORDER):
        y = n - 1 - i
        vals = [p[4][m] for p in present if m in p[4]]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
    for h, lab, c, mk, sp in present:
        xs = [sp[m] for m in ORDER if m in sp]
        ys = [n - 1 - ORDER.index(m) for m in ORDER if m in sp]
        ax.scatter(xs, ys, s=64, color=c, marker=mk, edgecolors=SURFACE, linewidths=2, zorder=3, label=lab)
    ax.set_xscale("log")
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    allv = [v for p in present for v in p[4].values()]
    ax.set_xlim(min(allv) / 1.6, max(allv) * 1.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([LABEL[m] for m in reversed(ORDER)], color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("decode tokens / second, one request (log scale, higher is better)", color=INK2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_title("Same models, same prompt: streamed with AirLLM on small GPUs vs. deployed directly on big ones",
                 loc="left", fontsize=11, color=INK, pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    out = ROOT / "results" / "gpus.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
