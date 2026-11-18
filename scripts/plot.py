#!/usr/bin/env python
"""results/speed.png: decode speed, stock AirLLM vs tuned, on both GPUs (small multiples).

"Stock" = the fastest stock AirLLM 4.0 configuration whose output is correct
(Gemma 4 bf16, because AirLLM's 4-bit garbles Gemma; Qwen 4-bit; E2B-qat native weights).
"Tuned" = NF4 on weight matrices only + no per-layer gc (E2B-qat: native weights + no gc).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["gemma-4-E2B-it-qat-mobile-ct", "gemma-4-E4B-it", "Qwen3.5-9B", "gemma-4-12B-it",
          "gemma-4-26B-A4B-it", "Qwen3.6-27B", "Qwen3.8-27B"]
LABEL = {"gemma-4-E2B-it-qat-mobile-ct": "Gemma-4-E2B-qat", "gemma-4-E4B-it": "Gemma-4-E4B",
         "Qwen3.5-9B": "Qwen3.5-9B", "gemma-4-12B-it": "Gemma-4-12B", "gemma-4-26B-A4B-it": "Gemma-4-26B-A4B",
         "Qwen3.6-27B": "Qwen3.6-27B", "Qwen3.8-27B": "Qwen3.8-27B"}
GARBLED = {("gemma-4-E4B-it", "4bit"), ("gemma-4-12B-it", "4bit"), ("gemma-4-26B-A4B-it", "4bit")}
HOSTS = [("local-rtx2070", "RTX 2070 Max-Q laptop (8 GB)"), ("kaggle-t4", "Kaggle Tesla T4 (16 GB)")]

# reference palette (dataviz skill): categorical slots 1-2 + chrome, light mode
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TUNED, STOCK = "#2a78d6", "#eb6834"


FAILED = {}


def load(host):
    out = {}
    for f in (ROOT / "results" / host).glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("status") != "ok" or "warm" not in r.get("runs", {}):
            if not r.get("no_gc"):
                FAILED[(host, r["model"])] = r.get("error_phase", "error")
            continue
        out[(r["model"], r["compression"], bool(r.get("no_gc")))] = r["runs"]["warm"]["decode_tok_per_s"]
    return out


def pick(res, model):
    stock = [v for (m, c, g), v in res.items()
             if m == model and not g and c in ("none", "4bit") and (m, c) not in GARBLED]
    tuned = [v for (m, c, g), v in res.items() if m == model and g and c in ("4bit-2d", "none")]
    if model.endswith("-ct"):
        tuned = [v for (m, c, g), v in res.items() if m == model and g]
    return (max(stock) if stock else None), (max(tuned) if tuned else None)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    xmax = 0
    data = {h: {m: pick(load(h), m) for m in MODELS} for h, _ in HOSTS}
    for h in data:
        for s, t in data[h].values():
            xmax = max(xmax, s or 0, t or 0)
    for ax, (host, title) in zip(axes, HOSTS):
        ax.set_facecolor(SURFACE)
        ys = range(len(MODELS))
        for i, m in enumerate(MODELS):
            s, t = data[host][m]
            y = len(MODELS) - 1 - i
            if t is not None:
                ax.barh(y + 0.19, t, height=0.36, color=TUNED, edgecolor=SURFACE, linewidth=1.5)
            if s is not None:
                ax.barh(y - 0.19, s, height=0.36, color=STOCK, edgecolor=SURFACE, linewidth=1.5)
            if s and t:
                ax.text(t + xmax * 0.012, y + 0.19, f"{t:.2f} tok/s  ×{t / s:.0f}", va="center",
                        fontsize=8.5, color=INK2)
            elif t:
                ax.text(t + xmax * 0.012, y + 0.19, f"{t:.2f} tok/s", va="center", fontsize=8.5, color=INK2)
            if s is None:
                why = ("stock AirLLM failed: 49.9 GB shard" if (host, m) in FAILED else "stock: pending")
                ax.text(xmax * 0.012, y - 0.19, why, va="center", fontsize=8, color=MUTED, style="italic")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([LABEL[m] for m in reversed(MODELS)], color=INK)
        ax.set_xlim(0, xmax * 1.32)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("decode tokens / second (warm, higher is better)", color=INK2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TUNED), plt.Rectangle((0, 0), 1, 1, color=STOCK)]
    fig.legend(handles, ["tuned: NF4 on weight matrices only + no per-layer gc",
                         "stock AirLLM 4.0 (fastest config with correct output)"],
               loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "results" / "speed.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
