#!/usr/bin/env python
"""Summarise results/<host>/*.json into markdown tables (printed, and written to results/SUMMARY.md)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["gemma-4-E2B-it-qat-mobile-ct", "gemma-4-E4B-it", "Qwen3.5-9B", "gemma-4-12B-it",
         "gemma-4-26B-A4B-it", "Qwen3.6-27B", "Qwen3.8-27B"]
PARAMS = {"gemma-4-E2B-it-qat-mobile-ct": "5.5B (2B eff.)", "gemma-4-E4B-it": "8.0B (4B eff.)",
          "Qwen3.5-9B": "9.7B", "gemma-4-12B-it": "12.0B", "gemma-4-26B-A4B-it": "25.8B MoE (4B act.)",
          "Qwen3.6-27B": "27.8B", "Qwen3.8-27B": "27.8B"}
CKPT_GB = {"gemma-4-E2B-it-qat-mobile-ct": 2.7, "gemma-4-E4B-it": 16.0, "Qwen3.5-9B": 19.3,
           "gemma-4-12B-it": 23.9, "gemma-4-26B-A4B-it": 51.6, "Qwen3.6-27B": 55.6, "Qwen3.8-27B": 55.6}


def fmt(v, nd=2, suffix=""):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def mode_label(r):
    return _mode_label(r) + (" + no per-layer gc" if r.get("no_gc") else "")


def _mode_label(r):
    if r["model"].endswith("-ct"):
        return "native QAT int2/4/8"
    return {"none": "bf16", "4bit": "NF4, all tensors (AirLLM 4bit)", "8bit": "int8 (AirLLM 8bit)",
            "4bit-2d": "NF4, matrices only", "8bit-2d": "int8, matrices only"}[r["compression"]]


def load(host_dir):
    rows = [json.loads(f.read_text()) for f in sorted(host_dir.glob("*.json"))]
    rank = {"4bit": 0, "none": 1, "4bit-2d": 2, "8bit": 3, "8bit-2d": 4}
    key = lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99, bool(r.get("no_gc")),
                     rank.get(r["compression"], 9))
    return sorted(rows, key=key)


def table(rows):
    out = ["| Model | Params | Weights | Layer files on disk | One-time convert | Load | TTFT cold | s/token cold | "
           "s/token warm | tok/s warm | Peak VRAM (proc / torch) | Peak RAM | Output (first words) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") != "ok":
            out.append(f"| {r['model']} | {PARAMS.get(r['model'], '')} | {mode_label(r)} | – | – | – | – | – | – | – | – | – | "
                       f"**failed** in `{r.get('error_phase')}`: {r.get('error', '')[:110].replace('|', '/')} |")
            continue
        p, c, w = r["phases"], r["runs"].get("cold", {}), r["runs"].get("warm", {})
        conv = "cached" if p.get("split_preexisting") else fmt(p.get("convert_s", 0) / 60, 1, " min")
        if p.get("lazy_split") and p["lazy_split"].get("bytes_read"):
            conv += " (lazy split" + (", from Hub" if "remote" in p["lazy_split"].get("sources", []) else "") + ")"
        vram = f"{fmt(max(c.get('peak_gpu_process_gb') or 0, w.get('peak_gpu_process_gb') or 0))} / " \
               f"{fmt(max(c.get('torch_peak_reserved_gb') or 0, w.get('torch_peak_reserved_gb') or 0))} GB"
        ram = fmt(max(c.get("peak_rss_gb") or 0, w.get("peak_rss_gb") or 0, (p.get("init_mem") or {}).get("peak_rss_gb") or 0), 1, " GB")
        text = (w.get("text") or c.get("text") or "").strip().replace("\n", " ").replace("|", "/")
        text = " ".join(text.split()[:12]) + ("…" if len(text.split()) > 12 else "")
        out.append(
            f"| {r['model']} | {PARAMS.get(r['model'], '')} | {mode_label(r)} | {fmt(p.get('split_size_gb'), 1, ' GB')} | {conv} | "
            f"{fmt(p.get('init_s'), 0, ' s')} | {fmt(c.get('ttft_s'), 1, ' s')} | {fmt(c.get('s_per_decode_token'), 2)} | "
            f"{fmt(w.get('s_per_decode_token'), 2)} | {fmt(w.get('decode_tok_per_s'), 3)} | {vram} | {ram} | {text} |")
    return "\n".join(out)


def io_table(rows):
    out = ["| Model | Weights | Streamed per token (GB) | Layer read time / token | Effective read GB/s (cold) | "
           "Host→GPU + dequant / token | Prefetch |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") != "ok":
            continue
        c = r["runs"]["cold"]
        n_pass = max(1, c["new_tokens"])  # one full model pass per generated token (prefill = first)
        out.append(f"| {r['model']} | {mode_label(r)} | {fmt(c['layer_load_gb'] / n_pass, 2)} | "
                   f"{fmt(c['layer_load_s'] / n_pass, 2, ' s')} | {fmt(c.get('layer_load_gbps'), 2)} | "
                   f"{fmt(c['to_device_s'] / n_pass, 2, ' s')} | {'on' if r.get('prefetching') else 'off'} |")
    return "\n".join(out)


def env_line(rows):
    e = next((r["env"] for r in rows if r.get("env")), {})
    return (f"{e.get('gpu', '?')} ({e.get('gpu_vram_gb', '?')} GB, sm_{e.get('gpu_cc', '?').replace('.', '')}, "
            f"bf16 native: {e.get('bf16_native')}) · {e.get('cpu', '?')} · {e.get('ram_total_gb', '?')} GB RAM · "
            f"torch {e.get('torch')} (CUDA {e.get('cuda_runtime')}) · transformers {e.get('transformers')} · "
            f"airllm {e.get('airllm')} · bitsandbytes {e.get('bitsandbytes')}")


GARBLED = {("gemma-4-E4B-it", "4bit"), ("gemma-4-12B-it", "4bit"), ("gemma-4-26B-A4B-it", "4bit")}
HOST_LABEL = {"local-rtx2070": "RTX 2070 Max-Q (8 GB)", "kaggle-t4": "Kaggle T4 (16 GB)"}


def _best(rows, model, tuned):
    """Fastest correct stock config, or the tuned config (NF4 matrices only / native + no gc)."""
    cands = []
    for r in rows:
        if r["model"] != model or r.get("status") != "ok" or "warm" not in r.get("runs", {}):
            continue
        if tuned != bool(r.get("no_gc")):
            continue
        if not tuned and (r["compression"] not in ("none", "4bit") or (model, r["compression"]) in GARBLED):
            continue
        if tuned and not (r["compression"] == "4bit-2d" or model.endswith("-ct")):
            continue
        cands.append(r)
    return max(cands, key=lambda r: r["runs"]["warm"]["decode_tok_per_s"], default=None)


def headline(hosts):
    out = ["| Model | GPU | Stock AirLLM 4.0 (best correct config) | Tuned (NF4 matrices only + no gc) | Speed-up | Peak VRAM (tuned) |",
           "|---|---|---|---|---|---|"]
    hosts = sorted(hosts, key=lambda d: (d.name != "local-rtx2070", d.name))
    for m in ORDER:
        for d in hosts:
            rows = load(d)
            s, t = _best(rows, m, False), _best(rows, m, True)
            if s is None and t is None:
                continue
            failed = [r for r in rows if r["model"] == m and r.get("status") != "ok" and not r.get("no_gc")]
            def cell(r):
                w = r["runs"]["warm"]
                return f"{w['s_per_decode_token']:.2f} s/token ({w['decode_tok_per_s']:.3f} tok/s), {mode_label(r).split(' (')[0].replace(' + no per-layer gc', '')}"
            sc = cell(s) if s else ("split fails (49.9 GB shard)" if failed else "pending")
            tc = cell(t) if t else "pending"
            sp = f"**×{t['runs']['warm']['decode_tok_per_s'] / s['runs']['warm']['decode_tok_per_s']:.0f}**" if s and t else "–"
            vr = f"{t['runs']['warm']['peak_gpu_process_gb']:.1f} GB" if t else "–"
            out.append(f"| {m} | {HOST_LABEL.get(d.name, d.name)} | {sc} | {tc} | {sp} | {vr} |")
    return "\n".join(out)


def main():
    hosts = [d for d in sorted((ROOT / "results").iterdir()) if d.is_dir() and d.name != "smoke"]
    md = []
    for d in hosts:
        rows = load(d)
        if not rows:
            continue
        md += [f"### {d.name}", "", env_line(rows), "", table(rows), "", "I/O breakdown (cold run):", "", io_table(rows), ""]
    text = "\n".join(md)
    (ROOT / "results" / "SUMMARY.md").write_text(text + "\n")
    if "--readme" in sys.argv:  # paste the tables into README.md between markers
        readme = ROOT / "README.md"
        r = readme.read_text()
        hs, he = "<!-- headline:start -->", "<!-- headline:end -->"
        r = r[:r.index(hs) + len(hs)] + "\n" + headline(hosts) + "\n" + r[r.index(he):]
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        i, j = r.index(start) + len(start), r.index(end)
        readme.write_text(r[:i] + "\n" + text.replace("### ", "#### ") + "\n" + r[j:])
        print("README.md updated")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
