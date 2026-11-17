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
    if r["model"].endswith("-ct"):
        return "native QAT int2/4/8"
    return {"none": "bf16", "4bit": "NF4 (AirLLM 4bit)", "8bit": "int8 (AirLLM 8bit)"}[r["compression"]]


def load(host_dir):
    rows = [json.loads(f.read_text()) for f in sorted(host_dir.glob("*.json"))]
    key = lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99, r["compression"] != "4bit")
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
    print(text)


if __name__ == "__main__":
    sys.exit(main())
