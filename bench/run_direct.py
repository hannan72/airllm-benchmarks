#!/usr/bin/env python
"""Benchmark one model deployed directly on a GPU that holds all of it (no AirLLM); write JSON.

Engines
  vllm  vLLM's offline engine: bf16 weights (or ``--quantization fp8`` when bf16 does not fit),
        torch.compile + CUDA graphs, prefix caching off, multimodal encoders disabled.
  hf    plain transformers ``generate()`` with the whole model on one GPU (eager, dynamic cache).
        Used on Hugging Face ZeroGPU, which runs PyTorch inside a Gradio Space but not vLLM.

Same prompt, chat template (thinking off) and greedy decoding as the AirLLM runs, measured as
  single  one request: time to first token (prefill) and decode tok/s (median of --repeats)
  batch   --batch identical requests x --batch-tokens tokens each: aggregate output tok/s (vllm)
Download and engine start-up (weights load, compile, CUDA-graph capture) are timed separately.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from measure import DEFAULT_PROMPT, Sampler, TokenTimer, build_inputs  # noqa: E402
from model_list import MODELS  # noqa: E402

ALLOW = ["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"]


# ---------------------------------------------------------------------------------------------
# environment


def host_info():
    import psutil

    info = {
        "hostname_tag": os.environ.get("BENCH_HOST_TAG", socket.gethostname()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "",
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    from importlib.metadata import version

    for pkg in ("torch", "transformers", "vllm", "compressed-tensors", "flashinfer-python", "spaces", "gradio"):
        try:
            info[pkg.replace("-", "_")] = version(pkg)
        except Exception:  # noqa: BLE001
            pass
    return info


def gpu_info():
    """Needs a real CUDA context (on ZeroGPU: call inside the @spaces.GPU function)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name,
        "gpu_vram_gb": round(p.total_memory / 1e9, 2),
        "gpu_cc": f"{p.major}.{p.minor}",
        "gpu_count": torch.cuda.device_count(),
        "cuda_runtime": torch.version.cuda,
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
    }


def download(repo_id):
    from huggingface_hub import snapshot_download

    t = time.perf_counter()
    path = Path(snapshot_download(repo_id, allow_patterns=ALLOW))
    size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    return path, round(time.perf_counter() - t, 2), round(size / 1e9, 3)


def modalities(path):
    """Multimodal inputs the checkpoint declares; vLLM gets a limit of 0 for each (text only)."""
    cfg = json.loads((Path(path) / "config.json").read_text())
    mods = []
    if cfg.get("vision_config"):
        mods.append("image")
        if cfg.get("model_type", "").startswith("qwen"):
            mods.append("video")
    if cfg.get("audio_config"):
        mods.append("audio")
    return mods


# ---------------------------------------------------------------------------------------------
# vLLM


class _LogTap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


_VLLM_PATTERNS = {
    "weights_gib": (r"Model loading took ([\d.]+) ?GiB", float),
    "weights_load_s": (r"Model loading took [\d.]+ ?GiB(?: memory)? and ([\d.]+) ?s", float),
    "kv_cache_gib": (r"Available KV cache memory: ([\d.]+) ?GiB", float),
    "kv_cache_tokens": (r"GPU KV cache size: ([\d,]+) tokens", lambda s: int(s.replace(",", ""))),
    "max_concurrency": (r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", float),
    "compile_s": (r"torch\.compile takes ([\d.]+) ?s in total", float),
    "graph_capture_s": (r"Graph capturing finished in ([\d.]+) ?sec", float),
    "attention_backend": (r"Using ([A-Z_]+[A-Z0-9_]*) (?:attention )?backend", str),
}


def parse_vllm_log(lines):
    out = {}
    for key, (pat, conv) in _VLLM_PATTERNS.items():
        for line in lines:
            m = re.search(pat, line)
            if m:
                try:
                    out[key] = conv(m.group(1))
                except ValueError:
                    pass
                break
    return out


def vllm_bench(path, args, result, sampler):
    # engine core in this process: its logs reach our handler and NVML per-process memory is ours
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tap = _LogTap()
    logging.getLogger("vllm").addHandler(tap)

    tok = AutoTokenizer.from_pretrained(path)
    enc, used_template, _ = build_inputs(tok, args.prompt)
    prompt = {"prompt_token_ids": enc["input_ids"][0].tolist()}
    result["chat_template"] = used_template

    kw = dict(model=str(path), dtype=args.dtype, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=False, seed=0)
    if args.quantization:
        kw["quantization"] = args.quantization
    mods = modalities(path)
    if mods:
        kw["limit_mm_per_prompt"] = {m: 0 for m in mods}
    result["engine_args"] = {k: v for k, v in kw.items() if k != "model"}

    sampler.reset()
    t = time.perf_counter()
    llm = LLM(**kw)
    result["phases"]["load_s"] = round(time.perf_counter() - t, 2)
    result["phases"]["load_mem"] = sampler.snapshot()

    def run(n, batch=1, ignore_eos=True):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=ignore_eos)
        t0 = time.perf_counter()
        outs = llm.generate([prompt] * batch, sp, use_tqdm=False)
        return time.perf_counter() - t0, outs

    run(8)  # warm-up
    sampler.reset()
    ttfts = [run(1)[0] for _ in range(args.repeats)]
    totals = [run(args.tokens)[0] for _ in range(args.repeats)]
    ttft, total = statistics.median(ttfts), statistics.median(totals)
    _, outs = run(32, ignore_eos=False)
    single = {
        "prompt_tokens": len(prompt["prompt_token_ids"]),
        "new_tokens": args.tokens,
        "ttft_s": round(ttft, 4),
        "ttft_samples_s": [round(x, 4) for x in ttfts],
        "total_s": round(total, 4),
        "total_samples_s": [round(x, 4) for x in totals],
        "decode_tok_per_s": round((args.tokens - 1) / (total - ttft), 2),
        "end_to_end_tok_per_s": round(args.tokens / total, 2),
        "text": outs[0].outputs[0].text,
    }
    single.update(sampler.snapshot())
    result["runs"]["single"] = single
    print(json.dumps(single, indent=1), flush=True)

    if args.batch > 1:
        sampler.reset()
        dt, outs = run(args.batch_tokens, batch=args.batch)
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        batch = {"batch": args.batch, "tokens_each": args.batch_tokens, "total_s": round(dt, 3),
                 "output_tokens": n_out, "output_tok_per_s": round(n_out / dt, 1)}
        batch.update(sampler.snapshot())
        result["runs"]["batch"] = batch
        print(json.dumps(batch, indent=1), flush=True)

    result["engine_info"] = parse_vllm_log(tap.lines)
    result["engine_log_tail"] = tap.lines[-60:]


# ---------------------------------------------------------------------------------------------
# plain transformers (also imported by the ZeroGPU Space)


def hf_load(path, dtype="bfloat16", device_map="cuda"):
    """Whole model on one device. Multimodal checkpoints load with their image-text class so that
    every language-model weight lands where the checkpoint has it (no silently missing keys)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    cls = AutoModelForImageTextToText if modalities(path) else AutoModelForCausalLM
    model, info = cls.from_pretrained(path, dtype=getattr(torch, dtype), device_map=device_map,
                                      output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    if missing:
        raise RuntimeError(f"{len(missing)} weights missing after load, e.g. {missing[:3]}")
    model.eval()
    return model, tok, cls.__name__


def hf_measure(model, tok, prompt=DEFAULT_PROMPT, n_tokens=64, repeats=2, text_tokens=32):
    import torch
    from transformers import LogitsProcessorList

    enc, used_template, _ = build_inputs(tok, prompt)
    dev = next(model.parameters()).device
    ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    gen = dict(do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    with torch.inference_mode():
        model.generate(ids, attention_mask=attn, max_new_tokens=4, **gen)  # warm-up
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for _ in range(repeats):
            timer = TokenTimer()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(ids, attention_mask=attn, max_new_tokens=n_tokens, min_new_tokens=n_tokens,
                                 logits_processor=LogitsProcessorList([timer]), **gen)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ts = timer.times
            runs.append({"ttft_s": ts[0] - t0, "decode_tok_per_s": (len(ts) - 1) / (ts[-1] - ts[0]),
                         "total_s": t1 - t0, "n": len(ts)})
    new = out[0, ids.shape[1]:]
    return {
        "chat_template": used_template,
        "prompt_tokens": int(ids.shape[1]),
        "new_tokens": runs[-1]["n"],
        "ttft_s": round(statistics.median(r["ttft_s"] for r in runs), 4),
        "ttft_samples_s": [round(r["ttft_s"], 4) for r in runs],
        "decode_tok_per_s": round(statistics.median(r["decode_tok_per_s"] for r in runs), 2),
        "decode_samples_tok_per_s": [round(r["decode_tok_per_s"], 2) for r in runs],
        "total_s": round(statistics.median(r["total_s"] for r in runs), 4),
        "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "text": tok.decode(new[:text_tokens], skip_special_tokens=True),
    }


def hf_bench(path, args, result, sampler):
    sampler.reset()
    t = time.perf_counter()
    model, tok, cls = hf_load(path, dtype=args.dtype if args.dtype != "auto" else "bfloat16")
    result["phases"]["load_s"] = round(time.perf_counter() - t, 2)
    result["phases"]["load_mem"] = sampler.snapshot()
    result["hf_class"] = cls
    sampler.reset()
    single = hf_measure(model, tok, args.prompt, n_tokens=args.tokens, repeats=args.repeats)
    single.update(sampler.snapshot())
    result["chat_template"] = single.pop("chat_template")
    result["runs"]["single"] = single
    print(json.dumps(single, indent=1), flush=True)


# ---------------------------------------------------------------------------------------------


def result_name(short, engine, quantization):
    return f"{short}__{engine}-{quantization or 'bf16'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="short name from bench/model_list.py or a HF repo id")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--quantization", default=None, help="vllm only, e.g. fp8 (on-the-fly, W8A16 on Ampere)")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tokens", type=int, default=256, help="decode length for the single-stream run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batch-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "direct"))
    args = ap.parse_args()

    repo_id = MODELS.get(args.model, args.model)
    short = args.model if args.model in MODELS else repo_id.split("/")[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{result_name(short, args.engine, args.quantization)}.json"

    result = {
        "model": short, "repo_id": repo_id, "engine": args.engine,
        "weights": args.quantization or "bf16", "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt": args.prompt, "phases": {}, "runs": {}, "env": host_info(),
    }
    sampler = Sampler()
    sampler.start()
    phase = "download"
    try:
        path, result["phases"]["download_s"], result["phases"]["checkpoint_gb"] = download(repo_id)
        phase = "load+generate"
        (vllm_bench if args.engine == "vllm" else hf_bench)(path, args, result, sampler)
        result["env"].update(gpu_info())
        result["status"] = "ok"
    except BaseException as e:  # noqa: BLE001 - record OOMs, KeyboardInterrupt too
        result["status"] = "error"
        result["error_phase"] = phase
        result["error"] = f"{type(e).__name__}: {e}"[:2000]
        result["traceback"] = traceback.format_exc()[-6000:]
        print(result["traceback"], file=sys.stderr, flush=True)
        try:
            result["env"].update(gpu_info())
        except Exception:  # noqa: BLE001
            pass
    finally:
        sampler.stop()
        result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_file.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_file} status={result['status']}", flush=True)
    # vLLM leaves non-daemon threads behind in-process; exit hard so the matrix moves on
    os._exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
