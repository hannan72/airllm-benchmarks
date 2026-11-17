#!/usr/bin/env python
"""Benchmark one model under AirLLM and write a JSON result.

Phases (each timed separately):
  download  - huggingface snapshot of the original checkpoint (0 if cached)
  convert   - AirLLM's one-time split into per-layer shards (+ optional 4/8-bit compression)
  init      - build the empty model and load resident modules
  generate  - greedy generation, run twice:
                cold: split files evicted from the OS page cache first (posix_fadvise)
                warm: immediately after, whatever the page cache kept
Per-token timestamps come from a LogitsProcessor, so prefill (TTFT) and decode speed are
measured separately. Layer I/O time and bytes are taken by wrapping AirLLM's loader.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_PROMPT = "Explain how a CPU cache works in one short paragraph."


# ---------------------------------------------------------------------------------------------
# monitoring


class Sampler(threading.Thread):
    """Samples process RSS and GPU memory (NVML, whole device and this process) every 100 ms."""

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self.reset()
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            self.nvml = None

    def reset(self):
        self.peak_rss = 0
        self.peak_gpu_device = 0
        self.peak_gpu_process = 0

    def gpu(self):
        if self.nvml is None:
            return 0, 0
        dev = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        mine = 0
        try:
            for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if p.pid == os.getpid() and p.usedGpuMemory:
                    mine = p.usedGpuMemory
        except Exception:  # noqa: BLE001
            pass
        return dev, mine

    def run(self):
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
                dev, mine = self.gpu()
                self.peak_gpu_device = max(self.peak_gpu_device, dev)
                self.peak_gpu_process = max(self.peak_gpu_process, mine)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def snapshot(self):
        return {
            "peak_rss_gb": round(self.peak_rss / 1e9, 3),
            "peak_gpu_device_used_gb": round(self.peak_gpu_device / 1e9, 3),
            "peak_gpu_process_gb": round(self.peak_gpu_process / 1e9, 3),
        }


class IOStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.load_s = 0.0
        self.load_bytes = 0
        self.load_calls = 0
        self.to_device_s = 0.0

    def as_dict(self):
        return {
            "layer_load_s": round(self.load_s, 3),
            "layer_load_gb": round(self.load_bytes / 1e9, 3),
            "layer_load_calls": self.load_calls,
            "layer_load_gbps": round(self.load_bytes / 1e9 / self.load_s, 3) if self.load_s else None,
            "to_device_s": round(self.to_device_s, 3),
        }


def instrument(model, stats: IOStats):
    """Wrap AirLLM's per-layer loader (runs on the prefetch thread too) and device move."""
    orig_load = model.load_layer_to_cpu
    orig_move = model.move_layer_to_device
    ckpt = Path(model.checkpoint_path)

    def load(layer_name):
        t = time.perf_counter()
        out = orig_load(layer_name)
        dt = time.perf_counter() - t
        size = 0
        f = ckpt / (layer_name + ".safetensors")
        if f.exists():
            size = f.stat().st_size
        with stats.lock:
            stats.load_s += dt
            stats.load_bytes += size
            stats.load_calls += 1
        return out

    def move(state_dict):
        t = time.perf_counter()
        out = orig_move(state_dict)
        torch.cuda.synchronize()
        with stats.lock:
            stats.to_device_s += time.perf_counter() - t
        return out

    model.load_layer_to_cpu = load
    model.move_layer_to_device = move


def evict_page_cache(path: Path) -> int:
    """Drop clean cached pages of every file under ``path`` (no root needed)."""
    n = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                fd = os.open(f, os.O_RDONLY)
                os.fdatasync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
    return n


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9, 3)


def env_info():
    import airllm  # noqa: F401
    import transformers
    from importlib.metadata import version

    info = {
        "hostname_tag": os.environ.get("BENCH_HOST_TAG", socket.gethostname()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "",
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "airllm": version("airllm"),
        "bitsandbytes": version("bitsandbytes"),
    }
    try:
        info["compressed_tensors"] = version("compressed-tensors")
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": p.name,
            "gpu_vram_gb": round(p.total_memory / 1e9, 2),
            "gpu_cc": f"{p.major}.{p.minor}",
            "gpu_count": torch.cuda.device_count(),
            "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
        })
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------------------------
# generation


class TokenTimer:
    """LogitsProcessor that timestamps each decoding step (called once per generated token)."""

    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return scores


def build_inputs(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = None
    if getattr(tokenizer, "chat_template", None):
        for kw in ({"enable_thinking": False}, {}):
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
                break
            except Exception:  # noqa: BLE001
                continue
    used_template = text is not None
    if text is None:
        text = prompt
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=not used_template)
    return enc, used_template, text


def timed_generate(model, enc, max_new_tokens, sampler, stats):
    from transformers import LogitsProcessorList

    timer = TokenTimer()
    stats.reset()
    sampler.reset()
    torch.cuda.reset_peak_memory_stats()
    input_ids = enc["input_ids"].cuda()
    attn = enc.get("attention_mask")
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        use_cache=True,
        return_dict_in_generate=True,
        logits_processor=LogitsProcessorList([timer]),
    )
    if attn is not None:
        kwargs["attention_mask"] = attn.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(input_ids, **kwargs)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    new_tokens = out.sequences[0, input_ids.shape[1]:]
    ts = timer.times
    n = len(ts)
    ttft = ts[0] - t0 if n else None
    decode_s = ts[-1] - ts[0] if n > 1 else None
    per_tok = [round(b - a, 4) for a, b in zip(ts, ts[1:])]
    res = {
        "prompt_tokens": int(input_ids.shape[1]),
        "new_tokens": int(new_tokens.shape[0]),
        "total_s": round(t_end - t0, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "decode_s": round(decode_s, 3) if decode_s is not None else None,
        "decode_tok_per_s": round((n - 1) / decode_s, 4) if decode_s else None,
        "s_per_decode_token": round(decode_s / (n - 1), 3) if decode_s else None,
        "end_to_end_tok_per_s": round(n / (t_end - t0), 4) if n else None,
        "per_token_s": per_tok,
        "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "text": model.tokenizer.decode(new_tokens, skip_special_tokens=True),
    }
    res.update(stats.as_dict())
    res.update(sampler.snapshot())
    return res


# ---------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="short name from bench/models.py or a HF repo id")
    ap.add_argument("--compression", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--work-dir", default=os.environ.get("AIRLLM_WORK", str(Path.home() / "airllm-work")))
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "local"))
    ap.add_argument("--delete-original", action="store_true",
                    help="let AirLLM delete original shards after splitting (needs an index.json)")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--stock", action="store_true", help="use AirLLM's stock class (Qwen keeps vision tower)")
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the result file name")
    args = ap.parse_args()

    from models import MODELS, airllm_class_for, build

    repo_id = MODELS.get(args.model, args.model)
    short = args.model if args.model in MODELS else repo_id.split("/")[-1]
    compression = None if args.compression == "none" else args.compression
    dtype = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{short}__{args.compression}" + (f"__{args.dtype}" if args.dtype != "auto" else "") + \
           (f"__{args.tag}" if args.tag else "")
    out_file = out_dir / f"{name}.json"

    layers_root = Path(args.work_dir) / "layers" / short
    split_dir = layers_root / ("splitted_model" + (f".{compression}" if compression else ""))

    result = {
        "model": short,
        "repo_id": repo_id,
        "compression": args.compression,
        "dtype_arg": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "airllm_class": None,
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": env_info(),
        "phases": {},
        "runs": {},
    }

    def save():
        out_file.write_text(json.dumps(result, indent=2))

    sampler = Sampler()
    sampler.start()
    stats = IOStats()
    phase = "download"
    try:
        import huggingface_hub
        from huggingface_hub import snapshot_download
        from airllm.utils import find_or_create_local_splitted_path

        cls = airllm_class_for(repo_id, args.stock)
        result["airllm_class"] = cls.__name__ if cls else "AutoModel"
        print(f"=== {repo_id} | compression={args.compression} | class={result['airllm_class']}", flush=True)

        # -- download --------------------------------------------------------------------------
        split_preexisting = split_dir.exists() and any(split_dir.glob("*.done"))
        t = time.perf_counter()
        if not split_preexisting:
            snapshot_download(repo_id)
        result["phases"]["download_s"] = round(time.perf_counter() - t, 2)
        huggingface_hub.constants.HF_HUB_OFFLINE = True  # no network from here on

        # -- convert (split) ----------------------------------------------------------------------
        phase = "convert"
        sampler.reset()
        if cls is None:
            from airllm.auto_model import AutoModel as _AM
            import importlib
            mod, cname = _AM.get_module_class(repo_id)
            cls_for_names = getattr(importlib.import_module(mod), cname)
        else:
            cls_for_names = cls
        probe = cls_for_names.__new__(cls_for_names)
        probe.set_layer_names_dict()
        t = time.perf_counter()
        find_or_create_local_splitted_path(repo_id, str(layers_root), compression=compression,
                                           layer_names=probe.layer_names_dict,
                                           delete_original=args.delete_original)
        result["phases"]["convert_s"] = round(time.perf_counter() - t, 2)
        result["phases"]["split_preexisting"] = split_preexisting
        result["phases"]["split_size_gb"] = dir_size_gb(split_dir)
        result["phases"]["convert_mem"] = sampler.snapshot()
        save()

        # -- init ---------------------------------------------------------------------------------
        phase = "init"
        sampler.reset()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        model = build(repo_id, stock=args.stock, layer_shards_saving_path=str(layers_root),
                      compression=compression, dtype=dtype, prefetching=not args.no_prefetch,
                      delete_original=args.delete_original)
        torch.cuda.synchronize()
        result["phases"]["init_s"] = round(time.perf_counter() - t, 2)
        result["phases"]["init_mem"] = sampler.snapshot()
        result["phases"]["init_torch_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        result["running_dtype"] = str(model.running_dtype)
        result["prefetching"] = bool(model.prefetching)
        result["n_streamed_modules"] = len(getattr(model, "_streamed_indices", []))
        instrument(model, stats)
        save()

        enc, used_template, _ = build_inputs(model.tokenizer, args.prompt)
        result["chat_template"] = used_template

        # -- generate -----------------------------------------------------------------------------
        phase = "generate_cold"
        n_evicted = evict_page_cache(split_dir)
        print(f"evicted {n_evicted} split files from the page cache", flush=True)
        result["runs"]["cold"] = timed_generate(model, enc, args.max_new_tokens, sampler, stats)
        print(json.dumps({k: v for k, v in result["runs"]["cold"].items() if k != "per_token_s"}, indent=1), flush=True)
        save()
        if not args.skip_warm:
            phase = "generate_warm"
            result["runs"]["warm"] = timed_generate(model, enc, args.max_new_tokens, sampler, stats)
            print(json.dumps({k: v for k, v in result["runs"]["warm"].items() if k != "per_token_s"}, indent=1), flush=True)
        result["status"] = "ok"
    except BaseException as e:  # noqa: BLE001 - record every failure, including OOM / Ctrl-C
        result["status"] = "error"
        result["error_phase"] = phase
        result["error"] = f"{type(e).__name__}: {e}"[:2000]
        result["traceback"] = traceback.format_exc()[-6000:]
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        sampler.stop()
        result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
        print(f"wrote {out_file} status={result['status']}", flush=True)
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
