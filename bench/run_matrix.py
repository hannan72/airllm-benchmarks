#!/usr/bin/env python
"""Run the whole benchmark matrix sequentially, one model at a time.

* each (model, compression) runs in its own process, so memory peaks are not shared
* waits until no other process is using the GPU before each job (shared workstation)
* prefetches the next model's checkpoint in the background when disk space allows
* deletes a model's split layers and HF cache once all its jobs are done (--cleanup)
* resumable: a job whose JSON already has status "ok" is skipped
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

# original checkpoint size in GB (safetensors), used for disk planning
SIZE_GB = {
    "gemma-4-E2B-it-qat-mobile-ct": 2.7,
    "gemma-4-E4B-it": 16.0,
    "Qwen3.5-9B": 19.3,
    "gemma-4-12B-it": 23.9,
    "gemma-4-26B-A4B-it": 51.6,
    "Qwen3.6-27B": 55.6,
    "Qwen3.8-27B": 55.6,
}

# (model, [compressions], max_new_tokens, extra run_bench args)
PLANS = {
    # 8 GB laptop GPU, ~150 GB free disk: keep originals, run bf16 and 4-bit for every model
    "local": [
        ("gemma-4-E2B-it-qat-mobile-ct", ["none"], 32, []),  # already int2/int4/int8 (compressed-tensors)
        ("gemma-4-E4B-it", ["4bit", "none"], 32, []),
        ("Qwen3.5-9B", ["4bit", "none"], 32, []),
        ("gemma-4-12B-it", ["4bit", "none"], 32, []),
        ("gemma-4-26B-A4B-it", ["4bit", "none"], 16, []),
        ("Qwen3.6-27B", ["4bit", "none"], 16, []),
        ("Qwen3.8-27B", ["4bit", "none"], 16, []),
    ],
    # Kaggle T4: limited disk, so the >25B models stream shards and keep only the 4-bit split
    "kaggle": [
        ("gemma-4-E2B-it-qat-mobile-ct", ["none"], 32, []),
        ("gemma-4-E4B-it", ["4bit", "none"], 32, []),
        ("Qwen3.5-9B", ["4bit", "none"], 32, []),
        ("gemma-4-12B-it", ["4bit", "none"], 32, []),
        ("gemma-4-26B-A4B-it", ["4bit"], 16, ["--stream-download", "--delete-original"]),
        ("Qwen3.6-27B", ["4bit"], 16, ["--stream-download", "--delete-original"]),
        ("Qwen3.8-27B", ["4bit"], 16, ["--stream-download", "--delete-original"]),
    ],
}


def repo_of(model):
    sys.path.insert(0, str(HERE))
    from models import MODELS

    return MODELS[model]


def hf_cache_dir(model):
    from huggingface_hub import constants  # honours HF_HOME / HF_HUB_CACHE

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo_of(model).replace("/", "--"))


def free_gb(path="/"):
    return shutil.disk_usage(path).free / 1e9


def other_gpu_procs():
    """PIDs (other than ours) with a CUDA context, and device memory used in GB."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        procs = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != os.getpid()]
        used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
        return procs, used
    except Exception:  # noqa: BLE001
        return [], 0.0


def wait_for_idle_gpu(max_used_gb=1.0, poll_s=60):
    waited = 0
    while True:
        procs, used = other_gpu_procs()
        if not procs and used < max_used_gb:
            if waited:
                print(f"[matrix] GPU idle after waiting {waited // 60} min", flush=True)
            return
        if waited % 600 == 0:
            print(f"[matrix] GPU busy (pids={procs}, used={used:.1f} GB); waiting...", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def result_ok(out_dir, model, comp):
    f = out_dir / f"{model}__{comp}.json"
    if not f.exists():
        return False
    try:
        return json.loads(f.read_text()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def prefetch(model):
    log = open(ROOT / "logs" / f"prefetch_{model}.log", "w")
    hf = Path(PY).with_name("hf")
    cmd = [str(hf), "download", repo_of(model), "--quiet"] if hf.exists() else \
        [PY, "-c", f"from huggingface_hub import snapshot_download; snapshot_download('{repo_of(model)}')"]
    print(f"[matrix] prefetching {model} in background", flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "local-rtx2070"))
    ap.add_argument("--work-dir", default=str(Path.home() / "airllm-work"))
    ap.add_argument("--plan", default="local", choices=sorted(PLANS))
    ap.add_argument("--only", nargs="*", help="subset of model names")
    ap.add_argument("--cleanup", action="store_true", help="delete split layers + HF cache after each model")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--no-wait-gpu", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=240)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    plan = [p for p in PLANS[args.plan] if not args.only or p[0] in args.only]
    env = dict(os.environ, BENCH_HOST_TAG=os.environ.get("BENCH_HOST_TAG", out_dir.name),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    prefetcher = None
    for i, (model, comps, max_new, extra) in enumerate(plan):
        # background-download the next model if it fits alongside this one's peak
        if not args.no_prefetch and i + 1 < len(plan) and prefetcher is None:
            nxt = plan[i + 1][0]
            need_now = SIZE_GB[model] * (1 + 0.3 + 1) if not hf_cache_dir(model).exists() else SIZE_GB[model] * 1.3
            if free_gb() - need_now - SIZE_GB[nxt] > 15 and not all(result_ok(out_dir, nxt, c) for c in plan[i + 1][1]):
                prefetcher = prefetch(nxt)

        for comp in comps:
            if result_ok(out_dir, model, comp):
                print(f"[matrix] skip {model} {comp} (done)", flush=True)
                continue
            if not args.no_wait_gpu:
                wait_for_idle_gpu()
            print(f"[matrix] >>> {model} {comp}  free disk {free_gb():.0f} GB", flush=True)
            cmd = [PY, str(HERE / "run_bench.py"), "--model", model, "--compression", comp,
                   "--max-new-tokens", str(max_new), "--out-dir", str(out_dir), "--work-dir", args.work_dir,
                   *extra]
            with open(ROOT / "logs" / f"{out_dir.name}_{model}__{comp}.log", "w") as log:
                try:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                        timeout=args.timeout_min * 60).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            print(f"[matrix] <<< {model} {comp} rc={rc}", flush=True)

        if prefetcher is not None and prefetcher.poll() is not None:
            prefetcher = None

        if args.cleanup:
            shutil.rmtree(Path(args.work_dir) / "layers" / model, ignore_errors=True)
            # don't delete a cache that the prefetcher is still writing for the *next* model
            shutil.rmtree(hf_cache_dir(model), ignore_errors=True)
            print(f"[matrix] cleaned {model}; free disk {free_gb():.0f} GB", flush=True)

    if prefetcher is not None:
        prefetcher.wait()
    print("[matrix] done", flush=True)


if __name__ == "__main__":
    main()
