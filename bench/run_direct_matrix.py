#!/usr/bin/env python
"""Run bench/run_direct.py for every model on one big-GPU host, one subprocess per model.

  --fit auto     bf16 when the checkpoint uses at most --fit-frac of VRAM, otherwise vLLM's
                 on-the-fly fp8 (W8A16 Marlin on Ampere, W8A8 on Hopper); hf engine: skip
  --cleanup      delete each model's HF cache after its run (scratch disks are small)
  --budget-min   start no new model after this many minutes (paid GPU credits)

Example (Lightning AI H100):
  BENCH_HOST_TAG=lightning-h100 python bench/run_direct_matrix.py --out-dir results/lightning-h100 --cleanup
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
sys.path.insert(0, str(HERE))

from model_list import MODELS  # noqa: E402
from run_direct import result_name  # noqa: E402


def vram_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.splitlines()[0]) * 2**20 / 1e9


def checkpoint_gb(repo_id):
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9


def cache_dir(repo_id):
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--models", default="all", help="comma-separated short names, or 'all'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fit", default="auto", choices=["auto", "bf16", "fp8"])
    ap.add_argument("--fit-frac", type=float, default=0.75)
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--budget-min", type=float, default=0, help="0 = no limit")
    ap.add_argument("--timeout-min", type=float, default=40)
    ap.add_argument("extra", nargs="*", help="passed through to run_direct.py (after --)")
    args = ap.parse_args()

    models = list(MODELS) if args.models == "all" else args.models.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = HERE.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    vram = vram_gb()
    t_start = time.time()
    print(f"[matrix] {len(models)} models, engine={args.engine}, VRAM {vram:.1f} GB", flush=True)

    for short in models:
        repo_id = MODELS[short]
        elapsed = (time.time() - t_start) / 60
        if args.budget_min and elapsed > args.budget_min:
            print(f"[matrix] budget of {args.budget_min} min used ({elapsed:.0f} min): stopping before {short}")
            break
        size = checkpoint_gb(repo_id)
        quant = None
        if args.fit == "fp8" or (args.fit == "auto" and size > args.fit_frac * vram):
            if args.engine != "vllm":
                print(f"[matrix] skip {short}: {size:.1f} GB checkpoint does not fit {vram:.0f} GB in bf16")
                continue
            quant = "fp8"
        res_file = out_dir / f"{result_name(short, args.engine, quant)}.json"
        if res_file.exists() and json.loads(res_file.read_text()).get("status") == "ok":
            print(f"[matrix] done already: {res_file.name}")
            continue
        cmd = [sys.executable, str(HERE / "run_direct.py"), "--model", short, "--engine", args.engine,
               "--out-dir", str(out_dir), *args.extra]
        if quant:
            cmd += ["--quantization", quant]
        log = log_dir / f"direct_{res_file.stem}.log"
        print(f"[matrix] {time.strftime('%H:%M:%S')} {short}: {size:.1f} GB checkpoint -> "
              f"{args.engine} {quant or 'bf16'} (log {log.name})", flush=True)
        t = time.time()
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    fh.write(line)
                    if any(k in line for k in ("decode_tok_per_s", "output_tok_per_s", "ttft_s\"", "wrote ",
                                               "Error", "error", "Traceback")):
                        print("   ", line.rstrip()[:300], flush=True)
                    if time.time() - t > args.timeout_min * 60:
                        proc.kill()
                        print(f"[matrix] timeout after {args.timeout_min} min", flush=True)
                        break
            finally:
                proc.wait()
        print(f"[matrix] {short} finished in {(time.time() - t) / 60:.1f} min (exit {proc.returncode})", flush=True)
        if args.cleanup:
            shutil.rmtree(cache_dir(repo_id), ignore_errors=True)
            print(f"[matrix] {shutil.disk_usage(cache_dir(repo_id).parent).free / 1e9:.0f} GB free after cleanup")
    print(f"[matrix] all done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
