"""Direct deployment (vLLM, no AirLLM) on Modal's H100 / A100 GPUs.

Modal (modal.com) bills per second; its free Starter plan includes $30 of compute per month, which
covers every model on H100 80 GB, A100 80 GB and A100 40 GB (GPU time is a few dollars per GPU type).

  1. download  CPU containers put the checkpoints in a Modal Volume, so no GPU time is spent on it
  2. bench     one GPU container per (GPU, model), all models of a GPU type in parallel, runs
               bench/run_direct.py on the Volume copy: bf16 when the checkpoint is at most
               --fit-frac of VRAM, otherwise vLLM's on-the-fly fp8 (26-27B on the 40 GB A100)
  3. results   saved locally: results/modal-<gpu>/<model>__vllm-<weights>.json, logs/modal_*.log

  pip install modal && modal setup                           # once: sign in, token in ~/.modal.toml
  modal run bench/modal_direct.py                            # H100 first ...
  modal run bench/modal_direct.py --gpus a100-80,a100-40     # ... then the A100s
  modal volume delete airllm-bench-hf                        # drop the ~220 GB of checkpoints after

Results that already have status "ok" are skipped, so a run can be repeated after a failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_list import MODELS  # noqa: E402

VLLM = "0.34.0"
# tag -> Modal GPU type. "H100!" pins an H100: plain "H100" may be upgraded to an H200 for free.
GPUS = {"h100": "H100!", "a100-80": "A100-80GB", "a100-40": "A100-40GB"}
USD_PER_GPU_S = {"h100": 0.001097, "a100-80": 0.000694, "a100-40": 0.000583}  # modal.com/pricing, 2026-11
CACHE = "/cache"

app = modal.App("airllm-bench-direct")
vol = modal.Volume.from_name("airllm-bench-hf", create_if_missing=True)
image = (
    # devel image: FlashInfer JIT-compiles some kernels with nvcc; torch 2.14 wheels are CUDA 13.0
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM}", "nvidia-ml-py", "psutil", "huggingface_hub[hf_xet]", "ninja")
    .env({"HF_HOME": f"{CACHE}/hf", "HF_XET_CACHE": "/tmp/xet", "HF_XET_HIGH_PERFORMANCE": "1",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    # same directory as this file in the container (/root), so the imports work in both places
    .add_local_file(HERE / "measure.py", "/root/measure.py")
    .add_local_file(HERE / "model_list.py", "/root/model_list.py")
    .add_local_file(HERE / "run_direct.py", "/root/run_direct.py")
)


@app.function(image=image, volumes={CACHE: vol}, cpu=4, memory=8192, timeout=3600)
def download(short):
    from huggingface_hub import snapshot_download
    from run_direct import ALLOW

    t = time.perf_counter()
    path = Path(snapshot_download(MODELS[short], allow_patterns=ALLOW))
    vol.commit()
    gb = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9
    return {"model": short, "checkpoint_gb": round(gb, 3), "download_s": round(time.perf_counter() - t, 1)}


def _bench(short, tag, fit_frac, args):
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    vol.reload()
    path = Path(snapshot_download(MODELS[short], local_files_only=True))
    ckpt_gb = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader,nounits"], text=True).splitlines()[0]
    gpu_name, vram_mib, driver = (s.strip() for s in smi.split(","))
    vram_gb = int(vram_mib) * 2**20 / 1e9
    quant = "fp8" if ckpt_gb > fit_frac * vram_gb else None
    cmd = [sys.executable, "/root/run_direct.py", "--model", short, "--engine", "vllm", "--out-dir", "/tmp/out",
           *args, *(["--quantization", quant] if quant else [])]
    env = dict(os.environ, BENCH_HOST_TAG=f"modal-{tag}", HF_HUB_OFFLINE="1")
    print(f"[{tag} {short}] {gpu_name} {vram_gb:.1f} GB, checkpoint {ckpt_gb:.1f} GB -> {quant or 'bf16'}", flush=True)
    lines = []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        lines.append(line)
        print(f"[{tag} {short}] {line}", end="", flush=True)
    proc.wait()
    files = list(Path("/tmp/out").glob("*.json"))
    if files:
        result = json.loads(files[0].read_text())
    else:
        result = {"model": short, "repo_id": MODELS[short], "engine": "vllm", "weights": quant or "bf16",
                  "status": "error", "error_phase": "process", "phases": {}, "runs": {},
                  "error": f"run_direct.py exited {proc.returncode} without a result: {''.join(lines[-5:])[-1500:]}"}
    wall = time.perf_counter() - t0
    result.setdefault("env", {}).update(modal_gpu=GPUS[tag], driver=driver, hostname_tag=f"modal-{tag}")
    result["phases"]["container_s"] = round(wall, 1)
    result["phases"]["gpu_usd_est"] = round(wall * USD_PER_GPU_S[tag], 3)
    return {"result": result, "log": "".join(lines)}


GPU_FN = dict(image=image, volumes={CACHE: vol}, cpu=8, memory=16384, timeout=45 * 60)


@app.function(gpu=GPUS["h100"], **GPU_FN)
def bench_h100(short, fit_frac, args):
    return _bench(short, "h100", fit_frac, args)


@app.function(gpu=GPUS["a100-80"], **GPU_FN)
def bench_a100_80(short, fit_frac, args):
    return _bench(short, "a100-80", fit_frac, args)


@app.function(gpu=GPUS["a100-40"], **GPU_FN)
def bench_a100_40(short, fit_frac, args):
    return _bench(short, "a100-40", fit_frac, args)


BENCH = {"h100": bench_h100, "a100-80": bench_a100_80, "a100-40": bench_a100_40}


def _done(out_dir, short):
    for f in out_dir.glob(f"{short}__vllm-*.json"):
        if json.loads(f.read_text()).get("status") == "ok":
            return True
    return False


@app.local_entrypoint()
def main(gpus: str = "h100", models: str = "all", fit_frac: float = 0.75, tokens: int = 256,
         repeats: int = 3, batch: int = 32):
    tags = gpus.split(",")
    unknown = [t for t in tags if t not in GPUS]
    if unknown:
        raise SystemExit(f"unknown GPU tag(s) {unknown}; choose from {list(GPUS)}")
    shorts = list(MODELS) if models == "all" else models.split(",")
    args = ["--tokens", str(tokens), "--repeats", str(repeats), "--batch", str(batch)]
    root = HERE.parent
    (root / "logs").mkdir(exist_ok=True)

    todo = {t: [s for s in shorts if not _done(root / "results" / f"modal-{t}", s)] for t in tags}
    need = sorted({s for ss in todo.values() for s in ss}, key=shorts.index)
    if need:
        print(f"[modal] downloading {len(need)} checkpoints into the Volume (CPU containers)", flush=True)
        for d in download.map(need, order_outputs=False):
            print(f"[modal] {d['model']}: {d['checkpoint_gb']:.1f} GB in {d['download_s']:.0f} s", flush=True)

    total_usd = 0.0
    for tag in tags:
        out_dir = root / "results" / f"modal-{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not todo[tag]:
            print(f"[modal] {tag}: all done already")
            continue
        print(f"[modal] {tag}: {len(todo[tag])} models in parallel on {GPUS[tag]}", flush=True)
        outs = BENCH[tag].map(todo[tag], kwargs={"fit_frac": fit_frac, "args": args}, return_exceptions=True)
        for short, out in zip(todo[tag], outs):
            if isinstance(out, BaseException):  # container lost (OOM kill, timeout, preemption)
                out = {"log": "", "result": {
                    "model": short, "repo_id": MODELS[short], "engine": "vllm", "weights": "?", "status": "error",
                    "error_phase": "container", "error": f"{type(out).__name__}: {out}"[:2000], "phases": {},
                    "runs": {}, "env": {"hostname_tag": f"modal-{tag}", "modal_gpu": GPUS[tag]}}}
            r = out["result"]
            name = f"{short}__vllm-{r['weights']}"
            (out_dir / f"{name}.json").write_text(json.dumps(r, indent=2))
            (root / "logs" / f"modal_{tag}_{name}.log").write_text(out["log"])
            usd = r["phases"].get("gpu_usd_est", 0)
            total_usd += usd
            s = r["runs"].get("single", {})
            b = r["runs"].get("batch", {})
            print(f"[modal] {tag} {name}: {r['status']} decode {s.get('decode_tok_per_s')} tok/s, "
                  f"ttft {s.get('ttft_s')} s, batch {b.get('output_tok_per_s')} tok/s, ~${usd:.2f} "
                  f"{r.get('error', '')[:200]}", flush=True)
    print(f"[modal] done; GPU time ~${total_usd:.2f} (CPU, memory and downloads extra, small)")
