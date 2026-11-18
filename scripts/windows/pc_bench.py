"""Helpers for scripts/windows/run_pc.ps1 (any OS works, but it is meant for a Windows PC).

  preflight --out FILE   GPU, torch build and the kernels AirLLM needs (fp16 / bf16 matmul,
                         bitsandbytes NF4); writes JSON and exits 1 if CUDA is unusable
  summary DIR            one line per result JSON in DIR
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path


def preflight(out):
    import psutil
    import torch

    info = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    try:
        from huggingface_hub import constants

        p = Path(constants.HF_HUB_CACHE)
        while not p.exists() and p != p.parent:
            p = p.parent
        info["hf_cache"] = constants.HF_HUB_CACHE
        info["hf_cache_disk_free_gb"] = round(shutil.disk_usage(p).free / 1e9, 1)
    except Exception as e:  # noqa: BLE001
        info["hf_cache"] = f"? {e}"
    checks = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_vram_gb=round(p.total_memory / 1e9, 2), gpu_cc=f"{p.major}.{p.minor}",
                    arch_list=torch.cuda.get_arch_list())
        free, _ = torch.cuda.mem_get_info()
        info["gpu_free_gb"] = round(free / 1e9, 2)  # the desktop may hold some VRAM on a display GPU

        def check(name, fn):
            try:
                checks[name] = fn()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                checks[name] = f"FAIL {type(e).__name__}: {e}"[:300]

        def matmul(dt):
            a, b = torch.randn(512, 512, device="cuda"), torch.randn(512, 512, device="cuda")
            ref = a @ b
            err = ((a.to(dt) @ b.to(dt)).float() - ref).abs().max().item() / ref.abs().max().item()
            return "ok" if err < 0.05 else f"FAIL wrong result (rel err {err:.3f})"

        def nf4(dt):
            import bitsandbytes as bnb

            lin = bnb.nn.Linear4bit(512, 512, bias=False, compute_dtype=dt, quant_type="nf4").cuda()
            y = lin(torch.randn(2, 512, device="cuda", dtype=dt))
            return "ok" if torch.isfinite(y).all().item() else "FAIL non-finite output"

        check("matmul_fp16", lambda: matmul(torch.float16))
        check("matmul_bf16", lambda: matmul(torch.bfloat16))
        check("nf4_fp16", lambda: nf4(torch.float16))
        check("nf4_bf16", lambda: nf4(torch.bfloat16))
    info["checks"] = checks
    # AirLLM runs in the checkpoint's dtype (bf16); fall back to fp16 only if bf16 kernels fail
    bf16_ok = checks.get("matmul_bf16") == "ok" and checks.get("nf4_bf16") == "ok"
    fp16_ok = checks.get("matmul_fp16") == "ok" and checks.get("nf4_fp16") == "ok"
    info["dtype"] = "auto" if bf16_ok else "fp16" if fp16_ok else None
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0 if info["dtype"] else 1


def summary(d):
    rows = []
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        w = r.get("runs", {}).get("warm", {})
        c = r.get("runs", {}).get("cold", {})
        rows.append(f"{f.stem:45s} {r.get('status', '?'):6s} warm {w.get('decode_tok_per_s', '-'):>7} tok/s  "
                    f"cold {c.get('decode_tok_per_s', '-'):>7} tok/s  ttft {w.get('ttft_s', '-'):>6} s  "
                    f"{(r.get('error') or '')[:120]}")
    print("\n".join(rows) if rows else f"no results in {d}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preflight" and len(sys.argv) == 4 and sys.argv[2] == "--out":
        sys.exit(preflight(sys.argv[3]))
    if cmd == "summary" and len(sys.argv) == 3:
        sys.exit(summary(sys.argv[2]))
    sys.exit(__doc__)
