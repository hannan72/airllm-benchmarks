#!/usr/bin/env python
"""Build kaggle/airllm_t4_benchmark.ipynb.

The repo is private, so instead of cloning it the notebook embeds the bench/*.py files with
%%writefile cells. Import the generated .ipynb on Kaggle (File > Import notebook), choose
Accelerator "GPU T4 x2", turn Internet on, then "Save Version > Save & Run All".
The final cell prints every result JSON, so results can be read from the version log.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_FILES = ["airllm_compat.py", "airllm_gemma4.py", "models.py", "lazy_split.py", "run_bench.py", "run_matrix.py"]
ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="kaggle", help="plan name in bench/run_matrix.py")
ap.add_argument("--out", default="kaggle/airllm_t4_benchmark.ipynb")
ARGS = ap.parse_args()


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = [
    md("# AirLLM 4.0 benchmark on Kaggle T4\n\n"
       "Gemma 4 (E2B-qat, E4B, 12B, 26B-A4B) and Qwen 3.5-9B / 3.6-27B / 3.8-27B, streamed layer by layer "
       "with AirLLM. Generated from `scripts/make_kaggle_notebook.py` in the airllm-benchmarks repo."),
    code("!nvidia-smi\n!df -h / /kaggle/working /tmp\n!free -g\n!nproc"),
    code("%%capture pip_log\n"
         "!pip install -q 'airllm==4.0.0' 'transformers>=5.8,<6' accelerate safetensors bitsandbytes "
         "compressed-tensors 'huggingface_hub[hf_xet]' psutil nvidia-ml-py sentencepiece"),
    code("import torch, transformers, importlib.metadata as md\n"
         "print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))\n"
         "print('transformers', transformers.__version__, 'airllm', md.version('airllm'), "
         "'bnb', md.version('bitsandbytes'), 'ct', md.version('compressed-tensors'))\n"
         "print('bf16 native:', torch.cuda.is_bf16_supported(including_emulation=False))"),
    code("import os\nos.makedirs('/kaggle/working/airllm-benchmarks/bench', exist_ok=True)"),
]
for name in BENCH_FILES:
    src = (ROOT / "bench" / name).read_text()
    cells.append(code(f"%%writefile /kaggle/working/airllm-benchmarks/bench/{name}\n{src}"))

cells += [
    code("# disk / read speed of the scratch disk used for the split layers\n"
         "!dd if=/dev/zero of=/tmp/ddtest bs=64M count=32 oflag=direct 2>&1 | tail -1\n"
         "!dd if=/tmp/ddtest of=/dev/null bs=64M iflag=direct 2>&1 | tail -1\n"
         "!rm -f /tmp/ddtest"),
    code("%cd /kaggle/working/airllm-benchmarks\n"
         "import os, shutil\n"
         "# keep checkpoints and split layers off /kaggle/working (20 GB output cap): use the roomiest scratch dir\n"
         "cands = [c for c in ['/kaggle/tmp', '/tmp', '/root'] if os.path.isdir(c)]\n"
         "SCRATCH = max(cands, key=lambda c: shutil.disk_usage(c).free)\n"
         "print('scratch:', SCRATCH, round(shutil.disk_usage(SCRATCH).free / 1e9), 'GB free')\n"
         "os.environ['HF_HOME'] = f'{SCRATCH}/hf'\n"
         "os.environ['BENCH_HOST_TAG'] = 'kaggle-t4'\n"
         f"!python bench/run_matrix.py --plan {ARGS.plan} --out-dir results/kaggle-t4 --work-dir {{SCRATCH}}/airllm-work "
         "--cleanup --no-prefetch --no-wait-gpu --timeout-min 300"),
    code("import json, glob\n"
         "for f in sorted(glob.glob('/kaggle/working/airllm-benchmarks/results/kaggle-t4/*.json')):\n"
         "    r = json.load(open(f))\n"
         "    c, w = r['runs'].get('cold', {}), r['runs'].get('warm', {})\n"
         "    print(f\"{r['model']:30s} {r['compression']:5s} {r['status']:5s} \"\n"
         "          f\"cold {c.get('decode_tok_per_s')} tok/s ttft {c.get('ttft_s')}s | \"\n"
         "          f\"warm {w.get('decode_tok_per_s')} tok/s | vram {c.get('peak_gpu_process_gb')} GB \"\n"
         "          f\"| {r.get('error', '')[:120]}\")"),
    code("# full JSON results (read from the version log; same files are in the notebook output)\n"
         "import json, glob\n"
         "for f in sorted(glob.glob('/kaggle/working/airllm-benchmarks/results/kaggle-t4/*.json')):\n"
         "    print('@@@RESULT', f.split('/')[-1])\n"
         "    print(json.dumps(json.load(open(f))))\n"
         "    print('@@@END')"),
    code("!tail -n 30 /kaggle/working/airllm-benchmarks/logs/*.log"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / ARGS.out
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, {len(cells)} cells)")
