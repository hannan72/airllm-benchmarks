"""ZeroGPU Space: the airllm-benchmarks models deployed directly with transformers (no AirLLM).

ZeroGPU runs PyTorch code inside a Gradio Space, not vLLM, so this is plain ``generate()`` with
the whole bf16 model on one GPU. ZeroGPU only intercepts CUDA placement done while the app module
is imported, so ONE model is loaded at start-up, chosen by the Space variable ``BENCH_MODEL``
(a short name from model_list.py; changing the variable restarts the Space):
  1. start-up (no GPU quota): download the checkpoint and place it on "cuda" (emulated);
  2. Run: one @spaces.GPU call does a warm-up, then time-to-first-token and decode tok/s
     (median of N runs). Models above 48 GB in bf16 use the full GPU (size="xlarge", 2x quota).
Results are JSON in the same format as bench/run_direct.py.

Build with ``python scripts/make_zerogpu_space.py`` (copies the bench modules next to this file).
"""

import spaces  # noqa: I001  (must be imported before torch on ZeroGPU)

import json
import os
import subprocess
import time
import traceback

import gradio as gr
import torch  # noqa: F401

from model_list import MODELS
from run_direct import download, gpu_info, hf_load, hf_measure, host_info

XLARGE = {"gemma-4-26B-A4B-it", "Qwen3.6-27B", "Qwen3.8-27B"}  # bf16 checkpoint > 48 GB
SHORT = os.environ.get("BENCH_MODEL", "gemma-4-E4B-it")
SIZE = "xlarge" if SHORT in XLARGE else "large"


def sysinfo():
    cmd = ("df -h / /tmp 2>/dev/null; free -g; cat /sys/fs/cgroup/memory.max 2>/dev/null; nproc; python -V; "
           "pip list 2>/dev/null | grep -iE '^(torch|transformers|spaces|gradio|flash-linear-attention|fla-core|"
           "compressed-tensors|triton) '")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


LOAD = {"model": SHORT, "repo_id": MODELS[SHORT], "engine": "hf", "weights": "bf16", "phases": {}}
MODEL = TOK = None
try:
    _path, LOAD["phases"]["download_s"], LOAD["phases"]["checkpoint_gb"] = download(MODELS[SHORT])
    _t = time.perf_counter()
    MODEL, TOK, LOAD["hf_class"] = hf_load(_path, dtype="bfloat16", device_map=None)
    MODEL.to("cuda")  # emulated here; ZeroGPU moves the weights onto the real GPU for each call
    LOAD["phases"]["load_s"] = round(time.perf_counter() - _t, 2)
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    LOAD.update(status="error", error_phase="load", error=f"{type(e).__name__}: {e}"[:2000],
                traceback=traceback.format_exc()[-4000:])
print("loaded:", json.dumps({k: v for k, v in LOAD.items() if k != "traceback"}), flush=True)


def _measure(n_tokens, repeats):
    t0 = time.perf_counter()
    info = gpu_info()
    res = hf_measure(MODEL, TOK, n_tokens=int(n_tokens), repeats=int(repeats))
    res["gpu_function_s"] = round(time.perf_counter() - t0, 2)
    return info, res


@spaces.GPU(duration=60)
def measure_large(n_tokens, repeats):
    return _measure(n_tokens, repeats)


@spaces.GPU(duration=60, size="xlarge")
def measure_xlarge(n_tokens, repeats):
    return _measure(n_tokens, repeats)


def run(n_tokens, repeats):
    result = dict(LOAD, started=time.strftime("%Y-%m-%dT%H:%M:%S"), runs={})
    if MODEL is not None:
        try:
            t = time.perf_counter()
            info, single = (measure_xlarge if SIZE == "xlarge" else measure_large)(n_tokens, repeats)
            single["gpu_call_wall_s"] = round(time.perf_counter() - t, 2)
            result["chat_template"] = single.pop("chat_template")
            result["runs"]["single"] = single
            result["env"] = {**host_info(), **info, "hostname_tag": "zerogpu", "zerogpu_size": SIZE}
            result["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            result.update(status="error", error_phase="generate", error=f"{type(e).__name__}: {e}"[:2000],
                          traceback=traceback.format_exc()[-4000:])
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("@@@RESULT", json.dumps(result), flush=True)
    return json.dumps(result, indent=1)


with gr.Blocks(title="AirLLM benchmarks: ZeroGPU direct") as demo:
    gr.Markdown(f"## Direct (no AirLLM) transformers decode benchmark on ZeroGPU\n"
                f"Loaded: **{SHORT}** ({SIZE} GPU) — status: {LOAD.get('status', 'ready')}. "
                "Set the Space variable `BENCH_MODEL` to switch. Greedy, chat template with thinking off.")
    with gr.Row():
        n_tokens = gr.Number(64, precision=0, label="decode tokens")
        repeats = gr.Number(2, precision=0, label="repeats")
    go = gr.Button("Run", variant="primary")
    out = gr.Textbox(lines=16, max_lines=40, label="result (JSON)", elem_id="result")
    go.click(run, [n_tokens, repeats], out, api_name="run")
    with gr.Accordion("System (start-up)", open=False):
        gr.Textbox(sysinfo() + "\n" + json.dumps(LOAD, indent=1), lines=20, label="system / load")

demo.queue(max_size=4).launch()
