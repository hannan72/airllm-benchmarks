"""Measurement helpers shared by the AirLLM runner (run_bench.py) and the direct-deployment
runner (run_direct.py): memory sampler, per-token timer, chat-template prompt builder."""

from __future__ import annotations

import os
import threading
import time

import psutil
import torch

DEFAULT_PROMPT = "Explain how a CPU cache works in one short paragraph."


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
        self.other_gpu_pids = set()

    def gpu(self):
        if self.nvml is None:
            return 0, 0
        dev = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        mine = 0
        try:
            for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if p.pid == os.getpid():
                    mine = p.usedGpuMemory or 0
                elif (p.usedGpuMemory or 0) > 256 * 2**20:
                    self.other_gpu_pids.add(p.pid)  # another workload shared the GPU
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
            "other_gpu_pids_seen": sorted(self.other_gpu_pids),
        }


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
