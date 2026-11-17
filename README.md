# AirLLM benchmarks: Gemma 4 and Qwen 3.x on small GPUs

Can [AirLLM](https://github.com/lyogavin/airllm) (v4.0.0, Sept 2026) run current open models on an
8 GB laptop GPU and on a free Kaggle T4, and how fast?

Models: `google/gemma-4-E2B-it-qat-mobile-ct`, `google/gemma-4-E4B-it`, `Qwen/Qwen3.5-9B`,
`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B`.

> Results: see [Results](#results) (filled in by `scripts/report.py` from `results/*/*.json`).

## How AirLLM works (and why it is slow)

AirLLM splits a checkpoint into one file per decoder layer. At inference time it builds the
transformers model on the `meta` device and hooks every layer so that, right before the layer runs,
its weights are read from disk, copied to the GPU, used, and dropped again. Only one or two layers
are ever on the GPU, so VRAM needs depend on the *largest layer*, not the model size.

The price: **every generated token streams the whole model from disk** (the KV cache stays on the
GPU, so prefill and decode each cost exactly one full pass). Speed is therefore bounded by
`model bytes / storage read bandwidth`, unless the OS page cache holds the layer files (then it is
RAM bandwidth). The optional `compression='4bit'` stores layers as bitsandbytes NF4 (~3.6x smaller
files) and dequantizes on the GPU at load time.

## Method

* `bench/run_bench.py` runs one (model, weight mode) configuration in a fresh process and records:
  * **download** time, **one-time convert** (split + optional NF4 compression) time and size,
  * **load** time (empty model + resident modules),
  * two greedy generations of the same chat prompt (*"Explain how a CPU cache works in one short
    paragraph."*, thinking disabled), 32 new tokens (16 for the >25B models):
    * **cold**: the layer files are first evicted from the page cache (`posix_fadvise(DONTNEED)`),
    * **warm**: run again immediately; whatever fits in RAM is now served from the page cache,
  * per-token timestamps (a `LogitsProcessor`), so **TTFT** (prefill + first token) and
    **s/token** (decode) are separate,
  * peak VRAM: NVML per-process (includes the CUDA context) and `torch.cuda.max_memory_reserved`,
  * peak host RSS, time spent reading layer files and moving/dequantizing them on the GPU.
* `bench/run_matrix.py` runs the whole matrix sequentially, waits for the GPU to be idle (shared
  workstation), and deletes each model's files when done.
* Text-only: vision / audio towers are not loaded for any model.

### What had to be fixed to run these models with AirLLM 4.0.0

AirLLM 4.0.0 supports Qwen 3.5 / 3.6 / 3.8 (`Qwen3_5ForConditionalGeneration`) natively. It has
**no Gemma 4 support**, and its compressed-tensors path breaks on transformers 5.2x:

| Problem | Fix (in this repo) |
|---|---|
| `AutoModel` builds the generic class, which expects `model.layers`; Gemma 4's decoder is at `model.language_model.layers` → "Could not instantiate the model on meta" | `bench/airllm_gemma4.py`: `AirLLMGemma4` layer map |
| Gemma 4's audio tower has no SDPA attention, so AirLLM falls back to eager attention for the whole model | build the model text-only (drop `audio_config` / `vision_config`) |
| Gemma 4 E2B/E4B Per-Layer-Embedding table (`embed_tokens_per_layer`, 4.7–5.6 GB) is not a decoder layer | keep it in host RAM, gather rows on the CPU (same idea as AirLLM's Flash-Next n-gram table); excluded from NF4 compression |
| KV-shared layers (last 18–20 layers of E2B/E4B) have no `k_proj`/`v_proj`, but the checkpoint ships those tensors → `AttributeError` | skip tensors that have no module |
| Gemma 4 reads `embed_tokens.weight[pad_id]` directly (outside the hooked call) → meta-tensor error when the embedding is streamed | keep the input embedding resident for PLE models with untied `lm_head` |
| transformers 5.2x: `CompressedTensorsHfQuantizer` needs `validate_environment()` before `preprocess_model()`; AirLLM skips it → `AttributeError: use_fp8_kernel` | `bench/airllm_compat.py` shim |
| compressed-tensors W4A8/W2A8 (Gemma 4 QAT "-ct"): AirLLM's decompress drops the static `input_scale`s, and loads `int-quantized` int8 weights as raw int8 | re-add + re-type activation scales, dequantize int8 weights |

## Reproduce

```bash
uv venv .venv --python 3.12 && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python bench/run_bench.py --model Qwen3.5-9B --compression 4bit   # one config
.venv/bin/python bench/run_matrix.py --plan local --cleanup                 # everything
.venv/bin/python scripts/report.py                                          # tables
```

Kaggle: `python scripts/make_kaggle_notebook.py`, import `kaggle/airllm_t4_benchmark.ipynb`, choose
GPU T4 x2 + Internet, "Save & Run All". The notebook embeds the `bench/` files (the repo is private).

## Results

_Pending._
