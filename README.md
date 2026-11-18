# AirLLM benchmarks: Gemma 4 and Qwen 3.x on small GPUs

Can [AirLLM](https://github.com/lyogavin/airllm) (v4.0.0, Sept 2026) run current open models on an
**8 GB laptop GPU** (RTX 2070 Max-Q) and on a **free Kaggle T4**, and how fast?

Models: `google/gemma-4-E2B-it-qat-mobile-ct`, `google/gemma-4-E4B-it`, `Qwen/Qwen3.5-9B`,
`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B`.

## TL;DR

* **Yes, it runs**: every model, including the 27B ones, fits in **≤ 4.7 GB of VRAM** and needs only
  2–10 GB of host RAM. That holds on both GPUs, since AirLLM keeps only one decoder layer on the GPU at a time.
* **Out of the box it is slow and partly broken**: 0.01–0.16 tokens/s (6–80 s per token).
  Gemma 4 is not supported (we wrote an adapter). AirLLM's own `4bit` mode **produces garbage for
  every bf16 Gemma 4 model**. Its splitter **cannot convert Gemma-4-26B-A4B** (one 49.9 GB shard).
* **Two small changes make it 2–38× faster** (5–38× whenever the layer files fit in free RAM):
  compress only the weight *matrices* to NF4 (not the
  norms/scalars), and skip AirLLM's `gc.collect()` after every layer. Tuned, the ≤ 12B models do
  **0.5–1.6 tok/s** and the 26–27B models **0.08–0.4 tok/s**, with correct output. The 27B speed
  depends on whether the 15 GB of 4-bit layer files fit in free RAM (page cache).
* Even tuned, AirLLM streams the **whole model (≈ 15 GB for a 4-bit 27B) for every token**, so it
  is a way to *run* a model that does not fit in VRAM, not a way to chat with it interactively.
  The MoE model (26B-A4B) gets no benefit from sparsity: all 128 experts are streamed per layer.

![decode speed](results/speed.png)

<!-- headline:start -->
| Model | GPU | Stock AirLLM 4.0 (best correct config) | Tuned (NF4 matrices only + no gc) | Speed-up | Peak VRAM (tuned) |
|---|---|---|---|---|---|
| gemma-4-E2B-it-qat-mobile-ct | RTX 2070 Max-Q (8 GB) | 6.76 s/token (0.148 tok/s), native QAT int2/4/8 | 1.11 s/token (0.900 tok/s), native QAT int2/4/8 | **×6** | 4.7 GB |
| gemma-4-E2B-it-qat-mobile-ct | Kaggle T4 (16 GB) | 10.89 s/token (0.092 tok/s), native QAT int2/4/8 | 1.29 s/token (0.777 tok/s), native QAT int2/4/8 | **×8** | 4.7 GB |
| gemma-4-E4B-it | RTX 2070 Max-Q (8 GB) | 7.74 s/token (0.129 tok/s), bf16 | 0.64 s/token (1.552 tok/s), NF4, matrices only | **×12** | 1.8 GB |
| gemma-4-E4B-it | Kaggle T4 (16 GB) | 12.07 s/token (0.083 tok/s), bf16 | 1.03 s/token (0.967 tok/s), NF4, matrices only | **×12** | 1.8 GB |
| Qwen3.5-9B | RTX 2070 Max-Q (8 GB) | 6.30 s/token (0.159 tok/s), NF4, all tensors | 0.95 s/token (1.054 tok/s), NF4, matrices only | **×7** | 2.9 GB |
| Qwen3.5-9B | Kaggle T4 (16 GB) | 10.90 s/token (0.092 tok/s), NF4, all tensors | 1.70 s/token (0.588 tok/s), NF4, matrices only | **×6** | 2.9 GB |
| gemma-4-12B-it | RTX 2070 Max-Q (8 GB) | 18.12 s/token (0.055 tok/s), bf16 | 1.17 s/token (0.852 tok/s), NF4, matrices only | **×15** | 2.7 GB |
| gemma-4-12B-it | Kaggle T4 (16 GB) | 79.80 s/token (0.013 tok/s), bf16 | 2.11 s/token (0.473 tok/s), NF4, matrices only | **×38** | 2.7 GB |
| gemma-4-26B-A4B-it | RTX 2070 Max-Q (8 GB) | 26.97 s/token (0.037 tok/s), bf16 | 2.65 s/token (0.378 tok/s), NF4, matrices only | **×10** | 4.4 GB |
| gemma-4-26B-A4B-it | Kaggle T4 (16 GB) | split fails (49.9 GB shard) | 5.01 s/token (0.200 tok/s), NF4, matrices only | – | 4.4 GB |
| Qwen3.6-27B | RTX 2070 Max-Q (8 GB) | 22.18 s/token (0.045 tok/s), NF4, all tensors | 12.32 s/token (0.081 tok/s), NF4, matrices only | **×2** | 3.7 GB |
| Qwen3.6-27B | Kaggle T4 (16 GB) | 24.54 s/token (0.041 tok/s), NF4, all tensors | 4.91 s/token (0.203 tok/s), NF4, matrices only | **×5** | 3.7 GB |
| Qwen3.8-27B | RTX 2070 Max-Q (8 GB) | 22.99 s/token (0.043 tok/s), NF4, all tensors | 12.31 s/token (0.081 tok/s), NF4, matrices only | **×2** | 3.7 GB |
| Qwen3.8-27B | Kaggle T4 (16 GB) | 24.47 s/token (0.041 tok/s), NF4, all tensors | 4.81 s/token (0.208 tok/s), NF4, matrices only | **×5** | 3.7 GB |
<!-- headline:end -->

Warm decode speed (second identical generation); "stock" = fastest stock AirLLM 4.0 configuration
whose output is correct (bf16 for Gemma 4, AirLLM 4-bit for Qwen). Full tables: [Results](#full-results).

## Findings

1. **Gemma 4 needs an adapter.** AirLLM 4.0.0's `AutoModel` falls back to a generic class that
   looks for `model.layers` and fails. `bench/airllm_gemma4.py` adds the layer map. It also handles the E2B/E4B
   Per-Layer-Embedding table (4.7–5.6 GB, kept in host RAM and gathered on the CPU), KV-shared
   layers, and the audio tower that would force eager attention (see the fixes table below).
2. **AirLLM's `compression='4bit'` breaks Gemma 4.** It NF4-quantizes *every* tensor in a layer,
   including RMSNorm weights and per-layer scalars. Gemma 4 multiplies by its norm weight directly
   (not `1 + w`), and the output degenerates ("pages page continues pagepsages…", " l de de de…"). Qwen 3.5
   survives it. Quantizing only ≥ 2-D tensors (what bitsandbytes/GPTQ/AWQ do anyway) fixes it at
   the same file size.
3. **The per-layer `gc.collect()` dominates small models.** AirLLM calls `clean_memory()`
   (`gc.collect()` + `malloc_trim` + `empty_cache`) after every layer. With a transformers model
   graph alive (≈ 435k Python objects for Gemma-4-E4B) one `gc.collect()` takes ≈ 106 ms, i.e. 4–7 s
   per token. It also stalls the prefetch thread on the GIL. Skipping it: E2B-qat 6.8 → 1.0 s/token,
   identical text; peak VRAM rises by ≤ 0.8 GB because the CUDA cache is no longer emptied.
4. **4-bit only pays off once the gc is gone.** AirLLM disables layer prefetching when compression is
   on, so stock 4-bit is barely faster than bf16 (E4B 6.7 vs 7.6 s/token) even though it reads 3.6×
   less data. Tuned 4-bit is the fastest configuration for every model.
5. **Speed ≈ bytes per token ÷ read bandwidth, and free RAM decides the bandwidth.** A 4-bit 27B
   streams ≈ 15 GB per token. On Kaggle (≈ 30 GB of free RAM) the layer files stay in the page cache
   after the first pass and load at ≈ 3.6 GB/s: 4.8–4.9 s/token. On the laptop only ≈ 14 GB was free
   (other applications), so every token reads from the NVMe through AirLLM's serial, un-prefetched
   4-bit loader at ≈ 1.3 GB/s: 12 s/token for the same model. The ≤ 12B models fit in the page
   cache on both machines, which is why they reach 0.5–1.6 tok/s.
6. **Big shards break AirLLM's splitter.** It loads whole checkpoint shards. Gemma-4-26B-A4B ships
   a 49.9 GB shard: locally it fails (`unable to mmap 49907246508 bytes … Cannot allocate memory`,
   31 GB RAM), and on Kaggle, streaming it filled the scratch quota and the filesystem went read-only.
   `bench/lazy_split.py` reads only one layer's byte ranges at a time. It works from a local file, or straight from
   the Hub over HTTP range requests, which never puts the checkpoint on disk. That made the 26B
   runnable on both machines; on Kaggle it split 50 GB from the Hub in 7 minutes.
7. **Kaggle T4 vs the laptop.** The GPU is never the bottleneck. In tuned mode the T4 run was
   1.2–1.9× slower than the laptop for the models that fit in both page caches (E4B 0.97 vs 1.55
   tok/s, 26B-A4B 0.20 vs 0.38): Kaggle's 4 vCPUs are slower. It was 2.6× *faster* for Qwen3.8-27B
   (0.21 vs 0.08 tok/s), whose layer files fit in Kaggle's free RAM but not the laptop's. Stock bf16 is very slow on Kaggle: layer reads ran at only
   ≈ 0.3 GB/s, probably because of prefetch's pinned-memory copies on 4 vCPUs.
   Gemma-4-12B bf16 takes 80 s/token there.
8. **Neither GPU has native bf16** (sm_75). PyTorch emulates it (2.7 vs 9 TFLOPS fp16 on the
   laptop), which hardly matters here because AirLLM is I/O-bound.

**Perspective.** E2B/E4B/9B/12B fit entirely in 8 GB of VRAM at 4-bit, so a normal runtime (e.g.
transformers + bitsandbytes or llama.cpp) would run them one to two orders of magnitude faster than
any AirLLM configuration here. That is a reasoned expectation, not something we measured. AirLLM is
worth it for the models that do *not* fit: the 26–27B models ran with ≈ 4 GB of VRAM.

## How AirLLM works (and why it is slow)

AirLLM splits a checkpoint into one file per decoder layer. At inference time it builds the
transformers model on the `meta` device and hooks every layer so that, right before the layer runs,
its weights are read from disk, copied to the GPU, used, and dropped again. Only one or two layers
are ever on the GPU, so VRAM needs depend on the *largest layer*, not the model size.

The price: **every generated token streams the whole model from disk** (the KV cache stays on the
GPU, so prefill and decode each cost exactly one full pass). Speed is therefore bounded by
`model bytes / storage read bandwidth`, unless the OS page cache holds the layer files (then it is
RAM bandwidth). The optional `compression='4bit'` stores layers as bitsandbytes NF4 (~3.6× smaller
files) and dequantizes on the GPU at load time.

## Method

* **Hardware.** Local: RTX 2070 Max-Q 8 GB (laptop, sm_75), i7-9750H, 31 GB RAM, Samsung 970 EVO
  Plus NVMe (3.1 GB/s direct read), torch 2.15 / transformers 5.23. Kaggle: Tesla T4 16 GB (one of
  the two used), 4 vCPU Xeon, 31 GB RAM, torch 2.11 / transformers 5.20. Both run AirLLM 4.0.0 and bitsandbytes 0.50.2.
* `bench/run_bench.py` runs one (model, weight mode) configuration in a fresh process and records:
  * **download** time, **one-time convert** (split + optional NF4 compression) time and size,
  * **load** time (empty model + resident modules),
  * two greedy generations of the same chat prompt (*"Explain how a CPU cache works in one short
    paragraph."*, thinking disabled): 32 new tokens, or 16 for the > 25B models.
    * **cold**: the layer files are first evicted from the page cache (`posix_fadvise(DONTNEED)`),
    * **warm**: run again immediately; whatever fits in RAM is now served from the page cache,
  * per-token timestamps (a `LogitsProcessor`), so **TTFT** (prefill + first token) and
    **s/token** (decode) are separate,
  * peak VRAM: NVML per-process (includes the CUDA context) and `torch.cuda.max_memory_reserved`,
  * peak host RSS, time spent reading layer files and moving/dequantizing them on the GPU,
  * other processes seen on the GPU during the run (the local GPU is shared; every local result
    reported here ran with the GPU to itself).
* Weight modes: `none` = bf16 layer files; `4bit` = AirLLM's NF4 on every tensor; `4bit-2d` = NF4 on
  ≥ 2-D tensors only; `:nogc` = AirLLM's per-layer `clean_memory()` disabled.
* `bench/run_matrix.py` runs a plan sequentially. It waits for 5 minutes of GPU idleness on the shared
  workstation and deletes each model's files when done. The Kaggle runs use the notebooks in `kaggle/`.
* Text-only: vision / audio towers are not loaded for any model. Qwen's stock class would keep its
  vision tower resident on the GPU; `--stock` runs it unmodified.

### What had to be fixed or worked around (AirLLM 4.0.0)

| Problem | Fix (in this repo) |
|---|---|
| `AutoModel` builds the generic class, which expects `model.layers`; Gemma 4's decoder is at `model.language_model.layers` → "Could not instantiate the model on meta" | `bench/airllm_gemma4.py`: `AirLLMGemma4` layer map |
| Gemma 4's audio tower has no SDPA attention, so AirLLM falls back to eager attention for the whole model | build the model text-only (drop `audio_config` / `vision_config`) |
| Gemma 4 E2B/E4B Per-Layer-Embedding table (`embed_tokens_per_layer`, 4.7–5.6 GB) is not a decoder layer | keep it in host RAM, gather rows on the CPU (same idea as AirLLM's Flash-Next n-gram table); excluded from NF4 compression |
| KV-shared layers (last 18–20 layers of E2B/E4B) have no `k_proj`/`v_proj`, but the checkpoint ships those tensors → `AttributeError` | skip tensors that have no module |
| Gemma 4 reads `embed_tokens.weight[pad_id]` directly (outside the hooked call) → meta-tensor error when the embedding is streamed | keep the input embedding resident for PLE models with untied `lm_head` |
| transformers 5.2x: `CompressedTensorsHfQuantizer` needs `validate_environment()` before `preprocess_model()`; AirLLM skips it → `AttributeError: use_fp8_kernel` | `bench/airllm_compat.py` shim |
| compressed-tensors W4A8/W2A8 (Gemma 4 QAT "-ct"): AirLLM's decompress drops the static `input_scale`s, and loads `int-quantized` int8 weights as raw int8 | re-add + re-type activation scales, dequantize int8 weights |
| `compression='4bit'` quantizes norms / scalars → garbage from Gemma 4 | `4bit-2d` mode (`AIRLLM_COMPRESS_MIN_DIM=2`) |
| per-layer `gc.collect()` costs ≈ 0.1 s × layers per token | `:nogc` mode (`AIRLLM_NO_GC=1`) |
| splitter loads whole shards; 49.9 GB shard of 26B-A4B fails (RAM) or overflows Kaggle's disk | `bench/lazy_split.py` (byte-range reads, local or from the Hub) |
| disk-space check *divides* by the 4-bit ratio (`size / 0.2813`): it demands 3.6× the checkpoint size (184 GB for a 27B) and refuses to split with less free | corrected `check_space` in `bench/airllm_compat.py` |

## Full results

`results/<host>/*.json` hold every run (per-token timings included for local runs); `results/SUMMARY.md` is
generated from them. Output = the first words of the warm generation; garbage output is AirLLM's
`4bit` mode on Gemma 4, see finding 2.

<!-- results:start -->
#### kaggle-t4

Tesla T4 (15.64 GB, sm_75, bf16 native: False) · Intel(R) Xeon(R) CPU @ 2.00GHz · 33.7 GB RAM · torch 2.11.0+cu130 (CUDA 13.0) · transformers 5.20.2 · airllm 4.0.0 · bitsandbytes 0.50.2

| Model | Params | Weights | Layer files on disk | One-time convert | Load | TTFT cold | s/token cold | s/token warm | tok/s warm | Peak VRAM (proc / torch) | Peak RAM | Output (first words) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-E2B-it-qat-mobile-ct | 5.5B (2B eff.) | native QAT int2/4/8 | 2.1 GB | 0.2 min | 26 s | 12.9 s | 10.85 | 10.89 | 0.092 | 4.73 / 4.60 GB | 10.1 GB | A CPU cache is a small, very fast memory located close to… |
| gemma-4-E2B-it-qat-mobile-ct | 5.5B (2B eff.) | native QAT int2/4/8 + no per-layer gc | 2.1 GB | 0.2 min | 26 s | 2.6 s | 1.28 | 1.29 | 0.777 | 4.73 / 4.60 GB | 10.3 GB | A CPU cache is a small, very fast memory located close to… |
| gemma-4-E4B-it | 8.0B (4B eff.) | NF4, all tensors (AirLLM 4bit) | 8.3 GB | 1.7 min | 4 s | 14.4 s | 11.47 | 11.43 | 0.087 | 1.80 / 1.67 GB | 1.9 GB | pages page continues pagepsagesegalesagesurécocesagesuragesurdemifiezages...each pagepezages_pagesizeagesagesurdemer |
| gemma-4-E4B-it | 8.0B (4B eff.) | bf16 | 15.0 GB | 1.9 min | 15 s | 26.2 s | 12.22 | 12.07 | 0.083 | 1.78 / 1.65 GB | 4.9 GB | A CPU cache is a small, extremely fast memory located close to… |
| gemma-4-E4B-it | 8.0B (4B eff.) | NF4, matrices only + no per-layer gc | 8.3 GB | 1.6 min | 8 s | 3.0 s | 1.04 | 1.03 | 0.967 | 1.80 / 1.67 GB | 1.9 GB | A CPU cache is a small, fast memory located near or on… |
| Qwen3.5-9B | 9.7B | NF4, all tensors (AirLLM 4bit) | 5.0 GB | 1.5 min | 2 s | 24.2 s | 10.94 | 10.90 | 0.092 | 2.87 / 2.73 GB | 2.4 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| Qwen3.5-9B | 9.7B | bf16 | 17.9 GB | 2.1 min | 2 s | 67.5 s | 52.13 | 67.71 | 0.015 | 2.26 / 2.13 GB | 6.3 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| Qwen3.5-9B | 9.7B | NF4, matrices only + no per-layer gc | 5.0 GB | 1.6 min | 2 s | 16.3 s | 1.71 | 1.70 | 0.588 | 2.91 / 2.78 GB | 2.4 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| gemma-4-12B-it | 12.0B | NF4, all tensors (AirLLM 4bit) | 6.7 GB | 2.2 min | 6 s | 29.2 s | 15.93 | 15.74 | 0.064 | 2.70 / 2.59 GB | 1.9 GB | चित्रों चित्रों चित्रों चित्रों चित्रों चित्रों februari februari februari februari februari februari… |
| gemma-4-12B-it | 12.0B | bf16 | 23.8 GB | 3.1 min | 19 s | 77.8 s | 66.23 | 79.80 | 0.013 | 2.68 / 2.54 GB | 5.3 GB | A CPU cache is a small, high-speed memory buffer located directly on… |
| gemma-4-12B-it | 12.0B | NF4, matrices only + no per-layer gc | 6.7 GB | 2.3 min | 6 s | 23.8 s | 2.63 | 2.11 | 0.473 | 2.72 / 2.59 GB | 1.9 GB | A CPU cache is a small, high-speed memory bank located directly on… |
| gemma-4-26B-A4B-it | 25.8B MoE (4B act.) | NF4, all tensors (AirLLM 4bit) | – | – | – | – | – | – | – | – | – | **failed** in `convert`: OSError: [Errno 30] Read-only file system: '/tmp/hf/hub/models--google--gemma-4-26B-A4B-it/blobs/1127684971bbc |
| gemma-4-26B-A4B-it | 25.8B MoE (4B act.) | NF4, matrices only + no per-layer gc | 14.2 GB | 7.1 min (lazy split, from Hub) | 12 s | 43.5 s | 18.03 | 5.01 | 0.200 | 4.41 / 4.27 GB | 10.1 GB | A CPU cache works as a high-speed, small-capacity memory buffer located |
| Qwen3.6-27B | 27.8B | NF4, all tensors (AirLLM 4bit) | 15.1 GB | 9.8 min | 12 s | 55.9 s | 26.41 | 24.54 | 0.041 | 4.38 / 4.25 GB | 2.6 GB | A CPU cache is a small, high-speed memory located close to the… |
| Qwen3.6-27B | 27.8B | NF4, matrices only + no per-layer gc | 15.1 GB | 10.6 min | 3 s | 61.2 s | 10.76 | 4.91 | 0.203 | 3.69 / 3.56 GB | 2.5 GB | A CPU cache is a small, high-speed memory component located close to… |
| Qwen3.8-27B | 27.8B | NF4, all tensors (AirLLM 4bit) | 15.1 GB | 7.5 min | 2 s | 68.7 s | 26.74 | 24.47 | 0.041 | 4.38 / 4.25 GB | 2.5 GB | A CPU cache is a small, high-speed memory located on or near… |
| Qwen3.8-27B | 27.8B | NF4, matrices only + no per-layer gc | 15.1 GB | 7.9 min | 3 s | 62.2 s | 11.24 | 4.81 | 0.208 | 3.69 / 3.56 GB | 2.5 GB | A CPU cache is a small, high-speed memory located on or near… |

I/O breakdown (cold run):

| Model | Weights | Streamed per token (GB) | Layer read time / token | Effective read GB/s (cold) | Host→GPU + dequant / token | Prefetch |
|---|---|---|---|---|---|---|
| gemma-4-E2B-it-qat-mobile-ct | native QAT int2/4/8 | 0.78 | 1.86 s | 0.42 | 1.06 s | on |
| gemma-4-E2B-it-qat-mobile-ct | native QAT int2/4/8 + no per-layer gc | 0.78 | 0.35 s | 2.24 | 0.85 s | on |
| gemma-4-E4B-it | NF4, all tensors (AirLLM 4bit) | 2.25 | 0.95 s | 2.38 | 0.07 s | off |
| gemma-4-E4B-it | bf16 | 8.00 | 11.24 s | 0.71 | 0.83 s | on |
| gemma-4-E4B-it | NF4, matrices only + no per-layer gc | 2.25 | 0.79 s | 2.85 | 0.10 s | off |
| Qwen3.5-9B | NF4, all tensors (AirLLM 4bit) | 5.04 | 1.96 s | 2.57 | 0.14 s | off |
| Qwen3.5-9B | bf16 | 17.91 | 50.26 s | 0.36 | 1.67 s | on |
| Qwen3.5-9B | NF4, matrices only + no per-layer gc | 5.04 | 1.78 s | 2.82 | 0.19 s | off |
| gemma-4-12B-it | NF4, all tensors (AirLLM 4bit) | 6.13 | 2.43 s | 2.52 | 0.08 s | off |
| gemma-4-12B-it | bf16 | 21.80 | 63.87 s | 0.34 | 2.08 s | on |
| gemma-4-12B-it | NF4, matrices only + no per-layer gc | 6.13 | 2.88 s | 2.13 | 0.12 s | off |
| gemma-4-26B-A4B-it | NF4, matrices only + no per-layer gc | 13.78 | 18.16 s | 0.76 | 0.99 s | off |
| Qwen3.6-27B | NF4, all tensors (AirLLM 4bit) | 15.13 | 8.68 s | 1.74 | 0.27 s | off |
| Qwen3.6-27B | NF4, matrices only + no per-layer gc | 15.13 | 13.01 s | 1.16 | 0.46 s | off |
| Qwen3.8-27B | NF4, all tensors (AirLLM 4bit) | 15.13 | 9.80 s | 1.54 | 0.27 s | off |
| Qwen3.8-27B | NF4, matrices only + no per-layer gc | 15.13 | 13.50 s | 1.12 | 0.46 s | off |

#### local-rtx2070

NVIDIA GeForce RTX 2070 with Max-Q Design (8.16 GB, sm_75, bf16 native: False) · Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz · 33.5 GB RAM · torch 2.15.0+cu132 (CUDA 13.2) · transformers 5.23.0 · airllm 4.0.0 · bitsandbytes 0.50.2

| Model | Params | Weights | Layer files on disk | One-time convert | Load | TTFT cold | s/token cold | s/token warm | tok/s warm | Peak VRAM (proc / torch) | Peak RAM | Output (first words) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-E2B-it-qat-mobile-ct | 5.5B (2B eff.) | native QAT int2/4/8 | 2.1 GB | cached | 14 s | 8.0 s | 6.81 | 6.76 | 0.148 | 4.72 / 4.60 GB | 10.1 GB | A CPU cache is a small, very fast memory located close to… |
| gemma-4-E2B-it-qat-mobile-ct | 5.5B (2B eff.) | native QAT int2/4/8 + no per-layer gc | 2.1 GB | 0.2 min | 13 s | 1.6 s | 1.01 | 1.11 | 0.900 | 4.72 / 4.60 GB | 9.9 GB | A CPU cache is a small, very fast memory located close to… |
| gemma-4-E4B-it | 8.0B (4B eff.) | NF4, all tensors (AirLLM 4bit) | 8.3 GB | 0.6 min | 4 s | 9.0 s | 6.69 | 6.74 | 0.148 | 1.79 / 1.67 GB | 1.8 GB | pages page continues pagepsagesegalesagesurécocesagesuragesurdemifiezages...each pagepezages_pagesizeagesagesurdemer |
| gemma-4-E4B-it | 8.0B (4B eff.) | bf16 | 15.0 GB | 0.7 min | 6 s | 10.6 s | 7.57 | 7.74 | 0.129 | 1.77 / 1.64 GB | 4.7 GB | A CPU cache is a small, extremely fast memory located close to… |
| gemma-4-E4B-it | 8.0B (4B eff.) | bf16 + no per-layer gc | 15.0 GB | 0.5 min | 6 s | 6.3 s | 1.96 | 2.01 | 0.499 | 1.77 / 1.64 GB | 4.7 GB | A CPU cache is a small, extremely fast memory located close to… |
| gemma-4-E4B-it | 8.0B (4B eff.) | NF4, matrices only + no per-layer gc | 8.3 GB | 0.5 min | 5 s | 2.6 s | 0.63 | 0.64 | 1.552 | 1.77 / 1.64 GB | 1.8 GB | A CPU cache is a small, fast memory located near or on… |
| Qwen3.5-9B | 9.7B | NF4, all tensors (AirLLM 4bit) | 5.0 GB | 0.4 min | 2 s | 9.4 s | 6.28 | 6.30 | 0.159 | 2.80 / 2.67 GB | 2.2 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| Qwen3.5-9B | 9.7B | bf16 | 17.9 GB | cached | 3 s | 16.0 s | 13.48 | 10.80 | 0.093 | 2.25 / 2.13 GB | 6.3 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| Qwen3.5-9B | 9.7B | bf16 + no per-layer gc | 17.9 GB | cached | 3 s | 11.5 s | 5.82 | 4.90 | 0.204 | 2.25 / 2.13 GB | 6.1 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| Qwen3.5-9B | 9.7B | NF4, matrices only + no per-layer gc | 5.0 GB | 0.4 min | 2 s | 4.6 s | 0.90 | 0.95 | 1.054 | 2.86 / 2.73 GB | 2.0 GB | A CPU cache is a small, high-speed memory layer situated between the… |
| gemma-4-12B-it | 12.0B | NF4, all tensors (AirLLM 4bit) | 6.7 GB | 0.5 min | 5 s | 13.3 s | 8.88 | 8.86 | 0.113 | 2.69 / 2.59 GB | 1.6 GB | चित्रों चित्रों चित्रों चित्रों चित्रों चित्रों februari februari februari februari februari februari… |
| gemma-4-12B-it | 12.0B | bf16 | 23.8 GB | 1.0 min | 8 s | 17.6 s | 17.43 | 18.12 | 0.055 | 2.67 / 2.54 GB | 4.9 GB | A CPU cache is a small, high-speed memory buffer located directly on… |
| gemma-4-12B-it | 12.0B | bf16 + no per-layer gc | 23.8 GB | cached | 6 s | 13.3 s | 11.83 | 12.48 | 0.080 | 2.67 / 2.54 GB | 5.3 GB | A CPU cache is a small, high-speed memory buffer located directly on… |
| gemma-4-12B-it | 12.0B | NF4, matrices only + no per-layer gc | 6.7 GB | 0.5 min | 4 s | 5.3 s | 1.12 | 1.17 | 0.852 | 2.71 / 2.59 GB | 1.6 GB | A CPU cache is a small, high-speed memory bank located directly on… |
| gemma-4-26B-A4B-it | 25.8B MoE (4B act.) | NF4, all tensors (AirLLM 4bit) | 14.2 GB | 1.8 min (lazy split) | 4 s | 16.5 s | 14.57 | 14.79 | 0.068 | 3.46 / 3.33 GB | 2.3 GB | l de de de de de de de de de de de… |
| gemma-4-26B-A4B-it | 25.8B MoE (4B act.) | bf16 | 50.5 GB | 2.1 min (lazy split) | 6 s | 26.4 s | 24.73 | 26.97 | 0.037 | 3.37 / 3.24 GB | 7.2 GB | A CPU cache acts as a high-speed buffer between the fast processor… |
| gemma-4-26B-A4B-it | 25.8B MoE (4B act.) | NF4, matrices only + no per-layer gc | 14.2 GB | 1.7 min (lazy split) | 5 s | 10.9 s | 2.44 | 2.65 | 0.378 | 4.44 / 4.31 GB | 2.1 GB | A CPU cache works as a high-speed, small-capacity memory buffer located |
| Qwen3.6-27B | 27.8B | NF4, all tensors (AirLLM 4bit) | 15.1 GB | 1.3 min | 3 s | 21.9 s | 21.65 | 22.18 | 0.045 | 3.64 / 3.51 GB | 2.3 GB | A CPU cache is a small, high-speed memory located close to the… |
| Qwen3.6-27B | 27.8B | bf16 | 53.8 GB | 2.2 min | 3 s | 35.3 s | 35.61 | 36.34 | 0.028 | 2.85 / 2.72 GB | 5.2 GB | A CPU cache is a small, high-speed memory located close to the… |
| Qwen3.6-27B | 27.8B | NF4, matrices only + no per-layer gc | 15.1 GB | 1.3 min | 3 s | 13.4 s | 11.93 | 12.32 | 0.081 | 3.73 / 3.60 GB | 2.0 GB | A CPU cache is a small, high-speed memory component located close to… |
| Qwen3.8-27B | 27.8B | NF4, all tensors (AirLLM 4bit) | 15.1 GB | 1.3 min | 3 s | 24.4 s | 25.05 | 22.99 | 0.043 | 3.64 / 3.51 GB | 2.0 GB | A CPU cache is a small, high-speed memory located on or near… |
| Qwen3.8-27B | 27.8B | bf16 | 53.8 GB | 2.1 min | 3 s | 35.3 s | 39.97 | 42.04 | 0.024 | 2.85 / 2.72 GB | 5.1 GB | A CPU cache is a small, high-speed memory located on or near… |
| Qwen3.8-27B | 27.8B | NF4, matrices only + no per-layer gc | 15.1 GB | 1.2 min | 3 s | 13.2 s | 11.95 | 12.31 | 0.081 | 3.73 / 3.60 GB | 2.0 GB | A CPU cache is a small, high-speed memory located on or near… |

I/O breakdown (cold run):

| Model | Weights | Streamed per token (GB) | Layer read time / token | Effective read GB/s (cold) | Host→GPU + dequant / token | Prefetch |
|---|---|---|---|---|---|---|
| gemma-4-E2B-it-qat-mobile-ct | native QAT int2/4/8 | 0.78 | 3.83 s | 0.20 | 0.76 s | on |
| gemma-4-E2B-it-qat-mobile-ct | native QAT int2/4/8 + no per-layer gc | 0.78 | 0.31 s | 2.51 | 0.64 s | on |
| gemma-4-E4B-it | NF4, all tensors (AirLLM 4bit) | 2.25 | 0.74 s | 3.05 | 0.06 s | off |
| gemma-4-E4B-it | bf16 | 8.00 | 6.58 s | 1.22 | 0.77 s | on |
| gemma-4-E4B-it | bf16 + no per-layer gc | 8.00 | 1.09 s | 7.33 | 0.94 s | on |
| gemma-4-E4B-it | NF4, matrices only + no per-layer gc | 2.25 | 0.46 s | 4.86 | 0.07 s | off |
| Qwen3.5-9B | NF4, all tensors (AirLLM 4bit) | 5.04 | 1.40 s | 3.61 | 0.10 s | off |
| Qwen3.5-9B | bf16 | 17.91 | 11.60 s | 1.54 | 1.63 s | on |
| Qwen3.5-9B | bf16 + no per-layer gc | 17.91 | 4.02 s | 4.45 | 1.95 s | on |
| Qwen3.5-9B | NF4, matrices only + no per-layer gc | 5.04 | 0.81 s | 6.21 | 0.07 s | off |
| gemma-4-12B-it | NF4, all tensors (AirLLM 4bit) | 6.13 | 1.96 s | 3.13 | 0.07 s | off |
| gemma-4-12B-it | bf16 | 21.80 | 15.04 s | 1.45 | 2.02 s | on |
| gemma-4-12B-it | bf16 + no per-layer gc | 21.80 | 9.02 s | 2.42 | 2.79 s | on |
| gemma-4-12B-it | NF4, matrices only + no per-layer gc | 6.13 | 0.97 s | 6.30 | 0.08 s | off |
| gemma-4-26B-A4B-it | NF4, all tensors (AirLLM 4bit) | 13.78 | 9.86 s | 1.40 | 0.08 s | off |
| gemma-4-26B-A4B-it | bf16 | 48.99 | 20.08 s | 2.44 | 4.35 s | on |
| gemma-4-26B-A4B-it | NF4, matrices only + no per-layer gc | 13.78 | 2.50 s | 5.51 | 0.13 s | off |
| Qwen3.6-27B | NF4, all tensors (AirLLM 4bit) | 15.13 | 11.73 s | 1.29 | 0.14 s | off |
| Qwen3.6-27B | bf16 | 53.79 | 27.40 s | 1.96 | 7.63 s | on |
| Qwen3.6-27B | NF4, matrices only + no per-layer gc | 15.13 | 11.46 s | 1.32 | 0.21 s | off |
| Qwen3.8-27B | NF4, all tensors (AirLLM 4bit) | 15.13 | 13.38 s | 1.13 | 0.16 s | off |
| Qwen3.8-27B | bf16 | 53.79 | 30.68 s | 1.75 | 8.40 s | on |
| Qwen3.8-27B | NF4, matrices only + no per-layer gc | 15.13 | 11.46 s | 1.32 | 0.21 s | off |

<!-- results:end -->

## Reproduce

```bash
uv venv .venv --python 3.12 && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python bench/run_bench.py --model Qwen3.5-9B --compression 4bit-2d --no-gc   # one config
.venv/bin/python bench/run_matrix.py --plan local --cleanup                             # everything
.venv/bin/python scripts/report.py --readme && .venv/bin/python scripts/plot.py         # tables + chart
```

Kaggle: `python scripts/make_kaggle_notebook.py --plan kaggle2 --out kaggle/run.ipynb`, import it on
Kaggle with "Save & Run All" + GPU T4 and Internet on. The notebook embeds the `bench/` files because the
repo is private. Results are printed at the end of the log and saved in the notebook output.

## Caveats

* One prompt, greedy decoding, 16–32 new tokens: this measures throughput, not quality. "Correct"
  means coherent and on-topic, judged by reading the outputs (all are in the JSONs).
* Kaggle results were transcribed from the notebook log / output viewer into `results/kaggle-t4/`;
  per-token arrays are omitted there.
* Kaggle run 1 lost the three > 25B models to the scratch-disk issue. Run 2 re-ran Qwen3.6/3.8-27B
  with AirLLM's own shard streaming and Gemma-4-26B-A4B with the lazy splitter. Stock bf16 for the
  > 25B models was not run on Kaggle, which has no disk room for 50+ GB of bf16 layer files.
* Locally, Gemma-4-26B-A4B is converted with the lazy splitter for all three modes (AirLLM's splitter
  cannot load its shard); the runtime is stock AirLLM unless the mode says otherwise.
