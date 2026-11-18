### kaggle-t4

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

### local-rtx2070

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
| Qwen3.8-27B | NF4, all tensors (AirLLM 4bit) | 15.13 | 13.38 s | 1.13 | 0.16 s | off |
| Qwen3.8-27B | bf16 | 53.79 | 30.68 s | 1.75 | 8.40 s | on |
| Qwen3.8-27B | NF4, matrices only + no per-layer gc | 15.13 | 11.46 s | 1.32 | 0.21 s | off |

