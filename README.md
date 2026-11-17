# AirLLM benchmarks: Gemma 4 and Qwen 3.x on small GPUs

Work in progress: results will be added here.

- `bench/run_bench.py`: benchmark one model/compression config and write a JSON result
- `bench/run_matrix.py`: run the whole matrix sequentially (resumable)
- `bench/airllm_gemma4.py`: Gemma 4 adapter (AirLLM 4.0.0 has no Gemma 4 class)
- `bench/airllm_compat.py`: compatibility shim for AirLLM 4.0.0 with transformers 5.2x
