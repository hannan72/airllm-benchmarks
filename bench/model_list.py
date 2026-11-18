"""Benchmarked models: short name -> Hugging Face repo id (no AirLLM import, so the vLLM and
ZeroGPU runners can use it without installing AirLLM)."""

MODELS = {
    "gemma-4-E2B-it-qat-mobile-ct": "google/gemma-4-E2B-it-qat-mobile-ct",
    "gemma-4-E4B-it": "google/gemma-4-E4B-it",
    "gemma-4-12B-it": "google/gemma-4-12B-it",
    "gemma-4-26B-A4B-it": "google/gemma-4-26B-A4B-it",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
}
