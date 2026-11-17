"""Model registry and AirLLM class selection."""

from __future__ import annotations

import airllm_compat  # noqa: F401  (patches AirLLM for transformers 5.2x)
from airllm import AirLLMQwen3_5, AutoModel

from airllm_gemma4 import AirLLMGemma4


class AirLLMQwen3_5TextOnly(AirLLMQwen3_5):
    """AirLLM's stock Qwen3.5/3.8 class keeps the vision tower resident on the GPU.

    These benchmarks are text-only, so the vision tower is not split or loaded. That keeps
    VRAM numbers comparable with the (text-only) Gemma 4 adapter. Use ``--stock`` to run
    AirLLM's unmodified class instead.
    """

    def set_layer_names_dict(self):
        super().set_layer_names_dict()
        self.layer_names_dict["resident"] = []


# short name -> (HF repo id, notes)
MODELS = {
    "gemma-4-E2B-it-qat-mobile-ct": "google/gemma-4-E2B-it-qat-mobile-ct",
    "gemma-4-E4B-it": "google/gemma-4-E4B-it",
    "gemma-4-12B-it": "google/gemma-4-12B-it",
    "gemma-4-26B-A4B-it": "google/gemma-4-26B-A4B-it",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
}


def airllm_class_for(repo_id: str, stock: bool = False):
    """Return the AirLLM class to use, or None to let ``AutoModel`` decide."""
    if "gemma-4" in repo_id.lower():
        return AirLLMGemma4  # no stock class exists in AirLLM 4.0.0
    if stock:
        return None
    if repo_id.startswith("Qwen/Qwen3."):
        return AirLLMQwen3_5TextOnly
    return None


def build(repo_id: str, stock: bool = False, **kwargs):
    cls = airllm_class_for(repo_id, stock)
    if cls is None:
        return AutoModel.from_pretrained(repo_id, **kwargs)
    return cls(repo_id, **kwargs)
