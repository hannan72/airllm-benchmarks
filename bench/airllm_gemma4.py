"""AirLLM adapter for Google Gemma 4 checkpoints.

AirLLM 4.0.0 has no Gemma 4 class. Its generic ``AirLLMBaseModel`` looks for
``model.embed_tokens`` / ``model.layers``, which do not exist in
``Gemma4ForConditionalGeneration`` (or ``Gemma4UnifiedForConditionalGeneration``), so
``AutoModel.from_pretrained`` fails while building the empty model.

This adapter maps the text decoder under ``model.language_model`` and handles the
part specific to Gemma 4's "E" models (E2B / E4B): the Per-Layer Embedding (PLE) table
``embed_tokens_per_layer``. It is a [262144 x num_layers*256] lookup table (5.6 GB bf16
for E4B). Streaming it to the GPU for every token would cost more than all the decoder
layers together. Only a handful of rows are needed per step, so we keep it in host RAM
and gather rows on the CPU. That is the same idea AirLLM itself uses for Qwen3.8
Flash-Next's n-gram table.

Text-only: vision / audio towers are not split or loaded.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

import airllm.utils as airllm_utils
from airllm import AirLLMBaseModel

PLE_NAME = "model.language_model.embed_tokens_per_layer"

# The PLE table is a CPU lookup table that never streams, so 4/8-bit block compression
# gains nothing there and would only break the CPU gather. Skip it when splitting.
_original_compress = airllm_utils.compress_layer_state_dict


def _compress_skip_ple(layer_state_dict, compression=None):
    if compression is not None and any(PLE_NAME in k for k in layer_state_dict):
        return layer_state_dict
    return _original_compress(layer_state_dict, compression)


airllm_utils.compress_layer_state_dict = _compress_skip_ple


def _ple_cpu_forward(module, input_ids):
    table = module._airllm_cpu_table
    out = F.embedding(input_ids.to("cpu"), table, module.padding_idx)
    out = out.to(input_ids.device, non_blocking=True)
    return out * module.embed_scale.to(out.dtype)


def _unpack_ct_int(packed: torch.Tensor, num_bits: int, shape) -> torch.Tensor:
    """Unpack compressed-tensors ``pack-quantized`` int32 words into signed ints (CPU)."""
    try:  # compressed-tensors >= 0.20
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    except ImportError:  # older layout
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

    return unpack_from_int32(packed, num_bits, torch.Size([int(s) for s in shape]))


class AirLLMGemma4(AirLLMBaseModel):
    """Gemma 4 dense / MoE / E-series, text-only."""

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            # Gemma 4 ties lm_head to embed_tokens. The QAT "-ct" checkpoint ships a
            # separate 2-bit lm_head, which is then streamed like a layer.
            "lm_head": "lm_head",
            "resident": [
                "model.language_model.per_layer_model_projection",
                "model.language_model.per_layer_projection_norm",
            ],
            "cpu_resident": [PLE_NAME],
        }

    def init_model(self):
        # Text-only: drop the audio / vision towers from the config before the empty model is
        # built. Besides saving memory, Gemma4AudioModel has no SDPA attention, and its
        # presence would force AirLLM to fall back to eager attention for the whole model.
        for attr in ("audio_config", "vision_config"):
            if getattr(self.config, attr, None) is not None:
                setattr(self.config, attr, None)
        super().init_model()

    # ---- PLE table on CPU -----------------------------------------------------------------

    def _load_cpu_resident_modules(self):
        self._cpu_resident_params = set()
        try:
            state_dict = airllm_utils.load_merged_ngram_embedding(self.checkpoint_path, PLE_NAME)
        except FileNotFoundError:
            return  # dense Gemma 4 (12B, 26B-A4B) has no PLE
        module = self.model.get_submodule(PLE_NAME)

        if f"{PLE_NAME}.weight_packed" in state_dict:
            # compressed-tensors pack-quantized (QAT checkpoints): int4, symmetric, group-wise.
            scheme = getattr(module, "quantization_scheme", None)
            num_bits = scheme.weights.num_bits if scheme is not None else 4
            q = _unpack_ct_int(state_dict[f"{PLE_NAME}.weight_packed"], num_bits,
                               state_dict[f"{PLE_NAME}.weight_shape"].tolist())
            scale = state_dict[f"{PLE_NAME}.weight_scale"].float()
            group = q.shape[1] // scale.shape[1]
            table = torch.empty(q.shape, dtype=self.running_dtype)
            step = 8192  # dequantize in row chunks to bound peak host RAM
            for r in range(0, q.shape[0], step):
                qs = q[r:r + step].float().view(-1, scale.shape[1], group)
                table[r:r + step] = (qs * scale[r:r + step, :, None]).view(qs.shape[0], -1).to(self.running_dtype)
            del q, scale
        else:
            table = state_dict[f"{PLE_NAME}.weight"].to(self.running_dtype)

        module._airllm_cpu_table = table.contiguous()
        if not hasattr(module, "padding_idx"):
            module.padding_idx = None
        module.forward = types.MethodType(_ple_cpu_forward, module)
        # Marks the post-hook to release only streamed tensors instead of module.to('meta').
        self._cpu_resident_params.add(f"{PLE_NAME}.weight")
        print(f"[gemma4] PLE table kept on CPU: {tuple(table.shape)} {table.dtype} "
              f"({table.numel() * table.element_size() / 1e9:.2f} GB host RAM)")
