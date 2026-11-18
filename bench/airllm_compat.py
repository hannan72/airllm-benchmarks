"""Compatibility shims for AirLLM 4.0.0 on newer transformers (tested: 5.23).

AirLLM builds a pre-quantized model's quantizer with ``AutoHfQuantizer.from_config`` and
calls ``preprocess_model`` straight away. transformers' own loader calls
``validate_environment()`` in between. Recent versions set required state there (e.g.
``CompressedTensorsHfQuantizer.use_fp8_kernel``), so without that call loading any
compressed-tensors checkpoint fails with an AttributeError.
"""

import airllm.airllm_base as _base
from transformers.quantizers.auto import AutoHfQuantizer as _AutoHfQuantizer


class _AutoHfQuantizerShim:
    @staticmethod
    def from_config(*args, **kwargs):
        quantizer = _AutoHfQuantizer.from_config(*args, **kwargs)
        try:
            quantizer.validate_environment(device_map=None)
        except TypeError:
            quantizer.validate_environment()
        return quantizer


_base.AutoHfQuantizer = _AutoHfQuantizerShim


# --- compressed-tensors checkpoints with activation quantization (e.g. Gemma 4 QAT "-ct") -------
#
# 1. ``AirLLMBaseModel._decompress_state_dict`` expands ``weight_packed`` into ``weight`` but
#    pops *every* tensor under that module's prefix, including the static activation scales
#    (``input_scale``) that compressed-tensors' quantized_forward needs for W4A8 / W2A8 layers.
#    Those stay on the meta device, and the first forward crashes.
# 2. ``int-quantized`` (plain int8) weights are not packed, so AirLLM loads them verbatim as int8
#    into a module whose forward does a bf16 matmul. Dequantize them (symmetric, per channel or group).

import torch as _torch

_ACT_SUFFIXES = (".input_scale", ".input_zero_point", ".output_scale", ".output_zero_point",
                 ".input_global_scale")
_orig_decompress = _base.AirLLMBaseModel._decompress_state_dict


def _has_tensor(module, attr):
    return attr in getattr(module, "_parameters", {}) or attr in getattr(module, "_buffers", {})


def _decompress_state_dict_fixed(self, state_dict):
    out = _orig_decompress(self, state_dict)
    if self.hf_quantizer is None:
        return out
    if out is state_dict:
        out = dict(state_dict)

    for key, value in state_dict.items():
        if not key.endswith(_ACT_SUFFIXES):
            continue
        prefix, _, attr = key.rpartition(".")
        try:
            module = self.model.get_submodule(prefix)
        except AttributeError:
            continue
        if _has_tensor(module, attr):
            # The empty model is built in fp32 and accelerate casts loaded values to the existing
            # parameter's dtype, so an fp32 input_scale would promote the fake-quantized
            # activation to fp32 and break the bf16 matmul that follows. Re-type the meta param.
            if value.is_floating_point() and attr in module._parameters:
                old = module._parameters[attr]
                module._parameters[attr] = _torch.nn.Parameter(
                    _torch.empty(old.shape, device="meta", dtype=self.running_dtype), requires_grad=False)
                value = value.to(self.running_dtype)
            out[key] = value

    for key in [k for k in out if k.endswith(".weight") and not out[k].is_floating_point()]:
        prefix = key[: -len(".weight")]
        scale = out.get(prefix + ".weight_scale")
        if scale is None:
            continue
        try:
            module = self.model.get_submodule(prefix)
        except AttributeError:
            continue
        q = out[key].to(self.running_device, _torch.float32)
        s = scale.to(self.running_device, _torch.float32)
        if s.dim() == 2 and s.shape[1] not in (1, q.shape[1]):  # group-wise
            g = q.shape[1] // s.shape[1]
            w = (q.view(q.shape[0], s.shape[1], g) * s[:, :, None]).view_as(q)
        else:
            w = q * s
        zp = out.pop(prefix + ".weight_zero_point", None)
        if zp is not None and zp.abs().sum() > 0:
            raise NotImplementedError(f"asymmetric int weights not handled: {prefix}")
        out[key] = w.to(self.running_dtype)
        out.pop(prefix + ".weight_scale", None)
        self._expose_plain_weight(module, out[key])
    return out


_base.AirLLMBaseModel._decompress_state_dict = _decompress_state_dict_fixed


# --- optional: NF4-compress only weight matrices ----------------------------------------------
#
# AirLLM's ``compression='4bit'`` quantizes *every* tensor of a layer to NF4 (blocksize 64),
# including RMSNorm weights, per-layer scalars and other 1-D parameters. Regular weight-only
# quantizers (bitsandbytes / GPTQ / AWQ) leave those in full precision. With
# AIRLLM_COMPRESS_MIN_DIM=2 only tensors with >= 2 dims are compressed; the rest are stored as is.

import os as _os

import airllm.utils as _utils

_COMPRESS_MIN_DIM = int(_os.environ.get("AIRLLM_COMPRESS_MIN_DIM", "0"))
_orig_compress = _utils.compress_layer_state_dict
_orig_uncompress = _utils.uncompress_layer_state_dict


def _compress_min_dim(layer_state_dict, compression=None):
    if compression is None or _COMPRESS_MIN_DIM <= 1:
        return _orig_compress(layer_state_dict, compression)
    big = {k: v for k, v in layer_state_dict.items() if v.dim() >= _COMPRESS_MIN_DIM}
    out = _orig_compress(big, compression) if big else {}
    out.update({k: v for k, v in layer_state_dict.items() if v.dim() < _COMPRESS_MIN_DIM})
    return out


def _uncompress_mixed(layer_state_dict):
    """Original only handles all-or-nothing compressed shards; pass uncompressed tensors through."""
    tags = (".4bit.", ".8bit.")
    if not any(t in k for k in layer_state_dict for t in tags):
        return layer_state_dict
    bases = {k.split(".4bit.")[0].split(".8bit.")[0] for k in layer_state_dict if any(t in k for t in tags)}
    quant = {k: v for k, v in layer_state_dict.items() if k in bases or any(t in k for t in tags)}
    out = _orig_uncompress(quant)
    out.update({k: v for k, v in layer_state_dict.items() if k not in quant})
    return out


_utils.compress_layer_state_dict = _compress_min_dim
_utils.uncompress_layer_state_dict = _uncompress_mixed


# --- optional: skip AirLLM's per-layer garbage collection -------------------------------------
#
# AirLLM calls clean_memory() (gc.collect + malloc_trim + torch.cuda.empty_cache) after *every*
# streamed module. With a transformers model graph alive, gc.collect() alone costs ~100 ms
# (≈435k tracked objects for Gemma 4 E4B), i.e. several seconds per generated token. Tensors are
# freed by reference counting anyway. AIRLLM_NO_GC=1 turns the per-layer call into a no-op.

if _os.environ.get("AIRLLM_NO_GC") == "1":
    _base.clean_memory = lambda: None


# --- disk-space check: AirLLM divides by the 4-bit size ratio instead of multiplying ----------
#
# ``check_space`` estimates the split size of a 4-bit model as ``checkpoint_bytes / 0.2813``,
# i.e. 3.6x the checkpoint instead of 0.28x, and refuses to split a 52 GB model with 183 GB free.
# (The 8-bit branch, ``// 2``, is in the right direction.) Re-implemented with the ratio applied.

import shutil as _shutil
from glob import glob as _glob
from pathlib import Path as _Path


def _check_space_fixed(checkpoint_path, layer_shards_saving_path=None, compression=None,
                       splitted_model_dir_name="splitted_model"):
    total = sum(_os.path.getsize(f) for f in _glob(str(_Path(checkpoint_path) / "*")) if _os.path.isfile(f))
    reusable = 0
    if layer_shards_saving_path is not None:
        reusable = sum(_os.path.getsize(f) for f in
                       _glob(str(_Path(layer_shards_saving_path) / splitted_model_dir_name / "*")))
    if compression == "4bit":
        total = int(total * 0.2813)
    elif compression == "8bit":
        total = total // 2
    free = _shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path).free
    if free + reusable < total:
        raise _utils.NotEnoughSpaceException(
            f"Not enough space: {free / 2**30:.1f} GB free, split needs ~{total / 2**30:.1f} GB")


_utils.check_space = _check_space_fixed
