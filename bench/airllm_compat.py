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
