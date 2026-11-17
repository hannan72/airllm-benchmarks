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
