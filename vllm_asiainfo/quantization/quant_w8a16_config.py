import torch
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config, Fp8LinearMethod, Fp8MoEMethod, Fp8KVCacheMethod)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from typing import Any, Callable, Dict, List, Mapping, Optional


# add a new config example
@register_quantization_config("asiainfo_quant")
class AsiainfoQuantConfig(Fp8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

