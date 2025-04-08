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

@register_quantization_config("asiainfo_quant")
class AsiainfoQuantConfig(Fp8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def get_name(self) -> str:
    #     return "asiainfo_quant"

    # def __repr__(self) -> str:
    #     return "AsiainfoQuantConfig:\n" + super().__repr__()

    # @classmethod
    # def from_config(cls, config: Dict[str, Any]) -> "AsiainfoQuantConfig":
    #     return cls(config)
    # def get_quant_method(self, layer: torch.nn.Module,
    #                      prefix: str) -> Optional["QuantizeMethodBase"]:
    #     from vllm.attention.layer import Attention  # Avoid circular import

    #     if isinstance(layer, LinearBase):
    #         if is_layer_skipped(prefix, self.ignored_layers):
    #             return UnquantizedLinearMethod()
    #         return Fp8LinearMethod(self)
    #     elif isinstance(layer, FusedMoE):
    #         return Fp8MoEMethod(self)
    #     elif isinstance(layer, Attention):
    #         return Fp8KVCacheMethod(self)
    #     return None
