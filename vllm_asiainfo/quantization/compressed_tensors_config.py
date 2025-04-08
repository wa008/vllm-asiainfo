# SPDX-License-Identifier: Apache-2.0

from contextlib import suppress
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.config import (CompressionFormat,
                                       SparsityCompressionConfig,
                                       SparsityStructure)
from vllm.model_executor.layers.quantization.compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from pydantic import BaseModel

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    W4A16SPARSE24_SUPPORTED_BITS, WNA16_SUPPORTED_BITS, CompressedTensors24,
    CompressedTensorsScheme, CompressedTensorsW4A16Sparse24,
    CompressedTensorsW8A8Fp8, CompressedTensorsW8A8Int8,
    CompressedTensorsW8A16Fp8, CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, is_activation_quantization_format,
    should_ignore_layer)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.compressed_tensors import CompressedTensorsConfig
from .quant_w8a16_schema import AsiainfoCompressedTensorsW8A16Fp8

logger = init_logger(__name__)

@register_quantization_config("asiainfo_quant_w8a16_fp8")
class AsiainfoCompressedTensorsConfig(CompressedTensorsConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_scheme_from_parts(
                    self, weight_quant: BaseModel,
                    input_quant: BaseModel) -> "CompressedTensorsScheme":
        return AsiainfoCompressedTensorsW8A16Fp8(
            strategy=weight_quant.strategy,
            is_static_input_scheme=not input_quant.dynamic
        )

