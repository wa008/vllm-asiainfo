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
import contextlib
import importlib
import torch.library
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import CompressedTensorsW8A16Fp8
from compressed_tensors.quantization import QuantizationStrategy
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise)
import triton
import triton.language as tl
from triton import Config

logger = init_logger(__name__)

class AsiainfoQuantCompressedTensorsW8A16Fp8(CompressedTensorsW8A16Fp8):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # we expand each scale to its shard's channels.
    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(layer.weight_scale,
                                                    layer.logical_widths)
            layer.weight_scale = torch.nn.Parameter(ws_channelwise,
                                                    requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.contiguous().detach(),
                                          requires_grad=False)

        if self.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                   requires_grad=False)
        logger.debug(f"process_weights_after_loading")
        # prepare_fp8_layer_for_marlin(layer, strategy="channel"

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.output_size_per_partition, )
        logger.debug(f"asiainfo_apply_weigth")
        output = asiainfo_fp8_float16_gemm(reshaped_x, layer.weight)
        if bias is not None:
            output.add_(bias)  # In-place add
        return output.reshape(out_shape)


# chitu: https://github.com/thu-pacman/chitu/blob/74398035e99ca1b3fb80402f32cd54886def0def/chitu/ops.py#L460
def asiainfo_fp8_float16_gemm(a: torch.Tensor, b: torch.Tensor):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.
    Args:
        a (torch.Tensor): The first input matrix, must be contiguous, fp8, MxK
        b (torch.Tensor): The second input matrix, must be contiguous, fp16, 
    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert len(b.shape) == 2, f"b.shape: {b.shape}"
    assert a.shape[-1] == b.shape[-1], f"a.shape: {a.shape}, b.shape: {b.shape}"
    # assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    asiainfo_fp8_float16_gemm_kernel[grid](
        # a, b.view(dtype=torch.uint8).T.contiguous(), c, M, N, K, group_n=128, group_k=128
        a, b.view(dtype=torch.uint8), c, M, N, K, group_n=128, group_k=128
    )
    return c


asiainfo_fp8_float16_gemm_configs = [
    Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for block_k in [128]
    for group_m in [1, 32]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [4, 8]
]

@triton.autotune(configs=asiainfo_fp8_float16_gemm_configs, key=["N", "K"])
@triton.jit
def asiainfo_fp8_float16_gemm_kernel(
    A,
    B,
    C,
    # Bs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        A (tl.tensor): Pointer to the first input matrix A.
        B (tl.tensor): Pointer to the second input matrix B.
        C (tl.tensor): Pointer to the output matrix C.
        Bs (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_M (tl.constexpr): Block-swizzle group size for the M dimension.

    Returns:
        None
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * K)

    # offs_bsn = offs_bn // group_n
    # Bs_ptrs = Bs + offs_bsn * tl.cdiv(K, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)# .to(dtype=tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # k_start = k * BLOCK_SIZE_K
        # offs_ks = k_start // group_k
        # b_s = tl.load(Bs_ptrs + offs_ks)

        b_uint32 = b.to(tl.uint8, bitcast=True).to(tl.uint32)
        b_unscaled_fp32 = (((b_uint32 & 0x80) << 24) | ((b_uint32 & 0x7F) << 20)).to(
            tl.float32, bitcast=True
        ).to(dtype=a.dtype)
        # b_new_scale = b_s * fp8_to_fp32_scale
        # b_scaled_fp32 = b_unscaled_fp32 * b_new_scale
        # b_scaled_fp32 = b_scaled_fp32.to(dtype=compute_dtype)

        accumulator += tl.dot(a, b_unscaled_fp32)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

