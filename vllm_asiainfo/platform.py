from vllm.platforms.cuda import NonNvmlCudaPlatform
from vllm.platforms import Platform, PlatformEnum
from typing import TYPE_CHECKING, Optional, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
    FlexibleArgumentParser = None


class AsiainfoPlatform(NonNvmlCudaPlatform):
    # _enum = PlatformEnum.OOT
    supported_quantization: list[str] = ["asiainfo_quant", "fp8", "compressed-tensors"]

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
        from vllm_asiainfo.quantization.quant_w8a16_config import AsiainfoQuantConfig # add new quant config
        from vllm_asiainfo.quantization.quant_compressed_tensors_config import AsiainfoQuantCompressedTensorsConfig # rewrite exist quant config

