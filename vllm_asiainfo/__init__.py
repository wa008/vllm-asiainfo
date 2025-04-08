from vllm.logger import init_logger
logger = init_logger(__name__)

def register():
    """Register the NPU platform."""
    return "vllm_asiainfo.platform.AsiainfoPlatform"
