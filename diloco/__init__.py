"""DiLoCo (Distributed Low-Communication training) for any PyTorch model."""

from .compression import dequantize, payload_bytes, quantize
from .diloco import DiLoCo
from .dist import init_distributed, launched_by_torchrun, pick_device
from .fragments import fragment_offsets, fragments_due, split_into_fragments

__all__ = [
    "DiLoCo",
    "init_distributed",
    "launched_by_torchrun",
    "pick_device",
    "quantize",
    "dequantize",
    "payload_bytes",
    "split_into_fragments",
    "fragment_offsets",
    "fragments_due",
]
__version__ = "0.1.0"
