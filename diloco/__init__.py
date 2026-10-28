"""DiLoCo (Distributed Low-Communication training) for any PyTorch model."""

from .compression import dequantize, payload_bytes, quantize
from .diloco import DiLoCo
from .dist import init_distributed, launched_by_torchrun, pick_device
from .fragments import fragment_offsets, fragments_due, split_into_fragments

__all__ = [
    "DiLoCo",
    "dequantize",
    "fragment_offsets",
    "fragments_due",
    "init_distributed",
    "launched_by_torchrun",
    "payload_bytes",
    "pick_device",
    "quantize",
    "split_into_fragments",
]
__version__ = "0.1.0"
