"""
Blockwise k-bit quantization of pseudo-gradients, packed into uint8 for the wire.

Each block of BLOCK_SIZE values gets one float32 scale. For bits >= 2 values are
rounded to 2^(bits-1) - 1 symmetric levels of the block's max magnitude; 1 bit
sends the sign and uses the block's mean magnitude as the scale.
"""

import torch

BLOCK_SIZE = 1024
SUPPORTED_BITS = (1, 2, 4, 8)


def _check_bits(bits):
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")


def num_blocks(numel):
    return -(-numel // BLOCK_SIZE)


def payload_bytes(numel, bits):
    """Bytes on the wire for `numel` values: packed codes plus one float32 scale per block."""
    _check_bits(bits)
    blocks = num_blocks(numel)
    return blocks * BLOCK_SIZE * bits // 8 + blocks * 4


def pack_bits(codes, bits):
    """Pack unsigned codes (< 2**bits, length divisible by 8 // bits) into uint8."""
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.int32) * bits
    grouped = codes.to(torch.int32).view(-1, per_byte)
    return (grouped << shifts).sum(dim=1).to(torch.uint8)


def unpack_bits(packed, bits):
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int32) * bits
    return ((packed.to(torch.int32).unsqueeze(1) >> shifts) & (2 ** bits - 1)).flatten()


def quantize(flat, bits):
    """Quantize a 1-D float tensor. Returns (packed uint8 codes, float32 scales)."""
    _check_bits(bits)
    flat = flat.float()
    padded = torch.zeros(num_blocks(flat.numel()) * BLOCK_SIZE, device=flat.device)
    padded[:flat.numel()] = flat
    blocks = padded.view(-1, BLOCK_SIZE)

    if bits == 1:
        scales = blocks.abs().mean(dim=1)
        codes = (blocks >= 0).to(torch.int32)
    else:
        levels = 2 ** (bits - 1) - 1
        scales = blocks.abs().amax(dim=1)
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        codes = (torch.round(blocks / safe[:, None] * levels).clamp(-levels, levels) + levels).to(torch.int32)

    return pack_bits(codes.flatten(), bits), scales


def dequantize(packed, scales, bits, numel):
    """Inverse of quantize: back to a 1-D float32 tensor of length `numel`."""
    codes = unpack_bits(packed, bits).float().view(-1, BLOCK_SIZE)

    if bits == 1:
        values = (codes * 2 - 1) * scales[:, None]
    else:
        levels = 2 ** (bits - 1) - 1
        values = (codes - levels) / levels * scales[:, None]

    return values.flatten()[:numel]
