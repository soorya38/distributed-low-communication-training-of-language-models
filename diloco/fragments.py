"""
Parameter fragments for Streaming DiLoCo.

The model's parameters are split into P contiguous fragments of roughly equal
size. Fragment p syncs every H steps, offset by p * H // P steps, so at most one
fragment is on the wire at a time and peak bandwidth drops by about P.
"""


def split_into_fragments(params, num_fragments):
    """Split a parameter list into `num_fragments` contiguous groups balanced by element count.

    Returns a list of index lists into `params`.
    """
    if not 1 <= num_fragments <= len(params):
        raise ValueError(f"num_fragments must be between 1 and {len(params)}, got {num_fragments}")

    sizes = [p.numel() for p in params]
    total = sum(sizes)
    fragments, current, running = [], [], 0

    for i, size in enumerate(sizes):
        fragments_left = num_fragments - len(fragments) - 1  # still to open after the current one
        if current and fragments_left > 0:
            boundary = total * (len(fragments) + 1) / num_fragments
            must_cut = len(params) - i == fragments_left  # just enough params left for the rest
            if must_cut or abs(running - boundary) <= abs(running + size - boundary):
                fragments.append(current)
                current = []
        current.append(i)
        running += size

    fragments.append(current)
    return fragments


def fragment_offsets(inner_steps, num_fragments):
    if inner_steps < num_fragments:
        raise ValueError("inner_steps must be at least num_fragments so every fragment gets its own sync step")
    return [p * inner_steps // num_fragments for p in range(num_fragments)]


def fragments_due(step, inner_steps, offsets):
    """Fragments that sync after completing inner step number `step` (1-based)."""
    return [f for f, offset in enumerate(offsets) if step % inner_steps == offset]
