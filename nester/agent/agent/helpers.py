import torch


def check_finite_logits(logits: torch.Tensor, label: str = "") -> torch.Tensor:
    prefix = f"{label} " if label else ""
    if torch.isnan(logits).any():
        raise ValueError(f"{prefix}logits contain NaN")
    if torch.isposinf(logits).any():
        raise ValueError(f"{prefix}logits contain +inf")
    return logits


def check_unit_interval(x: torch.Tensor, name: str) -> torch.Tensor:
    xf = x.float()
    if (xf < 0).any() or (xf > 1).any():
        raise ValueError(
            f"{name} must be in [0, 1], got min={float(xf.min().item())}, max={float(xf.max().item())}"
        )
    return x


def check_index_range(
    indices: torch.Tensor,
    lo: int,
    hi: int,
    name: str,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        if not mask.any():
            return indices
        indices = indices[mask]
    if indices.numel() == 0:
        return indices
    min_v = int(indices.min().item())
    max_v = int(indices.max().item())
    if min_v < lo or max_v > hi:
        raise ValueError(f"{name} out of range [{lo}, {hi}]: min={min_v}, max={max_v}")
    return indices


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))
