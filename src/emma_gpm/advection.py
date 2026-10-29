"""Historical-IR108 global-translation advection reference.

Phase correlation is a deliberately simple motion estimator, not dense
optical flow or a wind retrieval. A spatially constant displacement is
extrapolated at constant velocity. Out-of-domain predictions fall back to
persistence explicitly; the geometric support mask is returned separately.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def estimate_translation(previous, current, max_shift=12):
    """Return (dx,dy) in pixels per observed interval, with positive y south.

Integer phase correlation, bounded to avoid wrapped displacements. Uniform
images have no identifiable translation and receive zero displacement.
"""
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("Expected matching (B,H,W) historical IR images")
    if not torch.isfinite(previous).all() or not torch.isfinite(current).all():
        raise ValueError("Motion estimation requires finite observations")
    height, width = previous.shape[-2:]
    if max_shift < 0:
        raise ValueError("max_shift must be nonnegative")
    a = previous.float() - previous.float().mean((-2, -1), keepdim=True)
    b = current.float() - current.float().mean((-2, -1), keepdim=True)
    cross = torch.fft.fft2(b) * torch.fft.fft2(a).conj()
    correlation = torch.fft.ifft2(cross / cross.abs().clamp_min(1e-8)).real
    dy = torch.arange(height, device=a.device)
    dx = torch.arange(width, device=a.device)
    dy = torch.where(dy <= height // 2, dy, dy - height)
    dx = torch.where(dx <= width // 2, dx, dx - width)
    allowed = (dy[:, None].abs() <= max_shift) & (dx[None, :].abs() <= max_shift)
    peak = correlation.masked_fill(~allowed, -torch.inf).flatten(1).argmax(1)
    shift = torch.stack((dx[peak % width], dy[peak // width]), dim=1).float()
    textured = (a.square().mean((-2, -1)) > 1e-10) & (b.square().mean((-2, -1)) > 1e-10)
    return torch.where(textured[:, None], shift, 0.)


def warp_translation(frame, displacement):
    """Backward sample x-dx,y-dy. Return prediction and valid source footprint."""
    if frame.ndim != 4 or displacement.shape != (len(frame), 2):
        raise ValueError("Expected frame (B,C,H,W) and displacement (B,2)")
    if min(frame.shape[-2:]) < 2 or not torch.isfinite(displacement).all():
        raise ValueError("Warp requires finite displacement and spatial dimensions >=2")
    height, width = frame.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=frame.device, dtype=torch.float32),
                          torch.arange(width, device=frame.device, dtype=torch.float32), indexing="ij")
    sx = x[None] - displacement[:, 0, None, None]
    sy = y[None] - displacement[:, 1, None, None]
    valid = (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)
    grid = torch.stack((2 * sx / (width - 1) - 1, 2 * sy / (height - 1) - 1), -1)
    warped = F.grid_sample(frame.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return torch.where(valid[:, None], warped, frame.float()), valid[:, None]


@torch.no_grad()
def advection_forecast(history, steps, max_shift=12):
    """Only the last TWO historical IR108 frames determine displacement."""
    if history.ndim != 5 or history.shape[1] < 2 or history.shape[2] != 7:
        raise ValueError("Advection requires at least two historical seven-channel frames")
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("Forecast steps must be positive")
    shift = estimate_translation(history[:, -2, 4], history[:, -1, 4], max_shift)
    forecasts, masks = zip(*(warp_translation(history[:, -1], shift * step) for step in steps))
    return torch.stack(forecasts, 1), torch.stack(masks, 1), shift
