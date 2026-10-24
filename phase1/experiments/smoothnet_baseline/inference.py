"""
SmoothNet inference — loads third_party/SmoothNet in isolation (sys.path injection here only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from experiments.smoothnet_baseline.config import (
    DROPOUT,
    H36M_FLAT_DIM,
    HIDDEN_SIZE,
    INFERENCE_BATCH_SIZE,
    NUM_BLOCKS,
    RES_HIDDEN_SIZE,
    WINDOW_SIZE,
    WINDOW_STEP,
)
from experiments.smoothnet_baseline.bridge import (
    denormalize_pixels,
    frames_to_h36m_sequence,
    interpolate_nans_along_time,
    normalize_pixels,
)


def slide_window_to_sequence(
    slide_window: torch.Tensor,
    window_step: int,
    window_size: int,
) -> torch.Tensor:
    """
    Merge overlapping sliding-window predictions by averaging (from SmoothNet utils).
    slide_window: (N_windows, window_size, C)
    Returns: (T, C)
    """
    n_win = slide_window.shape[0]
    if n_win == 0:
        return slide_window.reshape(0, slide_window.shape[-1])

    output_len = (n_win - 1) * window_step + window_size
    sequence: list = [[] for _ in range(output_len)]

    for i in range(n_win):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).float().mean(0)

    return torch.stack(sequence)


def _inject_smoothnet_path(smoothnet_root: Path) -> None:
    root = str(smoothnet_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_smoothnet_model(
    checkpoint_path: Path,
    smoothnet_root: Path,
    device: torch.device,
    window_size: int = WINDOW_SIZE,
) -> torch.nn.Module:
    _inject_smoothnet_path(smoothnet_root)
    from lib.models.smoothnet import SmoothNet  # noqa: WPS433 — vendor import

    model = SmoothNet(
        window_size=window_size,
        output_size=window_size,
        hidden_size=HIDDEN_SIZE,
        res_hidden_size=RES_HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    )
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _build_sliding_windows(
    flat_norm: np.ndarray,
    window_size: int,
    window_step: int,
) -> np.ndarray:
    """
    flat_norm: (T, 34) normalized 2D pose.
    Returns windows: (N, window_size, 34)
    """
    t_len, dim = flat_norm.shape
    if t_len <= 0:
        return np.zeros((0, window_size, dim), dtype=np.float32)

    if t_len < window_size:
        padded = np.zeros((window_size, dim), dtype=np.float32)
        padded[:t_len] = flat_norm.astype(np.float32)
        return padded[np.newaxis, :, :]

    starts = np.arange(0, t_len - window_size + 1, window_step)
    windows = np.stack([flat_norm[s : s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)


def _run_model_on_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> torch.Tensor:
    """windows (N, T, 34) -> merged (T_total, 34) after per-window inference."""
    if windows.shape[0] == 0:
        return torch.zeros((0, H36M_FLAT_DIM), dtype=torch.float32)

    window_size = windows.shape[1]
    outputs: list = []

    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size]
            # SmoothNet expects (N, C, T)
            tensor = torch.from_numpy(batch).permute(0, 2, 1).to(device)
            out = model(tensor)
            out = out.permute(0, 2, 1).cpu()  # (N, T, 34)
            outputs.append(out)

    all_out = torch.cat(outputs, dim=0)
    merged = slide_window_to_sequence(all_out, WINDOW_STEP, window_size)
    return merged


def smooth_h36m_sequence(
    pose_pixels: np.ndarray,
    model: torch.nn.Module,
    width: float,
    height: float,
    device: torch.device,
    window_size: int = WINDOW_SIZE,
    window_step: int = WINDOW_STEP,
) -> np.ndarray:
    """
    Apply SmoothNet to (T, 17, 2) pixel coordinates.
    Returns smoothed (T, 17, 2) in pixel space.
    """
    t_len = pose_pixels.shape[0]
    if t_len == 0:
        return pose_pixels.copy()

    # Track which joints were valid before smoothing (all-NaN joints stay as input)
    valid_mask = np.isfinite(pose_pixels).all(axis=-1)  # (T, 17)

    filled = interpolate_nans_along_time(pose_pixels)
    norm = normalize_pixels(filled, width, height)
    flat = norm.reshape(t_len, H36M_FLAT_DIM)

    # Replace remaining NaN with 0 for network input
    nan_mask = ~np.isfinite(flat)
    flat_input = flat.copy()
    flat_input[nan_mask] = 0.0

    windows = _build_sliding_windows(flat_input, window_size, window_step)

    if windows.shape[0] == 0:
        return pose_pixels.copy()

    if t_len < window_size:
        # Single padded window — trim back to original length
        tensor_in = torch.from_numpy(windows).permute(0, 2, 1).to(device)
        with torch.no_grad():
            out = model(tensor_in).permute(0, 2, 1).cpu()
        flat_out = out[0, :t_len, :].numpy()
    else:
        merged = _run_model_on_windows(model, windows, device)
        flat_out = merged.numpy()

    flat_out[nan_mask] = flat[nan_mask]
    norm_out = flat_out.reshape(t_len, 17, 2)
    pixel_out = denormalize_pixels(norm_out, width, height)

    # Restore joints that were never valid in input
    never_valid = ~valid_mask.any(axis=0)
    for j in range(17):
        if never_valid[j]:
            pixel_out[:, j, :] = pose_pixels[:, j, :]

    return pixel_out


def smooth_keypoints_data(
    data: dict,
    model: torch.nn.Module,
    device: torch.device,
    key_field: str = "keypoints",
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Run SmoothNet on JSON-loaded keypoints dict; return (T, 17, 2) pixels."""
    frames = data.get("frames", [])
    image_size = data.get("image_size") or {}
    width = float(image_size.get("width", 1000))
    height = float(image_size.get("height", 1000))

    pose = frames_to_h36m_sequence(frames, key_field=key_field)
    return smooth_h36m_sequence(
        pose,
        model,
        width,
        height,
        device,
        window_size=window_size,
    )
