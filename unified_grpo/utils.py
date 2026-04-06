from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO
from typing import Optional

import torch


def prepare_rotary_emb(self, *, latents: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        # Only needed for some CogVideoX variants.
    tr = getattr(self.pipeline, "transformer", None)
    if tr is None:
        return None
    cfg = getattr(tr, "config", None)
    if cfg is None or not getattr(cfg, "use_rotary_positional_embeddings", False):
        return None

    fn = getattr(self.pipeline, "_prepare_rotary_positional_embeddings", None)
    if fn is None:
        # Not available in some diffusers versions; return None and rely on pipeline defaults.
        return None

    vae_scale_spatial = int(getattr(self.pipeline, "vae_scale_factor_spatial", getattr(self.pipeline, "vae_scale_factor", 8)))
    # Latents are [B, F, C, H, W]
    return fn(
            height=int(latents.size(3) * vae_scale_spatial),
            width=int(latents.size(4) * vae_scale_spatial),
            num_frames=int(latents.size(1)),
            device=device,
        )
        
def resolve_lora_blocks(
    *,
    spec: str | None,
    total_blocks: int | None,
    unfreeze_pct: float,
) -> Optional[list[int]]:
    """
    Resolve LoRA block selection.

    - None / 'all' / '*' -> None (meaning: ALL blocks)
    - 'last' -> last `unfreeze_pct` of [0..total_blocks-1]
    - 'i,j,k' -> explicit indices
    """
    if spec is None:
        return None

    s = str(spec).strip().lower()
    if s in ("", "all", "*"):
        return None

    if s in ("last", "last_pct", "pct"):
        if total_blocks is None:
            raise ValueError("Cannot use --lora-blocks last: total block count is unknown for this model.")
        p = max(0.0, min(1.0, float(unfreeze_pct)))
        n = max(1, int(round(total_blocks * p)))
        return list(range(total_blocks - n, total_blocks))

    return [int(x.strip()) for x in s.split(",") if x.strip()]


class WriteLogger:
    """Writes to both console and a log file simultaneously (line-buffered)."""

    def __init__(self, filename: str):
        self.terminal: TextIO = sys.stdout
        self.log: TextIO = open(filename, "w", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        term_isatty = getattr(self.terminal, "isatty", None)
        if callable(term_isatty):
            try:
                return bool(term_isatty())
            except Exception:
                return False
        return False

    def close(self) -> None:
        self.log.close()


def _video_tensor_to_thwc_uint8(video: "torch.Tensor") -> "tuple[object, float, float]":
    """
    Convert a video tensor to a NumPy uint8 array in [T, H, W, 3].

    Accepts common layouts:
    - [B, T, C, H, W]
    - [T, C, H, W]
    - [C, T, H, W]
    - [T, H, W, C]

    Returns: (video_np_uint8, pre_min, pre_max)
    """
    import numpy as np  # local import (utils.py is used in many contexts)
    import torch  # local import

    x = video.detach()
    if x.device.type != "cpu":
        x = x.cpu()

    x_np = x.float().numpy()

    # drop batch
    if x_np.ndim == 5:
        x_np = x_np[0]

    if x_np.ndim != 4:
        raise ValueError(f"Unexpected video tensor rank: {x_np.ndim} (shape={x_np.shape})")

    # normalize to [T,H,W,3]
    if x_np.shape[-1] == 3:
        thwc = x_np  # [T,H,W,3]
    elif x_np.shape[1] == 3:
        thwc = x_np.transpose(0, 2, 3, 1)  # [T,3,H,W] -> [T,H,W,3]
    elif x_np.shape[0] == 3:
        thwc = x_np.transpose(1, 2, 3, 0)  # [3,T,H,W] -> [T,H,W,3]
    else:
        raise ValueError(f"Unexpected 4D video tensor shape (can't find channel dim): {x_np.shape}")

    pre_min = float(np.min(thwc))
    pre_max = float(np.max(thwc))

    # normalize to [0,1]
    if pre_max > 10.0:  # likely [0,255]
        thwc = thwc / 255.0
    elif pre_min < -0.5:  # likely [-1,1]
        thwc = (thwc + 1.0) / 2.0
    elif pre_max <= 1.1:
        pass  # already [0,1]
    else:
        thwc = (thwc - pre_min) / (pre_max - pre_min + 1e-8)

    thwc_u8 = (thwc * 255.0).clip(0, 255).astype(np.uint8)
    return thwc_u8, pre_min, pre_max


def video_tensor_to_middle_frame_uint8(video: "torch.Tensor") -> "object":
    """
    From a decoded video tensor (adapter.decode_for_reward output), take the middle temporal frame.

    Returns H×W×3 uint8 numpy array (RGB).
    """
    thwc, _, _ = _video_tensor_to_thwc_uint8(video)
    t = int(thwc.shape[0])
    mid = max(0, t // 2)
    return thwc[mid]


def save_rgb_frame_png(frame_hw3: "object", out_path: Path) -> Path:
    """
    Save one RGB uint8 frame shaped [H, W, 3] as a PNG.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(frame_hw3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 frame, got shape {arr.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(str(out_path), optimize=True)
    return out_path


def save_denoising_trajectory_strip_png(
    frames_hw3: "list",
    out_path: Path,
    *,
    max_panel_height: int = 280,
) -> Path:
    """
    Concatenate many RGB uint8 [H,W,3] panels horizontally into one wide PNG.

    Each panel is resized so height <= max_panel_height (aspect preserved), then
    padded to a common height so the strip is a single rectangle.
    """
    import numpy as np
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_hw3:
        raise ValueError("save_denoising_trajectory_strip_png: empty frame list")

    resized: list = []
    for im in frames_hw3:
        arr = np.asarray(im)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 uint8 frame, got shape {arr.shape}")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h <= 0 or w <= 0:
            continue
        if h > int(max_panel_height):
            new_h = int(max_panel_height)
            new_w = max(1, int(round(w * new_h / h)))
        else:
            new_h, new_w = h, w
        pil = Image.fromarray(arr)
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.append(np.array(pil))

    if not resized:
        raise ValueError("save_denoising_trajectory_strip_png: no valid frames after resize")

    mh = max(int(r.shape[0]) for r in resized)
    padded: list = []
    for r in resized:
        h = int(r.shape[0])
        if h < mh:
            pad = np.zeros((mh - h, r.shape[1], 3), dtype=np.uint8)
            r = np.vstack([r, pad])
        padded.append(r)

    strip = np.concatenate(padded, axis=1)
    Image.fromarray(strip).save(str(out_path), optimize=True)
    return out_path


def save_video_tensor_as_mp4(
    *,
    video: "torch.Tensor",
    mp4_path: Path,
    fps: float = 8.0,
    codec: str = "libx264",
    quality: int = 8,
    verbose: bool = True,
) -> Path:
    """
    Save a decoded video tensor to an MP4 on disk.

    This is intentionally separate from `adapter.decode_for_reward()`:
    - decode_for_reward: latents -> torch video tensor (model-specific)
    - this function: torch video tensor -> uint8 frames -> mp4 (model-agnostic)
    """
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    video_u8, pre_min, pre_max = _video_tensor_to_thwc_uint8(video)
    if verbose:
        print(f"  [DEBUG] Video range before norm: [{pre_min:.3f}, {pre_max:.3f}]")
        print(f"  [DEBUG] Video range after norm: [{video_u8.min()}, {video_u8.max()}]")

    try:
        import imageio  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `imageio` for MP4 writing.\n"
            "Install it in your current env:\n"
            "  pip install imageio imageio-ffmpeg\n"
        ) from e

    writer = imageio.get_writer(str(mp4_path), fps=float(fps), codec=str(codec), quality=int(quality))
    try:
        for frame in video_u8:
            writer.append_data(frame)
    finally:
        writer.close()

    return mp4_path

