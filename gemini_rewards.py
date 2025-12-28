import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None
    
def decode_x0_to_video(
    x0_latent,
    pipeline,
    latent_height,
    latent_width,
    num_frames,
    height,
    width,
    is_patchified=True,
):
    """
    Decode x0 latent to video
    
    Args:
        x0_latent: Latent tensor (patchified or video format)
        pipeline: LTX-Video pipeline
        latent_height, latent_width: Latent dimensions
        num_frames, height, width: Target video dimensions
        is_patchified: Whether x0_latent is patchified
        
    Returns:
        video: [1, 3, T, H, W] tensor
    """
    with torch.no_grad():
        # Unpatchify if needed
        if is_patchified:
            x0_video = pipeline.patchifier.unpatchify(
                x0_latent,
                output_height=latent_height,
                output_width=latent_width,
                out_channels=4,
            )
        else:
            x0_video = x0_latent
        
        # Decode to pixels
        target_shape = (1, 3, num_frames, height, width)
        
        video = pipeline.vae.decode(
            x0_video / pipeline.vae.config.scaling_factor,
            target_shape=target_shape,
            timestep=torch.tensor([0.0], device="cuda"),
        ).sample
        
        return video
    
def reward_function(video_x0):
    return video_x0.mean()


# ---------- Gemini Flash ranking helpers ----------

def _require_genai() -> None:
    if genai is None:
        raise ImportError(
            "google-generativeai is required for Gemini ranking. "
            "Install with `pip install google-generativeai`."
        )


def _load_video(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path for video, got directory: {path}")
    return path


def _ensure_model(model_name: str, api_key: Optional[str]) -> "genai.GenerativeModel":
    _require_genai()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass api_key.")
    genai.configure(api_key=key)
    return genai.GenerativeModel(model_name)


def _extract_json(text: str) -> Dict[str, float]:
    """
    Parse a JSON payload from the model response. We prefer strict JSON but
    gracefully handle extra text by looking for the first and last braces.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

def score_video_with_gemini(
    video_path: str,
    *,
    model_name: str = "gemini-1.5-flash",
    api_key: Optional[str] = None,
    request_timeout: int = 120,
) -> Dict[str, float]:
    """
    Score a single video for motion dynamics and physical realism using Gemini Flash.

    Returns a dict containing motion_dynamics, physical_properties, overall, and
    the raw text response. Overall is the simple average of the two scores if
    the model does not return it.
    """
    model = _ensure_model(model_name, api_key)
    video_file = genai.upload_file(_load_video(Path(video_path)))

    prompt = (
        "You are judging a short video for action quality.\n"
        "- motion_dynamics: smoothness, speed appropriateness, temporal coherence (0-10)\n"
        "- physical_properties: physics realism, collisions, gravity consistency (0-10)\n"
        "Respond ONLY with JSON using keys: motion_dynamics, physical_properties, overall.\n"
        "If unsure, make your best estimate."
    )

    response = model.generate_content(
        [video_file, prompt],
        request_options={"timeout": request_timeout},
    )

    # Gemini responses usually surface plain text in .text; fallback to parts otherwise.
    text = response.text
    if not text and response.candidates:
        parts = response.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") or "" for p in parts)

    parsed = _extract_json(text)
    motion = float(parsed.get("motion_dynamics", 0.0))
    physical = float(parsed.get("physical_properties", 0.0))
    overall = float(parsed.get("overall", (motion + physical) / 2 if (motion or physical) else 0.0))

    return {
        "motion_dynamics": motion,
        "physical_properties": physical,
        "overall": overall,
        "raw_response": text,
    }


def rank_videos_by_gemini(
    video_paths: Iterable[str],
    *,
    model_name: str = "gemini-1.5-flash",
    api_key: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Rank a list of videos, returning the path with the highest overall score
    for motion dynamics and physical properties.

    Args:
        video_paths: Iterable of video file paths.
        model_name: Gemini model to use (default: gemini-1.5-flash).
        api_key: Optional Gemini API key (falls back to GEMINI_API_KEY env).

    Returns:
        (best_path, score_dict) where score_dict includes motion_dynamics,
        physical_properties, overall, raw_response.
    """
    best_path: Optional[str] = None
    best_score: Optional[Dict[str, float]] = None

    for path in video_paths:
        score = score_video_with_gemini(path, model_name=model_name, api_key=api_key)
        if best_score is None or score["overall"] > best_score["overall"]:
            best_path = path
            best_score = score

    if best_path is None or best_score is None:
        raise ValueError("No videos were provided for ranking.")

    return best_path, best_score
