#!/usr/bin/env python3
"""
Run baseline LTX-Video inference (with per-timestep intermediate x0 videos) and/or
GRPO-style training once per prompt, sequentially.

This script loops over a prompt file (default: prompt.txt). For each non-empty line:
  - runs baseline_intermediate_videos.py with --prompt set to that line
  - writes outputs into a per-prompt output directory

Usage:
  python run_prompts_sequential.py
  python run_prompts_sequential.py --prompt-file pipeline.txt
  python run_prompts_sequential.py --out-root runs/grpo_seq

Notes:
  - Run this inside your conda env (ltx-grpo).
  - If you want deterministic behavior, set DETERMINISTIC=1 (and CUBLAS_WORKSPACE_CONFIG).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


def _read_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            prompts.append(text)
    return prompts


def _slug(s: str, max_len: int = 48) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] or "prompt"

def _gpu_cleanup(python_exe: str) -> None:
    """
    Best-effort GPU memory cleanup between prompt runs.
    Note: pipeline.py is executed in a separate process; GPU memory is normally
    released when that process exits. This is just an extra safety step.
    """
    code = (
        "import gc; gc.collect();\n"
        "try:\n"
        "  import torch\n"
        "  if torch.cuda.is_available():\n"
        "    torch.cuda.empty_cache();\n"
        "    torch.cuda.ipc_collect();\n"
        "    print('✓ GPU cache cleared')\n"
        "except Exception as e:\n"
        "  print(f'gpu cleanup skipped: {e!r}')\n"
    )
    subprocess.run([python_exe, "-c", code], check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run baseline_intermediate_videos.py sequentially for each prompt line.")
    ap.add_argument("--prompt-file", default="prompt.txt", help="Text file with one prompt per line.")
    ap.add_argument(
        "--out-root",
        default=None,
        help="Root output directory. Defaults to grpo_baseline<timestamp>/ (e.g. grpo_baseline20260121_123456/).",
    )
    ap.add_argument(
        "--pipeline-config",
        default="configs/ltxv-2b-0.9.6-dev.yaml",
        help="YAML config for the baseline pipeline (passed through).",
    )
    ap.add_argument(
        "--mode",
        default="both",
        choices=["baseline_intermediates", "grpo_train", "both"],
        help="Which mode to run inside baseline_intermediate_videos.py for each prompt.",
    )
    ap.add_argument("--save-every", type=int, default=1, help="Save every N denoising steps for baseline intermediates.")
    ap.add_argument("--seed", type=int, default=2026, help="Seed to pass to baseline_intermediate_videos.py.")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--frame-rate", type=int, default=16)
    # GRPO options (used when --mode is grpo_train or both)
    ap.add_argument("--num-inference-steps", type=int, default=40)
    ap.add_argument("--num-grpo-steps", type=int, default=25)
    ap.add_argument("--num-rollouts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    # Accept both dashed and underscored variants for compatibility with older scripts.
    ap.add_argument("--attn1-blocks", "--attn1_blocks", dest="attn1_blocks", default="11,12,13,14")
    ap.add_argument("--attn2-blocks", "--attn2_blocks", dest="attn2_blocks", default="27")
    ap.add_argument("--rollout-noise-scale", type=float, default=0.5)
    ap.add_argument("--normalize-advantages", type=int, default=1)
    ap.add_argument("--use-grpo-kl", type=int, default=0)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument(
        "--grpo-from-start",
        action="store_true",
        help="If set, runs GRPO from the beginning of the timestep schedule (sets GRPO_FROM_START=1).",
    )
    ap.add_argument(
        "--python",
        default="python",
        help="Python executable to use (default: python). Use a full path if needed.",
    )
    args = ap.parse_args()

    prompt_file = Path(args.prompt_file).resolve()
    prompts = _read_prompts(prompt_file)
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {prompt_file}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Convention for this sequential baseline/GRPO runner: grpo_baseline{timestamp}
    out_root = Path(args.out_root or f"grpo_baseline{run_id}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Prompt file: {prompt_file}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output root: {out_root}")

    for i, prompt in enumerate(prompts):
        prompt_slug = _slug(prompt)
        prompt_dir = out_root / f"p{i:03d}_{prompt_slug}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        env = os.environ.copy()
        # Reduce CUDA allocator fragmentation (must be set before torch import in subprocess).
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Optional: run GRPO from the start of timesteps (affects baseline_intermediate_videos.py and pipeline.py).
        if args.grpo_from_start and args.mode in ("grpo_train", "both"):
            env["GRPO_FROM_START"] = "1"

        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(prompts)}] Running baseline_intermediate_videos.py")
        print(f"Prompt: {prompt}")
        print(f"Output: {prompt_dir}")
        print("=" * 80)

        cmd = [
            args.python,
            "baseline_intermediate_videos.py",
            "--mode",
            args.mode,
            "--prompt",
            prompt,
            "--pipeline_config",
            args.pipeline_config,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num_frames",
            str(args.num_frames),
            "--frame_rate",
            str(args.frame_rate),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(prompt_dir),
            "--save_every",
            str(args.save_every),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--num_grpo_steps",
            str(args.num_grpo_steps),
            "--num_rollouts",
            str(args.num_rollouts),
            "--lr",
            str(args.lr),
            "--attn1_blocks",
            args.attn1_blocks,
            "--attn2_blocks",
            args.attn2_blocks,
            "--rollout_noise_scale",
            str(args.rollout_noise_scale),
            "--normalize_advantages",
            str(args.normalize_advantages),
            "--use_grpo_kl",
            str(args.use_grpo_kl),
            "--kl_beta",
            str(args.kl_beta),
        ]
        proc = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parent))
        if proc.returncode != 0:
            print(f"ERROR: baseline_intermediate_videos.py failed for prompt {i} (exit={proc.returncode})")
            return proc.returncode

        # Extra safety: clear GPU allocator caches between prompts.
        _gpu_cleanup(args.python)

    print("\n✅ All prompts completed.")
    print(f"Outputs written under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

