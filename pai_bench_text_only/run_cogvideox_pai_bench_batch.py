#!/usr/bin/env python3
"""
Batch text-to-video for PAI-bench style prompts.

Reads a tab-separated file with header columns: video_id, prompt_en
(same format as cosmos_predict2_bench_video_prompts.txt) and writes:

  <output_dir>/<video_id>.mp4

Loads the CogVideoX pipeline once, then runs inference per row (much faster
than invoking cli_demo.py once per prompt).

Run from repo root (recommended):

  cd /path/to/angel-research
  conda activate cogvideox
  python pai_bench_text_only/run_cogvideox_pai_bench_batch.py \\
    --output-dir ./pai_bench_outputs/cogvideox2b \\
    --model-path THUDM/CogVideoX-2b

Multi-GPU: use --num-shards 8 --shard-id 0..7 on separate processes/machines.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.utils import export_to_video

REPO_ROOT = Path(__file__).resolve().parents[1]
_COG_INF = REPO_ROOT / "CogVideo" / "inference"
if str(_COG_INF) not in sys.path:
    sys.path.insert(0, str(_COG_INF))

from cli_demo import RESOLUTION_MAP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def iter_prompt_rows(tsv_path: Path):
    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if "video_id" not in fields or "prompt_en" not in fields:
            raise SystemExit(
                f"Expected columns video_id and prompt_en; got {reader.fieldnames!r}"
            )
        for row in reader:
            vid = (row.get("video_id") or "").strip()
            prompt = row.get("prompt_en") or ""
            if not vid:
                continue
            yield vid, prompt


def build_t2v_pipe(
    model_path: str,
    dtype: torch.dtype,
    lora_path: str | None,
    lora_rank: int,
    sequential_offload: bool,
):
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if lora_path:
        pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="pai_bench",
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    if sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def parse_args():
    default_tsv = Path(__file__).resolve().parent / "cosmos_predict2_bench_video_prompts.txt"
    p = argparse.ArgumentParser(
        description="PAI-bench batch T2V: TSV (video_id, prompt_en) -> video_id.mp4"
    )
    p.add_argument(
        "--prompts-tsv",
        type=Path,
        default=default_tsv,
        help="Tab-separated file with video_id and prompt_en columns",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write <video_id>.mp4 files",
    )
    p.add_argument("--model-path", type=str, default="THUDM/CogVideoX-2b")
    p.add_argument("--lora-path", type=str, default=None)
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    p.add_argument(
        "--no-sequential-offload",
        action="store_true",
        help="Keep full model on GPU (faster if VRAM fits; default uses sequential CPU offload)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose output mp4 already exists",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, only process the first N rows (after sharding)",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This worker index in [0, num-shards)",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split rows by row_index %% num_shards == shard_id",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.prompts_tsv.is_file():
        raise SystemExit(f"Prompts file not found: {args.prompts_tsv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_prompt_rows(args.prompts_tsv))
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < num-shards")

    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    logger.info("Rows for this shard: %d (shard %d / %d)", len(rows), args.shard_id, args.num_shards)

    model_key = args.model_path.split("/")[-1].lower()
    if model_key not in RESOLUTION_MAP:
        raise SystemExit(
            f"Model {args.model_path!r} -> key {model_key!r} not in RESOLUTION_MAP; "
            f"extend CogVideo/inference/cli_demo.py RESOLUTION_MAP or use a supported checkpoint."
        )
    height, width = RESOLUTION_MAP[model_key]

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    pipe = build_t2v_pipe(
        args.model_path,
        dtype,
        args.lora_path,
        args.lora_rank,
        sequential_offload=not args.no_sequential_offload,
    )

    gen = torch.Generator().manual_seed(args.seed)

    for idx, (video_id, prompt) in enumerate(rows):
        out_path = args.output_dir / f"{video_id}.mp4"
        if args.skip_existing and out_path.is_file():
            logger.info("[%d/%d] skip existing %s", idx + 1, len(rows), out_path.name)
            continue
        logger.info("[%d/%d] generating %s", idx + 1, len(rows), out_path.name)
        result = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            generator=gen,
        )
        export_to_video(result.frames[0], str(out_path), fps=args.fps)

    logger.info("Done. Outputs under %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
