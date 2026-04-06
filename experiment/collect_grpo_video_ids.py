#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CATEGORY_CODES = [
    ("FA", 1, 10),
    ("PR", 11, 20),
    ("SW", 21, 30),
    ("CP", 31, 40),
    ("SP", 41, 50),
    ("RO", 51, 60),
    ("BO", 61, 70),
    ("FL", 71, 80),
    ("FT", 81, 90),
    ("SL", 91, 100),
]


def index_to_video_id(i: int) -> str:
    for code, start, end in CATEGORY_CODES:
        if start <= i <= end:
            within = i - start + 1
            return f"{code}_{within:03d}"
    raise ValueError(f"Prompt index out of range: {i}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-prompt GRPO videos into flat category-coded IDs")
    ap.add_argument("--run-root", required=True, help="Root run directory containing pXXX prompt folders")
    ap.add_argument("--video-name", default="wan_grpo.mp4", help="Name of video file under each grpo/ folder")
    ap.add_argument("--subdir", default="grpo", help="Subdirectory under each prompt folder to look in (e.g. grpo or baseline)")
    ap.add_argument("--output-dir-name", default="grpo_video_id", help="Name of output subdirectory under experiment/")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument(
        "--output-prefix",
        default="",
        help=(
            "Optional prefix for the experiment output directory name, e.g. "
            "CogVideoX-2b_grpo_20260326_074123 -> "
            "<prefix>_<prompts_folder>_<output_dir_name> under experiment/"
        ),
    )
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    repo_root = Path(__file__).resolve().parents[1]
    experiment_root = repo_root / "experiment"
    prefix = str(args.output_prefix).strip().strip("_")
    name_parts = [p for p in (prefix, run_root.name, args.output_dir_name) if p]
    output_dir = experiment_root / "_".join(name_parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    linked = 0
    missing = []

    for i in range(1, 101):
        folder = next(run_root.glob(f"p{i:03d}_*"), None)
        if folder is None:
            missing.append(f"p{i:03d}")
            continue

        src = folder / args.subdir / args.video_name
        if not src.exists():
            missing.append(str(src))
            continue

        video_id = index_to_video_id(i)
        dst = output_dir / f"{video_id}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if args.copy:
            import shutil
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

        prompt_path = folder / "prompt.txt"
        prompt = prompt_path.read_text().strip() if prompt_path.exists() else ""
        manifest.append({
            "video_id": video_id,
            "prompt_en": prompt,
            "source_video": str(src),
        })
        linked += 1

    manifest_path = output_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Created {linked} video links in: {output_dir}")
    print(f"Prompt manifest: {manifest_path}")
    if missing:
        print("Missing entries:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
