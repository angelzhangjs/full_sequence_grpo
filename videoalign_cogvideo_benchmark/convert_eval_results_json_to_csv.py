#!/usr/bin/env python3
"""
Convert PBench evaluation results JSON to CSV.

Input JSON format (per dimension):
  {
    "aesthetic_quality": [overall_score, [{"video_path": "...", "video_results": ...}, ...]],
    ...
  }

Output CSV (one row per video per metric):
  video_id,seed,metric,score,video_path

Also appends summary rows:
  __SUMMARY__,,metric,overall_score,
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def _parse_video_id_seed(video_path: str) -> Tuple[str, str]:
    stem = Path(video_path).stem  # e.g. cogvideo_baseline__42
    if "__" in stem:
        video_id, seed = stem.split("__", 1)
        return video_id, seed
    return stem, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to *_eval_results.json")
    ap.add_argument("--output", default=None, help="Path to output .csv (default: alongside input)")
    ap.add_argument("--no-summary", action="store_true", help="Do not write __SUMMARY__ rows")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = json.loads(in_path.read_text())

    rows = []
    for metric, payload in data.items():
        if not isinstance(payload, list) or len(payload) != 2:
            continue
        overall, per_video = payload

        if not args.no_summary:
            rows.append(
                {
                    "video_id": "__SUMMARY__",
                    "seed": "",
                    "metric": str(metric),
                    "score": overall,
                    "video_path": "",
                }
            )

        if not isinstance(per_video, list):
            continue
        for item in per_video:
            if not isinstance(item, dict):
                continue
            video_path = str(item.get("video_path", ""))
            score = item.get("video_results", None)
            video_id, seed = _parse_video_id_seed(video_path)
            rows.append(
                {
                    "video_id": video_id,
                    "seed": seed,
                    "metric": str(metric),
                    "score": score,
                    "video_path": video_path,
                }
            )

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "seed", "metric", "score", "video_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

