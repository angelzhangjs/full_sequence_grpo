#!/usr/bin/env python3
"""
Run LTX hosted Text-to-Video API for one prompt or a prompt file.

This script intentionally does NOT embed API keys. Provide the key via env var:
  export LTX_API_KEY="..."

Examples:
  python run_ltx_api_prompts.py --prompt "A majestic eagle soaring through clouds at sunset"
  python run_ltx_api_prompts.py --prompt_file prompt.txt --model ltx-2-pro --duration 8 --resolution 1920x1080

Outputs:
  ./t2v_api/api_prompts_{timestamp}/p001_.../video_api_1.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _slugify(text: str, max_len: int = 80) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return (s[:max_len] or "prompt").strip("-") or "prompt"


def _iter_prompts(prompt: str | None, prompt_file: str | None) -> Iterable[str]:
    if prompt and prompt.strip():
        yield prompt.strip()
        return
    if not prompt_file:
        return
    p = Path(prompt_file)
    if not p.is_file():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t:
            yield t


def call_ltx_api(
    *,
    api_url: str,
    api_key: str,
    prompt: str,
    model: str,
    duration: int,
    resolution: str,
    timeout_s: int = 600,
) -> bytes:
    payload = {
        "prompt": prompt,
        "model": model,
        "duration": int(duration),
        "resolution": resolution,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        api_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "video/mp4,application/octet-stream,application/json",
        },
    )
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch call LTX hosted text-to-video API.")
    ap.add_argument("--prompt", type=str, default=None, help="Single prompt (overrides --prompt_file)")
    ap.add_argument("--prompt_file", type=str, default="prompt.txt", help="Prompt file (one prompt per line)")
    ap.add_argument("--api_url", type=str, default=os.getenv("LTX_API_URL", "https://api.ltx.video/v1/text-to-video"))
    ap.add_argument("--model", type=str, default=os.getenv("LTX_API_MODEL", "ltx-2-pro"))
    ap.add_argument("--duration", type=int, default=int(os.getenv("LTX_API_DURATION", "8")))
    ap.add_argument("--resolution", type=str, default=os.getenv("LTX_API_RESOLUTION", "1920x1080"))
    ap.add_argument("--out_root", type=str, default=os.getenv("OUTPUT_ROOT", "t2v_api"))
    ap.add_argument("--run_id", type=str, default=os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--timeout_s", type=int, default=int(os.getenv("LTX_API_TIMEOUT_S", "600")))
    args = ap.parse_args(argv)

    api_key = os.getenv("LTX_API_KEY", "").strip()
    if not api_key:
        print("ERROR: LTX_API_KEY is not set. Export it first: export LTX_API_KEY='...'", file=sys.stderr)
        return 2

    prompts = list(_iter_prompts(args.prompt, args.prompt_file))
    if not prompts:
        print("ERROR: No prompts provided (empty --prompt and empty/missing --prompt_file).", file=sys.stderr)
        return 2

    run_dir = Path(args.out_root) / f"api_prompts_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("LTX Hosted API text-to-video")
    print(f"API URL:     {args.api_url}")
    print(f"Model:       {args.model}")
    print(f"Duration:    {args.duration}s")
    print(f"Resolution:  {args.resolution}")
    print(f"Prompts:     {len(prompts)}")
    print(f"Output dir:  {run_dir}")
    print("======================================================================")

    for idx, prompt in enumerate(prompts, start=1):
        slug = _slugify(prompt)
        out_dir = run_dir / f"p{idx:03d}_{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"video_api_{idx}.mp4"

        print("\n" + "-" * 70)
        print(f"[{idx}/{len(prompts)}] {prompt}")
        print(f"Saving to: {out_mp4}")
        print("-" * 70)

        try:
            data = call_ltx_api(
                api_url=args.api_url,
                api_key=api_key,
                prompt=prompt,
                model=args.model,
                duration=args.duration,
                resolution=args.resolution,
                timeout_s=args.timeout_s,
            )
        except HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            print(f"ERROR: HTTP {e.code} {e.reason}\n{err_body}", file=sys.stderr)
            continue
        except URLError as e:
            print(f"ERROR: Network error: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            continue

        # Some APIs return JSON errors with 200; detect obvious JSON.
        if data[:1] in (b"{", b"["):
            try:
                obj = json.loads(data.decode("utf-8", errors="replace"))
                print(f"ERROR: API returned JSON instead of MP4:\n{json.dumps(obj, indent=2)[:4000]}", file=sys.stderr)
                continue
            except Exception:
                pass

        out_mp4.write_bytes(data)
        print(f"✅ Saved: {out_mp4} ({out_mp4.stat().st_size/1024/1024:.2f} MB)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

