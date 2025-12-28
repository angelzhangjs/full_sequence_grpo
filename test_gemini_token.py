#!/usr/bin/env python3
"""
Quick check to verify a Gemini API key works using the new google-genai client.

Usage:
  python3 test_gemini_token.py --key YOUR_KEY_HERE
  # or rely on env var:
  GEMINI_API_KEY=YOUR_KEY python3 test_gemini_token.py
"""

import argparse
import os
import sys

try:
    from google import genai  # new client (pip install -U google-genai)
except ImportError:
    print("google-genai is not installed. Run: pip install -U google-genai", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Gemini API key with a simple request.")
    parser.add_argument("--key", help="Gemini API key (falls back to GEMINI_API_KEY env).")
    parser.add_argument(
        "--model",
        default="models/gemini-2.5-flash",
        help="Model to hit for the test (e.g., models/gemini-2.5-flash).",
    )
    args = parser.parse_args()

    api_key = args.key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No API key provided. Use --key or set GEMINI_API_KEY.", file=sys.stderr)
        return 1

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=args.model,
            contents="Respond with a short 'ok'.",
        )
        text = getattr(resp, "text", "") or ""
        if not text and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        print(f"Success: received response from {args.model!r}: {text[:200]}")
        return 0
    except Exception as exc:  # broad on purpose for quick debug
        print(f"Failed to call Gemini: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

