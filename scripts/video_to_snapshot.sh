#!/usr/bin/env bash
# Extract a representative frame from a video as PNG (middle of clip by time).
# Usage: video_to_snapshot.sh <video.mp4|video_path> [output.png]
# If output omitted: <video_basename>_snapshot.png next to the video.

set -euo pipefail

VIDEO="${1:?Usage: $0 <video> [out.png]}"
OUT="${2:-}"

if [[ ! -f "$VIDEO" ]]; then
  echo "video_to_snapshot: file not found: $VIDEO" >&2
  exit 1
fi

if [[ -z "$OUT" ]]; then
  d=$(dirname "$VIDEO")
  b=$(basename "$VIDEO")
  base="${b%.*}"
  OUT="${d}/${base}_snapshot.png"
fi

mkdir -p "$(dirname "$OUT")"

if command -v ffmpeg >/dev/null 2>&1; then
  dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO" 2>/dev/null || echo "0")
  # Middle timestamp (avoid exact 0 for some codecs)
  mid=$(awk -v d="$dur" 'BEGIN { if (d > 0.1) print d/2; else print 0.05 }')
  if ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$mid" -i "$VIDEO" -vframes 1 -q:v 2 "$OUT" 2>/dev/null; then
    echo "  Snapshot: $OUT"
    exit 0
  fi
  # Fallback: first frame
  if ffmpeg -nostdin -hide_banner -loglevel error -y -i "$VIDEO" -vframes 1 -q:v 2 "$OUT" 2>/dev/null; then
    echo "  Snapshot: $OUT"
    exit 0
  fi
  echo "video_to_snapshot: ffmpeg failed for $VIDEO" >&2
  exit 1
fi

# Python fallback (imageio)
python3 - "$VIDEO" "$OUT" <<'PY'
import sys
from pathlib import Path
vid, out = sys.argv[1], sys.argv[2]
Path(out).parent.mkdir(parents=True, exist_ok=True)
try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio  # type: ignore
try:
    frames = list(iio.imiter(vid))
except Exception as e:
    print(f"video_to_snapshot: failed to read {vid}: {e}", file=sys.stderr)
    sys.exit(1)
if not frames:
    print(f"video_to_snapshot: no frames in {vid}", file=sys.stderr)
    sys.exit(1)
mid = frames[len(frames) // 2]
iio.imwrite(out, mid)
print(f"  Snapshot: {out}")
PY
