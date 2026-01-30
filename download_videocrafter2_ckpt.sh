#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CKPT_DIR="$ROOT/base_512_v2"
CKPT_PATH="$CKPT_DIR/model.ckpt"
TMP_PATH="$CKPT_PATH.partial"

# Official link referenced by scaling-noise README:
# https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt
URL="https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt"

mkdir -p "$CKPT_DIR"

echo "Downloading VideoCrafter2 checkpoint to: $CKPT_PATH"
echo "Source: $URL"

# Use a temp file then atomic move to avoid leaving a corrupt ckpt if the download is interrupted.
wget -O "$TMP_PATH" --continue "$URL"
mv -f "$TMP_PATH" "$CKPT_PATH"

echo "Verifying checkpoint with torch.load(...) (CPU)..."
/home/ubuntu/anaconda3/bin/conda run -n scaling-noise python - <<'PY'
import torch
path = "base_512_v2/model.ckpt"
obj = torch.load(path, map_location="cpu")
print("OK: torch.load succeeded. top-level type:", type(obj))
if isinstance(obj, dict):
    print("keys:", list(obj.keys())[:10])
PY

echo "Done."

