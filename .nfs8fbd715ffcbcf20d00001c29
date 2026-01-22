#!/usr/bin/env bash
set -euo pipefail

# Text-to-video batch runner for LTX-Video inference.
#
# Uses the repo's compatibility entrypoint:
#   python ltx_video/run_inference.py ...
#
# This is the text-only equivalent of a conditioning run like:
#   python inference.py --prompt "PROMPT" --conditioning_media_paths IMAGE_PATH ...
#
# Defaults are chosen to match the repo's common settings, but everything can be
# overridden via environment variables.
#
# Examples:
#   bash run_text_to_video_prompts.sh
#   PROMPT_FILE=prompt.txt SEED=2026 HEIGHT=512 WIDTH=768 NUM_FRAMES=81 bash run_text_to_video_prompts.sh
#   PIPELINE_CONFIG=configs/ltxv-13b-0.9.8-distilled.yaml bash run_text_to_video_prompts.sh
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Default to try.txt if present; otherwise prompt.txt. You can always override:
#   PROMPT_FILE=prompt.txt bash run_text_to_video_prompts.sh
PROMPT_FILE="${PROMPT_FILE:-}"
if [[ -z "$PROMPT_FILE" ]]; then
  if [[ -f "try.txt" ]]; then
    PROMPT_FILE="try.txt"
  else
    PROMPT_FILE="prompt.txt"
  fi
fi
PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/ltxv-2b-0.9.6-dev.yaml}"
export PIPELINE_CONFIG
SEED="${SEED:-2026}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-768}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FRAME_RATE="${FRAME_RATE:-16}"
# Match pipeline.py behavior (no negative prompt / no CFG-style negative conditioning).
# You can override, e.g. NEGATIVE_PROMPT="worst quality, blurry" bash run_text_to_video_prompts.sh
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# ---------------------------------------------------------------------
# Optional: LTX hosted API mode (instead of local python inference)
# ---------------------------------------------------------------------
# Example:
#   USE_LTX_API=1 LTX_API_KEY="..." bash run_text_to_video_prompts.sh
USE_LTX_API="${USE_LTX_API:-0}"
LTX_API_URL="${LTX_API_URL:-https://api.ltx.video/v1/text-to-video}"
LTX_API_KEY="${LTX_API_KEY:-}"
LTX_API_MODEL="${LTX_API_MODEL:-ltx-2-pro}"
LTX_API_DURATION="${LTX_API_DURATION:-8}"          # seconds
LTX_API_RESOLUTION="${LTX_API_RESOLUTION:-1920x1080}"

OUTPUT_ROOT="${OUTPUT_ROOT:-t2v}"
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"
RUN_DIR="${OUTPUT_ROOT}/t2v_prompts_${RUN_ID}"

# ---------------------------------------------------------------------
# Match older codebase config + robust checkpoint resolution
# (This mirrors the pipeline.py style: load YAML, read checkpoint_path,
# resolve local path or download from HF using basename.)
# ---------------------------------------------------------------------
config_path="${PIPELINE_CONFIG}"  # Match older codebase config
if [[ -z "${CKPT_PATH:-}" ]]; then
  set +e
  resolved_ckpt_path="$(
    python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
try:
    from ltx_video.ltx_video.inference import load_pipeline_config  # type: ignore
except ModuleNotFoundError:
    from ltx_video.inference import load_pipeline_config  # type: ignore

config_path = os.environ.get("PIPELINE_CONFIG", "configs/ltxv-2b-0.9.6-dev.yaml")
cfg = load_pipeline_config(config_path)
ckpt_name = cfg["checkpoint_path"]  # load the checkpoint name from the config file

def resolve_checkpoint(name: str):
    """
    Resolve a checkpoint reference.
    - If `name` is an existing local path, return it.
    - Otherwise, try to download from HF hub using the basename (handles YAMLs
      that store absolute paths).
    """
    try:
        p = Path(name)
        if p.exists() and p.is_file():
            print(str(p))
            return

        candidate = p.name  # basename for HF hub download
        # print(f"Attempting HF download checkpoint: {candidate} (from {name})")
        path = hf_hub_download("Lightricks/LTX-Video", candidate)
        print(path)
        return
    except Exception:
        return

ckpt_path = os.getenv("CKPT_PATH") or (ckpt_name if os.path.isfile(ckpt_name) else None)
if ckpt_path is None:
    resolve_checkpoint(ckpt_name)
else:
    print(ckpt_path)
PY
  )"
  set -e

  if [[ -n "$resolved_ckpt_path" && -f "$resolved_ckpt_path" ]]; then
    export CKPT_PATH="$resolved_ckpt_path"
    echo "✓ Resolved CKPT_PATH from config ($config_path): $CKPT_PATH"
  else
    echo "⚠️  Could not resolve CKPT_PATH from config ($config_path); inference will try its own download logic." >&2
  fi
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "ERROR: PROMPT_FILE not found: $PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "======================================================================"
echo "LTX-Video TEXT-TO-VIDEO batch inference"
echo "Prompt file:     $PROMPT_FILE"
echo "Pipeline config: $PIPELINE_CONFIG"
echo "Seed:            $SEED"
echo "Resolution:      ${WIDTH}x${HEIGHT}"
echo "Num frames:      $NUM_FRAMES"
echo "Frame rate:      $FRAME_RATE"
echo "Output dir:      $RUN_DIR"
echo "======================================================================"
echo

prompt_idx=0
while IFS= read -r line || [[ -n "$line" ]]; do
  prompt="$(echo "$line" | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//')"
  [[ -z "$prompt" ]] && continue

  prompt_idx=$((prompt_idx + 1))

  # Create a stable, filesystem-safe folder name per prompt (first 80 chars).
  slug="$(echo "$prompt" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' \
    | cut -c1-80)"
  [[ -z "$slug" ]] && slug="prompt_${prompt_idx}"

  out_dir="${RUN_DIR}/p$(printf "%03d" "$prompt_idx")_${slug}"
  mkdir -p "$out_dir"

  echo "---------------------------------------------------------------------"
  echo "[$prompt_idx] $prompt"
  echo "Output: $out_dir"
  echo "---------------------------------------------------------------------"

  if [[ "$USE_LTX_API" == "1" ]]; then
    if [[ -z "$LTX_API_KEY" ]]; then
      echo "ERROR: USE_LTX_API=1 but LTX_API_KEY is empty." >&2
      exit 1
    fi

    # JSON-escape the prompt safely without requiring jq.
    prompt_json="$(python - <<PY
import json
import sys
print(json.dumps(sys.argv[1]))
PY
"$prompt")"

    out_mp4="${out_dir}/video_api_${prompt_idx}.mp4"
    payload="$(cat <<JSON
{
  "prompt": ${prompt_json},
  "model": "${LTX_API_MODEL}",
  "duration": ${LTX_API_DURATION},
  "resolution": "${LTX_API_RESOLUTION}"
}
JSON
)"

    echo "Calling LTX API: $LTX_API_URL"
    echo "Saving: $out_mp4"
    curl -fsSL \
      -X POST "$LTX_API_URL" \
      -H "Authorization: Bearer ${LTX_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      -o "$out_mp4"

  else
    # IMPORTANT: No conditioning args => pure text-to-video (local inference).
    python ltx_video/run_inference.py \
      --prompt "$prompt" \
      --output_path "$out_dir" \
      --pipeline_config "$PIPELINE_CONFIG" \
      --seed "$SEED" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num_frames "$NUM_FRAMES" \
      --frame_rate "$FRAME_RATE" \
      --negative_prompt "$NEGATIVE_PROMPT"
  fi

done < "$PROMPT_FILE"

echo
echo "✅ Done. Outputs saved under: $RUN_DIR"

