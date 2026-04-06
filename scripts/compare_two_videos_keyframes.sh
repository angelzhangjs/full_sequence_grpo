#!/usr/bin/env bash
# Extract K evenly spaced frames from two videos. Same timestamps (min duration).
#
# Combined (default): one PNG, top row = baseline, bottom row = GRPO.
# Separate (--separate): two PNGs, each a single horizontal strip of K frames.
#
# Usage:
#   compare_two_videos_keyframes.sh <baseline.mp4> <grpo.mp4> [out.png] [K] [--separate]
#   compare_two_videos_keyframes.sh <baseline.mp4> <grpo.mp4> --out-dir <dir> [--frames K] [--separate]
#
# With --separate, [out] is a path prefix (optional .png stripped):
#   writes ${prefix}_baseline_keyframes.png and ${prefix}_grpo_keyframes.png
#
# With --out-dir <dir> (implies --separate): writes
#   <dir>/baseline_keyframes.png and <dir>/grpo_keyframes.png
#
# Examples (omit [out] to write next to <baseline.mp4>):
#   compare_two_videos_keyframes.sh path/baseline.mp4 path/wan_grpo.mp4
#   compare_two_videos_keyframes.sh path/baseline.mp4 path/wan_grpo.mp4 5 --separate
#   compare_two_videos_keyframes.sh b.mp4 g.mp4 --out-dir ./p001_prompt --frames 2 --separate
#   compare_two_videos_keyframes.sh b.mp4 g.mp4 custom_name.png 5
#
# Requires: ffmpeg

set -euo pipefail

BASE="${1:?usage: $0 <baseline.mp4> <grpo.mp4> [out_or_prefix] [K] [--separate]}"
GRPO="${2:?usage: $0 <baseline.mp4> <grpo.mp4> [out_or_prefix] [K] [--separate]}"
shift 2

OUT=""
K="5"
SEPARATE=0
KEYFRAME_OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --separate)
      SEPARATE=1
      shift
      ;;
    --out-dir)
      KEYFRAME_OUT_DIR="${2:?error: --out-dir requires a directory}"
      SEPARATE=1
      shift 2
      ;;
    --frames)
      K="${2:?error: --frames requires a positive integer}"
      shift 2
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        K="$1"
      elif [[ -z "${OUT}" ]]; then
        OUT="$1"
      else
        echo "error: unexpected argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

# Default output directory = folder containing baseline video (same as typical experiment layout).
FIXED_SEPARATE=0
OUT_DIR="$(dirname "$BASE")"
if [[ -z "$OUT" ]]; then
  if [[ "$SEPARATE" == 1 ]]; then
    [[ -z "${KEYFRAME_OUT_DIR}" ]] && FIXED_SEPARATE=1
  else
    OUT="${OUT_DIR}/baseline_vs_grpo_keyframes.png"
  fi
fi

if [[ ! -f "$BASE" ]] || [[ ! -f "$GRPO" ]]; then
  echo "error: missing video file" >&2
  exit 1
fi
if ! [[ "$K" =~ ^[0-9]+$ ]] || [[ "$K" -lt 1 ]]; then
  echo "error: num_frames must be a positive integer" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

dur_b=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$BASE")
dur_g=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$GRPO")
dur=$(awk -v a="$dur_b" -v b="$dur_g" 'BEGIN { a+=0;b+=0; print (a<b)?a:b }')
if awk -v d="$dur" 'BEGIN { exit !(d > 0) }'; then
  :
else
  echo "error: could not read duration" >&2
  exit 1
fi

W=320
H=180
VF="scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:black"

for i in $(seq 0 $((K - 1))); do
  t=$(awk -v i="$i" -v k="$K" -v d="$dur" 'BEGIN {
    if (k <= 1) { print 0; exit }
    print (i + 0.5) * d / k
  }')
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$BASE" -ss "$t" -vf "$VF" -vframes 1 "$TMP/b_$i.png"
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$GRPO" -ss "$t" -vf "$VF" -vframes 1 "$TMP/g_$i.png"
done

hstack_pngs() {
  local out="$1"
  shift
  local n=$#
  local args=()
  local fc=""
  local idx=0
  for f in "$@"; do
    args+=(-i "$f")
    if [[ "$idx" -lt $((n - 1)) ]]; then
      fc+="[${idx}:v]"
    fi
    idx=$((idx + 1))
  done
  fc+="[$((n - 1)):v]hstack=inputs=${n}[outv]"
  ffmpeg -nostdin -hide_banner -loglevel error -y "${args[@]}" -filter_complex "$fc" -map "[outv]" -frames:v 1 "$out"
}

if [[ "$SEPARATE" == 1 ]]; then
  if [[ -n "${KEYFRAME_OUT_DIR}" ]]; then
    mkdir -p "$KEYFRAME_OUT_DIR"
    out_b="${KEYFRAME_OUT_DIR}/baseline_keyframes.png"
    out_g="${KEYFRAME_OUT_DIR}/grpo_keyframes.png"
  elif [[ "$FIXED_SEPARATE" == 1 ]]; then
    mkdir -p "$OUT_DIR"
    out_b="${OUT_DIR}/baseline_keyframes.png"
    out_g="${OUT_DIR}/grpo_keyframes.png"
  else
    prefix="$OUT"
    [[ "$prefix" == *.png ]] && prefix="${prefix%.png}"
    dir=$(dirname "$prefix")
    mkdir -p "$dir"
    out_b="${prefix}_baseline_keyframes.png"
    out_g="${prefix}_grpo_keyframes.png"
  fi
  b_files=()
  g_files=()
  for i in $(seq 0 $((K - 1))); do
    b_files+=("$TMP/b_$i.png")
    g_files+=("$TMP/g_$i.png")
  done
  hstack_pngs "$out_b" "${b_files[@]}"
  hstack_pngs "$out_g" "${g_files[@]}"
  echo "Wrote: $out_b"
  echo "Wrote: $out_g (${K} frames each, single row)"
else
  args=()
  fc=""
  for i in $(seq 0 $((K - 1))); do
    args+=(-i "$TMP/b_$i.png")
  done
  for i in $(seq 0 $((K - 1))); do
    args+=(-i "$TMP/g_$i.png")
  done
  for i in $(seq 0 $((K - 2))); do
    fc+="[${i}:v]"
  done
  fc+="[$((K - 1)):v]hstack=inputs=${K}[row0];"
  for i in $(seq 0 $((K - 2))); do
    fc+="[$((K + i)):v]"
  done
  fc+="[$((2 * K - 1)):v]hstack=inputs=${K}[row1];[row0][row1]vstack[outv]"
  mkdir -p "$(dirname "$OUT")"
  ffmpeg -nostdin -hide_banner -loglevel error -y "${args[@]}" -filter_complex "$fc" -map "[outv]" -frames:v 1 "$OUT"
  echo "Wrote: $OUT (${K} columns × 2 rows; top=baseline, bottom=GRPO)"
fi
