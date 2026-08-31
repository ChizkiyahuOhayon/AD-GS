#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    printf 'usage: %s SOURCE_DIR OUTPUT_ROOT EVIDENCE_ROOT [baseline|gfdgs|both]\n' "$0" >&2
    exit 2
fi

repo=$(cd "$(dirname "$0")/../.." && pwd)
source_dir=$1
output_root=$2
evidence_root=$3
arm=${4:-both}
python=${PYTHON:-python}
commit=$(git -C "$repo" rev-parse --short=7 HEAD)

if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    printf 'refusing dirty tracked worktree: %s\n' "$repo" >&2
    exit 2
fi
mkdir -p "$output_root" "$evidence_root"

run_arm() {
    label=$1
    config=$2
    output="$output_root/scene090-${label}-1k-${commit}"
    train_log="$evidence_root/scene090-${label}-1k-${commit}.train.log"
    render_log="$evidence_root/scene090-${label}-1k-${commit}.render.log"
    memory_log="$evidence_root/scene090-${label}-1k-${commit}.memory.csv"
    summary="$evidence_root/scene090-${label}-1k-${commit}.summary.txt"

    if [ -e "$output" ]; then
        printf 'refusing to overwrite %s\n' "$output" >&2
        return 2
    fi

    SECONDS=0
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 "$python" "$repo/train.py" \
        -c "$repo/$config" -s "$source_dir" -m "$output" \
        --data_device cuda:0 --iterations 1000 --save_iterations 1000 \
        > "$train_log" 2>&1 &
    train_pid=$!

    printf 'unix_seconds,memory_mib\n' > "$memory_log"
    while kill -0 "$train_pid" 2>/dev/null; do
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        printf '%s,%s\n' "$(date +%s)" "$memory" >> "$memory_log"
        sleep 1
    done
    if wait "$train_pid"; then
        train_exit=0
    else
        train_exit=$?
    fi
    train_seconds=$SECONDS
    peak_memory=$(awk -F, 'NR > 1 && $2 + 0 > max { max = $2 + 0 } END { print max + 0 }' "$memory_log")
    printf 'commit=%s\ntrain_exit=%s\ntrain_seconds=%s\npeak_memory_mib=%s\n' \
        "$commit" "$train_exit" "$train_seconds" "$peak_memory" > "$summary"
    if [ "$train_exit" -ne 0 ]; then
        return "$train_exit"
    fi

    SECONDS=0
    if PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 "$python" "$repo/render.py" \
        -c "$repo/$config" -m "$output" --iteration 1000 --skip_train \
        > "$render_log" 2>&1; then
        render_exit=0
    else
        render_exit=$?
    fi
    printf 'render_exit=%s\nrender_seconds=%s\n' \
        "$render_exit" "$SECONDS" >> "$summary"
    return "$render_exit"
}

case "$arm" in
    baseline) run_arm baseline arguments/waymo.py ;;
    gfdgs) run_arm gfdgs arguments/waymo_gfdgs.py ;;
    both)
        run_arm baseline arguments/waymo.py
        run_arm gfdgs arguments/waymo_gfdgs.py
        ;;
    *) printf 'unknown arm: %s\n' "$arm" >&2; exit 2 ;;
esac
