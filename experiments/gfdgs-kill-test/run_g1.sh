#!/usr/bin/env bash
set -u

repo=/root/autodl-tmp/gfdgs/src/AD-GS
python=/root/autodl-tmp/gfdgs/envs/adgs-official/bin/python
source_dir=/root/autodl-tmp/gfdgs/data/processed/waymo/scene090
output_root=/root/autodl-tmp/gfdgs/output/kill-test
evidence_root=/root/autodl-tmp/gfdgs/evidence/gfdgs-kill-test

run_arm() {
    label=$1
    config=$2
    output="$output_root/scene090-$label-1k"
    train_log="$evidence_root/scene090-$label-1k.train.log"
    render_log="$evidence_root/scene090-$label-1k.render.log"
    memory_log="$evidence_root/scene090-$label-1k.memory.csv"
    summary="$evidence_root/scene090-$label-1k.summary.txt"

    if [ -e "$output" ]; then
        printf 'Refusing to overwrite %s\n' "$output" >&2
        return 2
    fi

    cd "$repo" || return 2
    SECONDS=0
    env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 "$python" train.py \
        -c "$config" -s "$source_dir" -m "$output" \
        --data_device cuda:0 --iterations 1000 --save_iterations 1000 \
        > "$train_log" 2>&1 &
    train_pid=$!

    printf 'unix_seconds,memory_mib\n' > "$memory_log"
    while kill -0 "$train_pid" 2>/dev/null; do
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        printf '%s,%s\n' "$(date +%s)" "$memory" >> "$memory_log"
        sleep 1
    done
    wait "$train_pid"
    train_exit=$?
    train_seconds=$SECONDS
    peak_memory=$(awk -F, 'NR > 1 && $2 + 0 > max { max = $2 + 0 } END { print max + 0 }' "$memory_log")
    printf 'train_exit=%s\ntrain_seconds=%s\npeak_memory_mib=%s\n' \
        "$train_exit" "$train_seconds" "$peak_memory" > "$summary"
    if [ "$train_exit" -ne 0 ]; then
        return "$train_exit"
    fi

    SECONDS=0
    env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 "$python" render.py \
        -c "$config" -m "$output" --iteration 1000 --skip_train \
        > "$render_log" 2>&1
    render_exit=$?
    render_seconds=$SECONDS
    printf 'render_exit=%s\nrender_seconds=%s\n' \
        "$render_exit" "$render_seconds" >> "$summary"
    return "$render_exit"
}

run_arm baseline ./arguments/waymo.py || exit $?
run_arm oracle ./arguments/waymo_oracle_contact.py || exit $?
