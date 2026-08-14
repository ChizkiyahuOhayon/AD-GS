#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <scene006-path> <ENV-001-evidence-dir> <new-run-dir> <new-evidence-dir>" >&2
    exit 2
fi

scene=$(realpath "$1")
env_evidence=$(realpath "$2")
run_dir=$(realpath -m "$3")
evidence=$(realpath -m "$4")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adgs_root=$(git -C "$script_dir" rev-parse --show-toplevel)
env_name=trust4d-adgs-baseline
official_base=9a208512e49c8ddbaa20387921d9648adcd21cb4

command -v conda >/dev/null
command -v nvidia-smi >/dev/null
git -C "$adgs_root" merge-base --is-ancestor "$official_base" HEAD
if [[ -n "$(git -C "$adgs_root" status --porcelain)" ]]; then
    echo "AD-GS working tree is not clean" >&2
    git -C "$adgs_root" status --short >&2
    exit 2
fi
if [[ ! -f "$env_evidence/smoke.json" \
    || ! -f "$env_evidence/evaluator.json" \
    || ! -f "$env_evidence/exitcode.txt" ]]; then
    echo "ENV-001 evidence is missing or failed: $env_evidence" >&2
    exit 2
fi
if [[ "$(<"$env_evidence/exitcode.txt")" != 0 ]]; then
    echo "ENV-001 evidence records a failed setup: $env_evidence" >&2
    exit 2
fi
conda run -n "$env_name" python -c \
    'import json,sys; assert all(json.load(open(p))["passed"] is True for p in sys.argv[1:])' \
    "$env_evidence/smoke.json" "$env_evidence/evaluator.json"
if [[ -e "$run_dir" || -e "$evidence" ]]; then
    echo "run and evidence directories must both be new" >&2
    exit 2
fi
for path in "$run_dir" "$evidence"; do
    case "$path/" in "$adgs_root/"*)
        echo "run and evidence directories must be outside the Git repository" >&2
        exit 2
    esac
done

while IFS=$'\t' read -r status path; do
    [[ -z "$status" ]] && continue
    if [[ "$status" != A ]]; then
        echo "official AD-GS source is not byte-identical: $status $path" >&2
        exit 2
    fi
    case "$path" in
        experiments.md|server.md|scripts/trust4d/*|tests/*) ;;
        *)
            echo "unexpected file relative to official AD-GS: $path" >&2
            exit 2
            ;;
    esac
done < <(git -C "$adgs_root" diff --name-status "$official_base" HEAD)

mkdir -p "$evidence"
export CUDA_VISIBLE_DEVICES=0

git -C "$adgs_root" remote get-url origin > "$evidence/git_remote.txt"
git -C "$adgs_root" rev-parse HEAD > "$evidence/git_commit.txt"
git -C "$adgs_root" status --short --branch > "$evidence/git_status.txt"
git -C "$adgs_root" diff --name-status "$official_base" HEAD \
    > "$evidence/official_source_diff.tsv"
nvidia-smi > "$evidence/nvidia-smi.before.txt"
conda run --no-capture-output -n "$env_name" python -m pip freeze \
    > "$evidence/environment.txt"
cp "$env_evidence/smoke.json" "$evidence/env001.smoke.json"
sha256sum "$env_evidence/smoke.json" > "$evidence/env001.smoke.sha256"
cp "$env_evidence/evaluator.json" "$evidence/env001.evaluator.json"
sha256sum "$env_evidence/evaluator.json" > "$evidence/env001.evaluator.sha256"

data_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/validate_adgs_waymo_baseline.py"
    --scene "$scene" --output "$evidence/data002.json"
)
smoke_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/smoke_adgs_runtime.py"
    --adgs-root "$adgs_root" --output "$evidence/runtime_smoke.json"
)
train_command=(
    env PYTHONUNBUFFERED=1
    conda run --no-capture-output -n "$env_name"
    python "$adgs_root/train.py"
    -c "$adgs_root/arguments/waymo.py"
    -s "$scene" -m "$run_dir" --data_device cuda:0
)
render_command=(
    env PYTHONUNBUFFERED=1
    conda run --no-capture-output -n "$env_name"
    python "$adgs_root/render.py"
    -c "$adgs_root/arguments/waymo.py"
    -m "$run_dir" --data_device cuda:0 -v
)
result_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/validate_base001_results.py"
    --run-dir "$run_dir" --output "$evidence/metrics.json"
)
{
    printf '%q ' "${data_command[@]}"; printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=0 '; printf '%q ' "${smoke_command[@]}"; printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=0 '; printf '%q ' "${train_command[@]}"; printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=0 '; printf '%q ' "${render_command[@]}"; printf '\n'
    printf '%q ' "${result_command[@]}"; printf '\n'
} > "$evidence/command.sh"

finish_preflight_failure() {
    local stage=$1
    local stage_exitcode=$2
    printf '%s\n' "$stage" > "$evidence/failed_stage.txt"
    printf '%s\n' "$stage_exitcode" > "$evidence/exitcode.txt"
    nvidia-smi > "$evidence/nvidia-smi.after.txt"
    find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
        | sort -z | xargs -0 sha256sum > "$evidence/artifacts.sha256"
    cat "$evidence/$stage.stdout.log"
    cat "$evidence/$stage.stderr.log" >&2
    echo "BASE-001 stopped at $stage; evidence: $evidence" >&2
    exit "$stage_exitcode"
}

set +e
"${data_command[@]}" > "$evidence/data002.stdout.log" \
    2> "$evidence/data002.stderr.log"
data_exitcode=$?
set -e
printf '%s\n' "$data_exitcode" > "$evidence/data002.exitcode.txt"
if [[ "$data_exitcode" -ne 0 ]]; then
    finish_preflight_failure data002 "$data_exitcode"
fi

set +e
"${smoke_command[@]}" > "$evidence/smoke.stdout.log" \
    2> "$evidence/smoke.stderr.log"
smoke_exitcode=$?
set -e
printf '%s\n' "$smoke_exitcode" > "$evidence/smoke.exitcode.txt"
if [[ "$smoke_exitcode" -ne 0 ]]; then
    finish_preflight_failure smoke "$smoke_exitcode"
fi

nvidia-smi --id=0 \
    --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
    --format=csv -l 5 > "$evidence/gpu_monitor.csv" &
monitor_pid=$!
stop_monitor() {
    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap stop_monitor EXIT INT TERM

start_seconds=$(date +%s)
set +e
"${train_command[@]}" > "$evidence/train.stdout.log" \
    2> "$evidence/train.stderr.log"
train_exitcode=$?
train_end_seconds=$(date +%s)
render_exitcode=125
if [[ "$train_exitcode" -eq 0 ]]; then
    "${render_command[@]}" > "$evidence/render.stdout.log" \
        2> "$evidence/render.stderr.log"
    render_exitcode=$?
fi
render_end_seconds=$(date +%s)
set -e
stop_monitor
trap - EXIT INT TERM

printf '%s\n' "$train_exitcode" > "$evidence/train.exitcode.txt"
printf '%s\n' "$render_exitcode" > "$evidence/render.exitcode.txt"
printf '%s\n' "$((train_end_seconds - start_seconds))" \
    > "$evidence/train.wall_time_seconds.txt"
printf '%s\n' "$((render_end_seconds - train_end_seconds))" \
    > "$evidence/render.wall_time_seconds.txt"
nvidia-smi > "$evidence/nvidia-smi.after.txt"

exitcode=$train_exitcode
if [[ "$exitcode" -eq 0 ]]; then
    exitcode=$render_exitcode
fi
if [[ "$exitcode" -eq 0 ]]; then
    set +e
    "${result_command[@]}" > "$evidence/metrics.stdout.log" \
        2> "$evidence/metrics.stderr.log"
    exitcode=$?
    set -e
fi

if [[ -f "$run_dir/results.json" ]]; then
    cp "$run_dir/results.json" "$evidence/results.test.json"
fi
if [[ -f "$run_dir/results-train.json" ]]; then
    cp "$run_dir/results-train.json" "$evidence/results.train.json"
fi
printf '%s\n' "$exitcode" > "$evidence/exitcode.txt"
if [[ -d "$run_dir" ]]; then
    find "$run_dir" -type f -print0 | sort -z | xargs -0 sha256sum \
        > "$evidence/output_artifacts.sha256"
fi
find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$evidence/artifacts.sha256"

cat "$evidence/train.stdout.log"
cat "$evidence/train.stderr.log" >&2
if [[ -f "$evidence/render.stdout.log" ]]; then
    cat "$evidence/render.stdout.log"
    cat "$evidence/render.stderr.log" >&2
fi
if [[ -f "$evidence/metrics.stdout.log" ]]; then
    cat "$evidence/metrics.stdout.log"
    cat "$evidence/metrics.stderr.log" >&2
fi
echo "BASE-001 evidence: $evidence"
exit "$exitcode"
