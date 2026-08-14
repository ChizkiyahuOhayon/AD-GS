#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <scene006-path> <DGGT-root> <checkpoint> <new-output-dir>" >&2
    exit 2
fi

scene_path=$1
dggt_root=$2
checkpoint=$3
output=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adgs_root=$(git -C "$script_dir" rev-parse --show-toplevel)
env_name=trust4d-dggt-exp001

command -v conda >/dev/null
if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "required Conda environment does not exist: $env_name" >&2
    exit 2
fi

if [[ -e "$output" ]]; then
    echo "output already exists; choose a new directory: $output" >&2
    exit 2
fi
if [[ -n "$(git -C "$adgs_root" status --porcelain)" ]]; then
    echo "AD-GS working tree is not clean" >&2
    git -C "$adgs_root" status --short >&2
    exit 2
fi

mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=0

git -C "$adgs_root" remote get-url origin > "$output/git_remote.txt"
git -C "$adgs_root" rev-parse HEAD > "$output/git_commit.txt"
{
    echo "===== AD-GS ====="
    git -C "$adgs_root" status --short --branch
    echo "===== DGGT ====="
    git -C "$dggt_root" status --short --branch
    git -C "$dggt_root" rev-parse HEAD
} > "$output/git_status.txt"

nvidia-smi > "$output/nvidia-smi.txt"
{
    date --iso-8601=seconds
    hostname
    conda run --no-capture-output -n "$env_name" python --version
    conda run --no-capture-output -n "$env_name" python -m pip freeze
} > "$output/environment.txt" 2>&1

selection_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/select_waymo_training_frames.py"
    --scene "$scene_path"
    --count 4
    --output "$output/selection.json"
)
probe_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/probe_dggt.py"
    --dggt-root "$dggt_root"
    --checkpoint "$checkpoint"
    --selection "$output/selection.json"
    --output "$output/metrics.json"
)
{
    printf '%q ' "${selection_command[@]}"
    printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=0 '
    printf '%q ' "${probe_command[@]}"
    printf '\n'
} > "$output/command.sh"

"${selection_command[@]}" > "$output/selection.stdout.log" 2> "$output/selection.stderr.log"

start_seconds=$(date +%s)
set +e
"${probe_command[@]}" > "$output/stdout.log" 2> "$output/stderr.log"
exitcode=$?
set -e
end_seconds=$(date +%s)

printf '%s\n' "$exitcode" > "$output/exitcode.txt"
printf '%s\n' "$((end_seconds - start_seconds))" > "$output/wall_time_seconds.txt"
if [[ -f "$output/metrics.json" ]]; then
    conda run --no-capture-output -n "$env_name" python -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["peak_memory_allocated_mib"])' \
        "$output/metrics.json" > "$output/peak_gpu_memory_mib.txt"
fi

find "$output" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output/artifacts.sha256"

cat "$output/stdout.log"
cat "$output/stderr.log" >&2
echo "EXP-001 evidence: $output"
exit "$exitcode"
