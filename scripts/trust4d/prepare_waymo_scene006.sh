#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <raw-scene006-tfrecord> <new-processed-scene-dir> <new-evidence-dir>" >&2
    exit 2
fi

raw_tfrecord=$(realpath "$1")
scene_output=$(realpath -m "$2")
evidence=$(realpath -m "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adgs_root=$(git -C "$script_dir" rev-parse --show-toplevel)
env_name=trust4d-waymo-prep
expected_filename=individual_files_validation_segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord
expected_preprocessor_sha=f87905fb7c867d572679a6b7ea92dbe4b085d5a4a695f70ee1776c2058188bd6

[[ "$(basename "$raw_tfrecord")" == "$expected_filename" ]]
[[ -f "$raw_tfrecord" ]]
[[ "$(sha256sum "$adgs_root/scripts/waymo/waymo.py" | awk '{print $1}')" == "$expected_preprocessor_sha" ]]
[[ -z "$(git -C "$adgs_root" status --porcelain)" ]]

case "$scene_output/" in "$adgs_root/"*)
    echo "processed data must be outside the Git repository" >&2
    exit 2
esac
case "$evidence/" in "$adgs_root/"*)
    echo "evidence must be outside the Git repository" >&2
    exit 2
esac
if [[ -e "$scene_output" || -e "$evidence" ]]; then
    echo "processed scene and evidence paths must both be new" >&2
    exit 2
fi

mkdir -p "$evidence"
git -C "$adgs_root" remote get-url origin > "$evidence/git_remote.txt"
git -C "$adgs_root" rev-parse HEAD > "$evidence/git_commit.txt"
git -C "$adgs_root" status --short --branch > "$evidence/git_status.txt"
sha256sum "$raw_tfrecord" > "$evidence/raw_tfrecord.sha256"
env CUDA_VISIBLE_DEVICES= conda run -n "$env_name" python -m pip freeze \
    > "$evidence/environment.txt"

prepare_command=(
    env CUDA_VISIBLE_DEVICES=
    conda run --no-capture-output -n "$env_name"
    python "$adgs_root/scripts/waymo/waymo.py"
    "$raw_tfrecord" "$scene_output"
    --first_frame 0 --last_frame 85 --use_color
)
validate_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/validate_waymo_scene006.py"
    --scene "$scene_output"
    --output "$evidence/validation.json"
)
select_command=(
    conda run --no-capture-output -n "$env_name"
    python "$script_dir/select_waymo_training_frames.py"
    --scene "$scene_output" --count 4
    --output "$evidence/selection.json"
)
{
    printf '%q ' "${prepare_command[@]}"; printf '\n'
    printf '%q ' "${validate_command[@]}"; printf '\n'
    printf '%q ' "${select_command[@]}"; printf '\n'
} > "$evidence/command.sh"

start_seconds=$(date +%s)
set +e
"${prepare_command[@]}" > "$evidence/stdout.log" 2> "$evidence/stderr.log"
exitcode=$?
if [[ "$exitcode" -eq 0 ]]; then
    "${validate_command[@]}" > "$evidence/validation.stdout.log" \
        2> "$evidence/validation.stderr.log"
    exitcode=$?
fi
if [[ "$exitcode" -eq 0 ]]; then
    "${select_command[@]}" > "$evidence/selection.stdout.log" \
        2> "$evidence/selection.stderr.log"
    exitcode=$?
fi
set -e
end_seconds=$(date +%s)

printf '%s\n' "$exitcode" > "$evidence/exitcode.txt"
printf '%s\n' "$((end_seconds - start_seconds))" > "$evidence/wall_time_seconds.txt"
find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$evidence/artifacts.sha256"

cat "$evidence/stdout.log"
cat "$evidence/stderr.log" >&2
echo "DATA-001 evidence: $evidence"
exit "$exitcode"
