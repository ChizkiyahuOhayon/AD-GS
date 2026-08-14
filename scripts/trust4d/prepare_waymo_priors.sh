#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <scene006-path> <dependencies-root> <ENV-002-evidence-root> <new-DATA-002-evidence-dir>" >&2
    exit 2
fi

scene=$(realpath "$1")
deps_root=$(realpath "$2")
env_evidence_root=$(realpath "$3")
evidence=$(realpath -m "$4")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adgs_root=$(git -C "$script_dir" rev-parse --show-toplevel)

command -v conda >/dev/null
command -v nvidia-smi >/dev/null
if [[ -n "$(git -C "$adgs_root" status --porcelain)" ]]; then
    echo "AD-GS working tree is not clean" >&2
    git -C "$adgs_root" status --short >&2
    exit 2
fi
if [[ -e "$evidence" ]]; then
    echo "DATA-002 evidence directory must be new: $evidence" >&2
    exit 2
fi
case "$evidence/" in "$adgs_root/"*)
    echo "DATA-002 evidence must be outside the Git repository" >&2
    exit 2
esac

declare -A env_names=(
    [dpt]=trust4d-dpt
    [sam]=trust4d-sam
    [flow]=trust4d-flow
    [colmap]=trust4d-colmap
)
for stage in dpt sam flow colmap; do
    stage_evidence="$env_evidence_root/$stage"
    if [[ ! -f "$stage_evidence/exitcode.txt" \
        || ! -f "$stage_evidence/smoke.json" ]]; then
        echo "ENV-002/$stage evidence is missing or failed: $stage_evidence" >&2
        exit 2
    fi
    if [[ "$(<"$stage_evidence/exitcode.txt")" != 0 ]]; then
        echo "ENV-002/$stage records a failed setup: $stage_evidence" >&2
        exit 2
    fi
    if ! conda env list | awk '{print $1}' | grep -Fxq "${env_names[$stage]}"; then
        echo "required Conda environment is missing: ${env_names[$stage]}" >&2
        exit 2
    fi
done
conda run -n trust4d-flow python -c \
    'import json,sys; assert all(json.load(open(path))["passed"] is True for path in sys.argv[1:])' \
    "$env_evidence_root/dpt/smoke.json" \
    "$env_evidence_root/sam/smoke.json" \
    "$env_evidence_root/flow/smoke.json" \
    "$env_evidence_root/colmap/smoke.json"

dpt_root="$deps_root/dpt"
sam_root="$deps_root/sam"
flow_root="$deps_root/flow"
[[ "$(git -C "$dpt_root" rev-parse HEAD)" == e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1 ]]
[[ "$(git -C "$sam_root" rev-parse HEAD)" == dd4c5141b75e4838dd486c64f773c43b4db3a07b ]]
[[ "$(git -C "$flow_root/torch-home/hub/facebookresearch_co-tracker_main" rev-parse HEAD)" == 82e02e8029753ad4ef13cf06be7f4fc5facdda4d ]]

declare -A source_hashes=(
    [run-dpt.py]=f02545c46410b3fe8bc6b4a527ddf99a9383901eabe272c2a4e0e7450e605a1d
    [semantic.py]=8429c19c6b50591103342d1205391e805866f1745a2d7cb4109e364f2911198f
    [flow.py]=5d4725c7f25d077aaef31fbade55f0e9ba58b26282321b87f0bde14b5969c083
    [segment_pcd.py]=acf98743fff4572cfbb0ea14a624398544b4b1eadaa3e2d2e9ea5c68802cdb57
    [colmap.py]=070437ba06cbeaf784ddc0508ebbd016936047c71519f588175ca9ecf7cd1cca
)
for filename in "${!source_hashes[@]}"; do
    actual=$(sha256sum "$adgs_root/scripts/$filename" | awk '{print $1}')
    [[ "$actual" == "${source_hashes[$filename]}" ]]
done
[[ "$(sha256sum "$dpt_root/run-dpt.py" | awk '{print $1}')" == "${source_hashes[run-dpt.py]}" ]]
[[ "$(sha256sum "$sam_root/semantic.py" | awk '{print $1}')" == "${source_hashes[semantic.py]}" ]]

for output in depth semantic sky flow colmap colmap.ply semantic.mp4 sky.mp4 flow.mp4; do
    if [[ -e "$scene/$output" ]]; then
        echo "scene contains a pre-existing DATA-002 output: $scene/$output" >&2
        exit 2
    fi
done
if [[ -e "$sam_root/outputs" ]]; then
    echo "Grounded-SAM working output already exists: $sam_root/outputs" >&2
    exit 2
fi

mkdir -p "$evidence"
export CUDA_VISIBLE_DEVICES=0
git -C "$adgs_root" remote get-url origin > "$evidence/git_remote.txt"
git -C "$adgs_root" rev-parse HEAD > "$evidence/git_commit.txt"
git -C "$adgs_root" status --short --branch > "$evidence/git_status.txt"
nvidia-smi > "$evidence/nvidia-smi.before.txt"
for stage in dpt sam flow colmap; do
    cp "$env_evidence_root/$stage/smoke.json" "$evidence/env002.$stage.smoke.json"
    sha256sum "$env_evidence_root/$stage/smoke.json" \
        >> "$evidence/env002.smoke.sha256"
done

data001_command=(
    conda run --no-capture-output -n trust4d-waymo-prep
    python "$script_dir/validate_waymo_scene006.py"
    --scene "$scene" --output "$evidence/data001.json"
)
depth_command=(
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
    conda run --no-capture-output -n trust4d-dpt
    python "$dpt_root/run-dpt.py"
    --img-path "$scene/image" --outdir "$scene/depth"
)
sky_command=(
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
    HF_HOME="$sam_root/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    conda run --no-capture-output -n trust4d-sam
    python "$sam_root/semantic.py" "$scene"
    --device cuda:0 --text sky. --name sky --step 1
)
semantic_command=(
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
    HF_HOME="$sam_root/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    conda run --no-capture-output -n trust4d-sam
    python "$sam_root/semantic.py" "$scene"
    --device cuda:0 --text car.bus.truck.van.human. --name semantic --step 1
)
segment_command=(
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
    conda run --no-capture-output -n trust4d-flow
    python "$adgs_root/scripts/segment_pcd.py" "$scene"
)
flow_command=(
    env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
    TORCH_HOME="$flow_root/torch-home"
    conda run --no-capture-output -n trust4d-flow
    python "$adgs_root/scripts/flow.py" "$scene"
    --device cuda:0 --downsample 1 --step 4
)
colmap_command=(
    env CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
    conda run --no-capture-output -n trust4d-colmap
    python "$adgs_root/scripts/colmap.py" "$scene" --cam 1
)
validate_command=(
    conda run --no-capture-output -n trust4d-flow
    python "$script_dir/validate_adgs_waymo_baseline.py"
    --scene "$scene" --output "$evidence/data002.json"
)

record_command() {
    local working_directory=$1
    shift
    printf 'cd %q && ' "$working_directory"
    printf '%q ' "$@"
    printf '\n'
}
{
    record_command "$adgs_root" "${data001_command[@]}"
    record_command "$dpt_root" "${depth_command[@]}"
    record_command "$sam_root" "${sky_command[@]}"
    record_command "$sam_root" "${semantic_command[@]}"
    record_command "$adgs_root" "${segment_command[@]}"
    record_command "$adgs_root" "${flow_command[@]}"
    record_command "$adgs_root" "${colmap_command[@]}"
    record_command "$adgs_root" "${validate_command[@]}"
} > "$evidence/command.sh"

finalize_failure() {
    local stage=$1
    local stage_exitcode=$2
    if [[ "$stage" == segment && -f "$evidence/points3d.unsegmented.ply" ]]; then
        cp "$evidence/points3d.unsegmented.ply" "$scene/points3d.ply"
        sha256sum "$scene/points3d.ply" > "$evidence/points3d.restored.sha256"
    fi
    printf '%s\n' "$stage" > "$evidence/failed_stage.txt"
    printf '%s\n' "$stage_exitcode" > "$evidence/exitcode.txt"
    nvidia-smi > "$evidence/nvidia-smi.after.txt"
    find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
        | sort -z | xargs -0 sha256sum > "$evidence/artifacts.sha256"
    cat "$evidence/$stage.stdout.log"
    cat "$evidence/$stage.stderr.log" >&2
    echo "DATA-002 stopped at $stage; evidence: $evidence" >&2
    exit "$stage_exitcode"
}

segment_transaction_open=0
restore_segment_input() {
    if [[ "$segment_transaction_open" == 1 \
        && -f "$evidence/points3d.unsegmented.ply" ]]; then
        cp "$evidence/points3d.unsegmented.ply" "$scene/points3d.ply"
        sha256sum "$scene/points3d.ply" > "$evidence/points3d.restored.sha256"
    fi
}
trap restore_segment_input EXIT

run_stage() {
    local stage=$1
    local working_directory=$2
    shift 2
    local start_seconds end_seconds stage_exitcode
    start_seconds=$(date +%s)
    set +e
    (cd "$working_directory" && "$@") > "$evidence/$stage.stdout.log" \
        2> "$evidence/$stage.stderr.log"
    stage_exitcode=$?
    set -e
    end_seconds=$(date +%s)
    printf '%s\n' "$stage_exitcode" > "$evidence/$stage.exitcode.txt"
    printf '%s\n' "$((end_seconds - start_seconds))" \
        > "$evidence/$stage.wall_time_seconds.txt"
    if [[ "$stage_exitcode" -ne 0 ]]; then
        finalize_failure "$stage" "$stage_exitcode"
    fi
}

run_stage data001 "$adgs_root" "${data001_command[@]}"
run_stage depth "$dpt_root" "${depth_command[@]}"
run_stage sky "$sam_root" "${sky_command[@]}"
run_stage semantic "$sam_root" "${semantic_command[@]}"

cp "$scene/points3d.ply" "$evidence/points3d.unsegmented.ply"
sha256sum "$evidence/points3d.unsegmented.ply" \
    > "$evidence/points3d.unsegmented.sha256"
segment_transaction_open=1
run_stage segment "$adgs_root" "${segment_command[@]}"
segment_transaction_open=0
sha256sum "$scene/points3d.ply" > "$evidence/points3d.segmented.sha256"

run_stage flow "$adgs_root" "${flow_command[@]}"
run_stage colmap "$adgs_root" "${colmap_command[@]}"
run_stage validate "$adgs_root" "${validate_command[@]}"

printf '0\n' > "$evidence/exitcode.txt"
nvidia-smi > "$evidence/nvidia-smi.after.txt"
for video in sky.mp4 semantic.mp4 flow.mp4; do
    if [[ -f "$scene/$video" ]]; then
        sha256sum "$scene/$video" >> "$evidence/visualizations.sha256"
    fi
done
find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$evidence/artifacts.sha256"

cat "$evidence/validate.stdout.log"
cat "$evidence/validate.stderr.log" >&2
echo "DATA-002 evidence: $evidence"
