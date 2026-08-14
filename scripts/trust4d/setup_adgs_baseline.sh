#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <new-evidence-dir>" >&2
    exit 2
fi

env_name=trust4d-adgs-baseline
official_base=9a208512e49c8ddbaa20387921d9648adcd21cb4
pytorch3d_commit=297020a4b1d7492190cb4a909cafbd2c81a12cb5
simple_knn_tree=d5b756edadeef66644510a23633c23803d6b61db
rasterizer_tree=b78a10882e6a99927b74303a83cb2c107666cdd3
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adgs_root=$(git -C "$script_dir" rev-parse --show-toplevel)
evidence=$(realpath -m "$1")

command -v conda >/dev/null
command -v nvcc >/dev/null
command -v nvidia-smi >/dev/null
command -v python3 >/dev/null
git -C "$adgs_root" merge-base --is-ancestor "$official_base" HEAD
[[ -z "$(git -C "$adgs_root" status --porcelain)" ]]
[[ "$(git -C "$adgs_root" rev-parse HEAD:submodules/simple-knn)" == "$simple_knn_tree" ]]
[[ "$(git -C "$adgs_root" rev-parse HEAD:submodules/depth-diff-gaussian-rasterization)" == "$rasterizer_tree" ]]
nvcc --version | grep -Fq 'release 11.8'

if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "Conda environment already exists; refusing to mutate it: $env_name" >&2
    exit 2
fi
case "$evidence/" in "$adgs_root/"*)
    echo "evidence must be outside the Git repository" >&2
    exit 2
esac
if [[ -e "$evidence" ]]; then
    echo "evidence path must be new: $evidence" >&2
    exit 2
fi

source_audit=$(python3 "$script_dir/audit_official_source.py" \
    --repository "$adgs_root")
mkdir -p "$evidence"
printf '%s\n' "$source_audit" > "$evidence/official_source_audit.json"
build_dir=$(mktemp -d)
cuda_home=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
start_seconds=$(date +%s)

git -C "$adgs_root" remote get-url origin > "$evidence/git_remote.txt"
git -C "$adgs_root" rev-parse HEAD > "$evidence/git_commit.txt"
git -C "$adgs_root" status --short --branch > "$evidence/git_status.txt"
nvidia-smi > "$evidence/nvidia-smi.txt"
nvcc --version > "$evidence/nvcc.txt"
free -h > "$evidence/memory.txt"
printf '%q ' bash "$script_dir/setup_adgs_baseline.sh" "$evidence" \
    > "$evidence/command.sh"
printf '\n' >> "$evidence/command.sh"

run_setup() {
    conda create -n "$env_name" -y python=3.8 pip=23.1 \
        setuptools=65.6.3 wheel=0.38.4
    conda run --no-capture-output -n "$env_name" \
        python -m pip install \
        torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
        --index-url https://download.pytorch.org/whl/cu117
    conda run --no-capture-output -n "$env_name" \
        python -m pip install \
        numpy==1.21.6 scipy==1.7.3 Pillow==9.5.0 opencv-python==4.9.0.80 \
        open3d==0.17.0 plyfile==0.9 roma==1.5.1 flow-vis==0.1 \
        imageio==2.31.2 imageio-ffmpeg==0.4.9 matplotlib==3.5.2 \
        tensorboard==2.11.2 tqdm==4.66.2 ninja==1.11.1.1 \
        fvcore==0.1.5.post20221221 iopath==0.1.10 portalocker==2.7.0 \
        yacs==0.1.8

    env CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$cuda_home" FORCE_CUDA=1 \
        TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 \
        conda run --no-capture-output -n "$env_name" python -m pip install \
        --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@$pytorch3d_commit"

    cp -a "$adgs_root/submodules/simple-knn" "$build_dir/simple-knn"
    cp -a "$adgs_root/submodules/depth-diff-gaussian-rasterization" \
        "$build_dir/depth-diff-gaussian-rasterization"
    env CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$cuda_home" \
        TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 \
        conda run --no-capture-output -n "$env_name" python -m pip install \
        --no-build-isolation "$build_dir/simple-knn"
    env CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$cuda_home" \
        TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 \
        conda run --no-capture-output -n "$env_name" python -m pip install \
        --no-build-isolation "$build_dir/depth-diff-gaussian-rasterization"

    conda run --no-capture-output -n "$env_name" python -m pip check
    conda run --no-capture-output -n "$env_name" python -m pip freeze \
        > "$evidence/environment.txt"
    env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n "$env_name" \
        python "$script_dir/cache_adgs_evaluator.py" \
        --adgs-root "$adgs_root" --output "$evidence/evaluator.json"
    env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n "$env_name" \
        python "$script_dir/smoke_adgs_runtime.py" \
        --adgs-root "$adgs_root" --output "$evidence/smoke.json"
}

set +e
run_setup > "$evidence/stdout.log" 2> "$evidence/stderr.log"
exitcode=$?
set -e
end_seconds=$(date +%s)
rm -rf "$build_dir"

printf '%s\n' "$exitcode" > "$evidence/exitcode.txt"
printf '%s\n' "$((end_seconds - start_seconds))" > "$evidence/wall_time_seconds.txt"
find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$evidence/artifacts.sha256"

cat "$evidence/stdout.log"
cat "$evidence/stderr.log" >&2
echo "ENV-001 evidence: $evidence"
exit "$exitcode"
