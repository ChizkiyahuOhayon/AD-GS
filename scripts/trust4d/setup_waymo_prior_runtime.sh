#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <dpt|sam|flow|colmap> <new-dependencies-root> <new-evidence-dir>" >&2
    exit 2
fi

stage=$1
deps_root=$(realpath -m "$2")
evidence=$(realpath -m "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adgs_root=$(git -C "$script_dir" rev-parse --show-toplevel)
stage_root="$deps_root/$stage"

case "$stage" in
    dpt) env_name=trust4d-dpt ;;
    sam) env_name=trust4d-sam ;;
    flow) env_name=trust4d-flow ;;
    colmap) env_name=trust4d-colmap ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac

command -v conda >/dev/null
command -v git >/dev/null
command -v curl >/dev/null
if [[ "$stage" != colmap ]]; then
    command -v nvidia-smi >/dev/null
fi
if [[ "$stage" == sam ]]; then
    command -v nvcc >/dev/null
fi
[[ -z "$(git -C "$adgs_root" status --porcelain)" ]]
if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "Conda environment already exists; refusing to mutate it: $env_name" >&2
    exit 2
fi
if [[ -e "$stage_root" || -e "$evidence" ]]; then
    echo "stage dependency root and evidence directory must both be new" >&2
    exit 2
fi
for path in "$stage_root" "$evidence"; do
    case "$path/" in "$adgs_root/"*)
        echo "dependencies and evidence must be outside the Git repository" >&2
        exit 2
    esac
done

declare -A source_hashes=(
    [dpt]=f02545c46410b3fe8bc6b4a527ddf99a9383901eabe272c2a4e0e7450e605a1d
    [sam]=8429c19c6b50591103342d1205391e805866f1745a2d7cb4109e364f2911198f
    [flow]=5d4725c7f25d077aaef31fbade55f0e9ba58b26282321b87f0bde14b5969c083
    [colmap]=070437ba06cbeaf784ddc0508ebbd016936047c71519f588175ca9ecf7cd1cca
)
declare -A source_files=(
    [dpt]=scripts/run-dpt.py
    [sam]=scripts/semantic.py
    [flow]=scripts/flow.py
    [colmap]=scripts/colmap.py
)
actual_source_hash=$(sha256sum "$adgs_root/${source_files[$stage]}" | awk '{print $1}')
[[ "$actual_source_hash" == "${source_hashes[$stage]}" ]]

mkdir -p "$deps_root" "$evidence"
git -C "$adgs_root" remote get-url origin > "$evidence/git_remote.txt"
git -C "$adgs_root" rev-parse HEAD > "$evidence/git_commit.txt"
git -C "$adgs_root" status --short --branch > "$evidence/git_status.txt"
if [[ "$stage" != colmap ]]; then
    nvidia-smi > "$evidence/nvidia-smi.txt"
fi
if [[ "$stage" == sam ]]; then
    nvcc --version > "$evidence/nvcc.txt"
fi
printf '%s  %s\n' "$actual_source_hash" "${source_files[$stage]}" \
    > "$evidence/generator_source.sha256"
printf '%q ' bash "$script_dir/setup_waymo_prior_runtime.sh" \
    "$stage" "$deps_root" "$evidence" > "$evidence/command.sh"
printf '\n' >> "$evidence/command.sh"

download_checked() {
    local url=$1
    local target=$2
    local expected_size=$3
    local expected_hash=$4
    mkdir -p "$(dirname "$target")"
    curl -fL --retry 3 --output "$target.part" "$url"
    test "$(stat -c %s "$target.part")" -eq "$expected_size"
    echo "$expected_hash  $target.part" | sha256sum --check --strict
    mv "$target.part" "$target"
    sha256sum "$target" >> "$evidence/checkpoints.sha256"
    printf '%s\t%s\n' "$(stat -c %s "$target")" "$target" \
        >> "$evidence/checkpoint_sizes.tsv"
}

install_torch() {
    conda run --no-capture-output -n "$env_name" python -m pip install \
        torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
}

setup_dpt() {
    local commit=e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1
    conda create -n "$env_name" -y python=3.11 pip=24.2 setuptools=75.1.0 wheel=0.44.0
    install_torch
    conda run --no-capture-output -n "$env_name" python -m pip install \
        numpy==1.26.4 opencv-python==4.10.0.84 matplotlib==3.9.2
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$stage_root"
    git -C "$stage_root" checkout "$commit"
    cp "$adgs_root/scripts/run-dpt.py" "$stage_root/run-dpt.py"
    download_checked \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth" \
        "$stage_root/checkpoints/depth_anything_v2_vitl.pth" \
        1341395338 a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345
    env CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n "$env_name" \
        python "$script_dir/smoke_waymo_prior_runtime.py" --stage dpt \
        --root "$stage_root" \
        --checkpoint "$stage_root/checkpoints/depth_anything_v2_vitl.pth" \
        --output "$evidence/smoke.json"
}

setup_sam() {
    local commit=dd4c5141b75e4838dd486c64f773c43b4db3a07b
    local cuda_home
    cuda_home=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
    conda create -n "$env_name" -y python=3.10 pip=24.2 setuptools=75.1.0 wheel=0.44.0
    install_torch
    conda run --no-capture-output -n "$env_name" python -m pip install \
        numpy==1.26.4 pillow==10.4.0 tqdm==4.66.5 hydra-core==1.3.2 \
        iopath==0.1.10 opencv-python==4.10.0.84 matplotlib==3.9.2 \
        transformers==4.46.3 supervision==0.22.0 pycocotools==2.0.8 \
        huggingface-hub==0.26.2 safetensors==0.4.5 tokenizers==0.20.3 \
        ninja==1.11.1.1
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "$stage_root"
    git -C "$stage_root" checkout "$commit"
    cp "$adgs_root/scripts/semantic.py" "$stage_root/semantic.py"
    env CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$cuda_home" FORCE_CUDA=1 \
        TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8 SAM2_BUILD_CUDA=1 \
        SAM2_BUILD_ALLOW_ERRORS=0 \
        conda run --no-capture-output -n "$env_name" python -m pip install \
        --no-build-isolation -e "$stage_root"
    download_checked \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        "$stage_root/checkpoints/sam2.1_hiera_large.pt" \
        898083611 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
    conda run --no-capture-output -n "$env_name" \
        python "$script_dir/cache_grounding_dino.py" \
        --hf-home "$stage_root/hf-home" --output "$evidence/grounding_dino.json"
    env CUDA_VISIBLE_DEVICES=0 HF_HOME="$stage_root/hf-home" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        conda run --no-capture-output -n "$env_name" \
        python "$script_dir/smoke_waymo_prior_runtime.py" --stage sam \
        --root "$stage_root" \
        --checkpoint "$stage_root/checkpoints/sam2.1_hiera_large.pt" \
        --output "$evidence/smoke.json"
}

setup_flow() {
    local commit=82e02e8029753ad4ef13cf06be7f4fc5facdda4d
    local hub_repo="$stage_root/torch-home/hub/facebookresearch_co-tracker_main"
    conda create -n "$env_name" -y python=3.10 pip=24.2 setuptools=75.1.0 wheel=0.44.0
    install_torch
    conda run --no-capture-output -n "$env_name" python -m pip install \
        numpy==1.26.4 pillow==10.4.0 imageio==2.35.1 imageio-ffmpeg==0.5.1 \
        matplotlib==3.9.2 flow-vis==0.1 tqdm==4.66.5 \
        einops==0.8.0 timm==1.0.9 plyfile==1.1
    mkdir -p "$(dirname "$hub_repo")"
    git clone https://github.com/facebookresearch/co-tracker.git "$hub_repo"
    git -C "$hub_repo" checkout "$commit"
    download_checked \
        "https://huggingface.co/facebook/cotracker3/resolve/bf55ea50d4390e1820a267f131cd6587240fb2c5/scaled_offline.pth" \
        "$stage_root/torch-home/hub/checkpoints/scaled_offline.pth" \
        101890938 2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834
    env CUDA_VISIBLE_DEVICES=0 TORCH_HOME="$stage_root/torch-home" \
        conda run --no-capture-output -n "$env_name" \
        python "$script_dir/smoke_waymo_prior_runtime.py" --stage flow \
        --root "$stage_root" --output "$evidence/smoke.json"
}

setup_colmap() {
    mkdir -p "$stage_root"
    conda create -n "$env_name" -y -c conda-forge python=3.10 pip=24.2 colmap=3.7
    conda run --no-capture-output -n "$env_name" python -m pip install \
        numpy==1.23.5 scipy==1.10.1 pillow==9.5.0 plyfile==0.9 tqdm==4.66.2
    env CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n "$env_name" \
        python "$script_dir/smoke_waymo_prior_runtime.py" --stage colmap \
        --root "$stage_root" --output "$evidence/smoke.json"
}

start_seconds=$(date +%s)
set +e
"setup_$stage" > "$evidence/stdout.log" 2> "$evidence/stderr.log"
exitcode=$?
set -e
end_seconds=$(date +%s)

if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    conda run -n "$env_name" python -m pip freeze > "$evidence/environment.txt" 2>&1
fi
if [[ -d "$stage_root/.git" ]]; then
    {
        git -C "$stage_root" rev-parse HEAD
        git -C "$stage_root" status --short --branch
    } > "$evidence/dependency_git_status.txt"
elif [[ "$stage" == flow && -d "$stage_root/torch-home/hub/facebookresearch_co-tracker_main/.git" ]]; then
    {
        git -C "$stage_root/torch-home/hub/facebookresearch_co-tracker_main" rev-parse HEAD
        git -C "$stage_root/torch-home/hub/facebookresearch_co-tracker_main" status --short --branch
    } > "$evidence/dependency_git_status.txt"
fi
printf '%s\n' "$exitcode" > "$evidence/exitcode.txt"
printf '%s\n' "$((end_seconds - start_seconds))" > "$evidence/wall_time_seconds.txt"
find "$evidence" -maxdepth 1 -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$evidence/artifacts.sha256"

cat "$evidence/stdout.log"
cat "$evidence/stderr.log" >&2
echo "ENV-002/$stage evidence: $evidence"
exit "$exitcode"
