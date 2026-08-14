#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <DGGT-root>" >&2
    exit 2
fi

dggt_root=$(realpath "$1")
env_name=trust4d-dggt-exp001
expected_commit=a3276d2bbe4cbb03bcc117830b1836110a27adeb
expected_requirements_sha=33428d6d7da037a7a344ea230819901493d2dbf9ca3c2b264fdeab689722854f
wheel_name='gsplat-1.5.3+pt24cu121-cp310-cp310-linux_x86_64.whl'
wheel_url="https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/${wheel_name}"
wheel_sha=0493bab68ed5fc71f4ce8bfc2be03b584d8a41a06a6d9362e09a795340f8c488

command -v conda >/dev/null
command -v curl >/dev/null
[[ "$(git -C "$dggt_root" rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git -C "$dggt_root" status --porcelain)" ]]
[[ "$(sha256sum "$dggt_root/requirements.txt" | awk '{print $1}')" == "$expected_requirements_sha" ]]

if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "conda environment already exists; refusing to mutate it: $env_name" >&2
    exit 2
fi

wheel_dir=$(mktemp -d)
trap 'rm -rf "$wheel_dir"' EXIT
curl --fail --location --retry 5 --output "$wheel_dir/$wheel_name" "$wheel_url"
[[ "$(sha256sum "$wheel_dir/$wheel_name" | awk '{print $1}')" == "$wheel_sha" ]]

conda create -n "$env_name" -y python=3.10 pip
conda run --no-capture-output -n "$env_name" \
    python -m pip install \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -n "$env_name" \
    python -m pip install 'numpy<2' ninja jaxtyping rich
conda run --no-capture-output -n "$env_name" \
    python -m pip install "$wheel_dir/$wheel_name"
conda run --no-capture-output -n "$env_name" \
    python -m pip install -r "$dggt_root/requirements.txt" scikit-learn
conda run --no-capture-output -n "$env_name" python -m pip check

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$dggt_root" \
    conda run --no-capture-output -n "$env_name" python -c \
    'import importlib.metadata as m; import torch; import dggt.models.vggt; print({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "torchvision": m.version("torchvision"), "gsplat": m.version("gsplat"), "scikit-learn": m.version("scikit-learn"), "gpu": torch.cuda.get_device_name(0)})'

echo "DGGT EXP-001 environment ready: $env_name"
