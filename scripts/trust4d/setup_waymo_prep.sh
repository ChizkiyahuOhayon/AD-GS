#!/usr/bin/env bash
set -euo pipefail

env_name=trust4d-waymo-prep
waymo_wheel=waymo_open_dataset_tf_2_11_0-1.6.1-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
waymo_wheel_url=https://files.pythonhosted.org/packages/27/2e/1aa476186c4e0ab0ad36d2d1958c03521a17d6ed53cc8293b87597200a02/waymo_open_dataset_tf_2_11_0-1.6.1-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
waymo_wheel_sha256=2d0d4a4bbc59fe2fc8e19e9958f560383e81454e1c0662ad2e642d0797ce7f6e
command -v conda >/dev/null
command -v curl >/dev/null

if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "conda environment already exists; refusing to mutate it: $env_name" >&2
    exit 2
fi

conda create -n "$env_name" -y python=3.10 pip
conda run --no-capture-output -n "$env_name" \
    python -m pip install torch==1.13.1 \
    --index-url https://download.pytorch.org/whl/cpu
conda run --no-capture-output -n "$env_name" \
    python -m pip install \
    numpy==1.23.5 tensorflow==2.11.0 plyfile==0.9 \
    Pillow==9.5.0 tqdm==4.66.2

download_dir=$(mktemp -d)
trap 'rm -rf "$download_dir"' EXIT
curl -fL --retry 3 --output "$download_dir/$waymo_wheel" "$waymo_wheel_url"
echo "$waymo_wheel_sha256  $download_dir/$waymo_wheel" | sha256sum --check --strict
test "$(stat -c %s "$download_dir/$waymo_wheel")" -eq 4041796
conda run --no-capture-output -n "$env_name" \
    python -m pip install "$download_dir/$waymo_wheel" --no-dependencies

env CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n "$env_name" python -c \
    'import numpy, tensorflow, torch; from waymo_open_dataset import dataset_pb2; from waymo_open_dataset.utils import range_image_utils, transform_utils; print({"numpy": numpy.__version__, "tensorflow": tensorflow.__version__, "torch": torch.__version__, "waymo_frame": dataset_pb2.Frame.__name__})'

echo "Waymo preprocessing environment ready: $env_name"
