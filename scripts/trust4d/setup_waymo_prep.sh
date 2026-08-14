#!/usr/bin/env bash
set -euo pipefail

env_name=trust4d-waymo-prep
command -v conda >/dev/null

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
conda run --no-capture-output -n "$env_name" \
    python -m pip install waymo-open-dataset-tf-2-11-0==1.6.1 --no-dependencies

env CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n "$env_name" python -c \
    'import numpy, tensorflow, torch; from waymo_open_dataset import dataset_pb2; from waymo_open_dataset.utils import range_image_utils, transform_utils; print({"numpy": numpy.__version__, "tensorflow": tensorflow.__version__, "torch": torch.__version__, "waymo_frame": dataset_pb2.Frame.__name__})'

echo "Waymo preprocessing environment ready: $env_name"
