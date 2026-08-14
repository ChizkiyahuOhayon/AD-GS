#!/usr/bin/env bash
set -euo pipefail

nas_root=${1:-"$HOME/dy/nas"}
output=${2:-"$HOME/dy/results/EXP-001/server_inventory.txt"}
adgs_root="$HOME/dy/work/AD-GS-Trust4D"
dggt_root="$HOME/dy/work/DGGT"

mkdir -p "$(dirname "$output")"

{
    date --iso-8601=seconds
    hostname
    uname -a

    echo "===== GPU ====="
    nvidia-smi
    nvcc --version || true

    echo "===== PYTHON/CONDA ====="
    python3 --version
    conda --version || true
    conda env list || true

    echo "===== GIT ====="
    git -C "$adgs_root" status --short --branch
    git -C "$adgs_root" rev-parse HEAD
    git -C "$dggt_root" status --short --branch
    git -C "$dggt_root" rev-parse HEAD

    echo "===== STORAGE ====="
    df -h "$nas_root"

    echo "===== RELEVANT DIRECTORIES ====="
    find "$nas_root" -maxdepth 5 -type d \
        \( -name scene006 -o -name scene026 -o -name scene090 \
        -o -name image -o -name images \) \
        | sort | sed -n '1,240p'

    echo "===== WAYMO ARTIFACTS ====="
    find "$nas_root" -maxdepth 7 -type f \
        \( -iname '*.tfrecord' -o -name cameras.npz \
        -o -name points3d.ply -o -name colmap.ply \) \
        | sort | sed -n '1,240p'

    echo "===== SCENE006 IMAGES ====="
    find "$nas_root" -maxdepth 8 -type f -path '*scene006*' \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        | sort | sed -n '1,60p'
} 2>&1 | tee "$output"
