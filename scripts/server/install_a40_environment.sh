#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    printf 'usage: %s ENV_PREFIX REPO SOURCE_ENV_RECEIPT_DIR EVIDENCE_DIR\n' "$0" >&2
    exit 2
fi

env_prefix=$1
repo=$2
source_receipt=$3
evidence_dir=$4
conda=${CONDA_EXE:-$(command -v conda)}
cuda_home=/usr/local/cuda-11.8
python="$env_prefix/bin/python"

test -d "$repo/.git"
test -f "$source_receipt/conda-explicit-linux-64.txt"
test -f "$source_receipt/pip-freeze.txt"
test -f "$source_receipt/MANIFEST.sha256"
test -x "$cuda_home/bin/nvcc"
test "$(uname -m)" = x86_64
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
"$cuda_home/bin/nvcc" --version | grep -q 'release 11.8'
(cd "$source_receipt" && sha256sum -c MANIFEST.sha256)

mkdir -p "$evidence_dir" "${PIP_CACHE_DIR:-$HOME/dy/nas/adgs-gfdgs/cache/pip}"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/dy/nas/adgs-gfdgs/cache/pip}
export CUDA_HOME=$cuda_home
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=${MAX_JOBS:-8}

if [ ! -x "$python" ]; then
    "$conda" create -y -p "$env_prefix" \
        --file "$source_receipt/conda-explicit-linux-64.txt"
fi

"$python" -m pip install \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

runtime_lock="$evidence_dir/a40-runtime-lock.txt"
awk '
    / @ file:\/\// { next }
    /^pytorch3d==/ { next }
    /^nvidia-.*-cu12==/ { next }
    { print }
' "$source_receipt/pip-freeze.txt" > "$runtime_lock"
"$python" -m pip install -r "$runtime_lock"

"$python" -m pip install --no-deps --no-index \
    --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html \
    pytorch3d==0.7.4

"$python" -m pip install --no-build-isolation -e "$repo/submodules/simple-knn"
"$python" -m pip install --no-build-isolation \
    -e "$repo/submodules/depth-diff-gaussian-rasterization"

"$python" -m pip check | tee "$evidence_dir/a40-pip-check.txt"
"$python" -m pip freeze --all > "$evidence_dir/a40-pip-freeze.txt"
"$python" -c 'import diff_gaussian_rasterization, numpy, pytorch3d, roma, simple_knn, torch; print(torch.__version__, torch.version.cuda, numpy.__version__, pytorch3d.__version__, roma.__version__)' \
    | tee "$evidence_dir/a40-imports.txt"
sha256sum \
    "$runtime_lock" \
    "$evidence_dir/a40-pip-check.txt" \
    "$evidence_dir/a40-pip-freeze.txt" \
    "$evidence_dir/a40-imports.txt" \
    > "$evidence_dir/a40-install-manifest.sha256"
