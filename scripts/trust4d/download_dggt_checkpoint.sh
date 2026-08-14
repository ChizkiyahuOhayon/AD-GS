#!/usr/bin/env bash
set -euo pipefail

target=${1:-"$HOME/dy/checkpoints/dggt/model_latest_waymo.pt"}
source_url='https://huggingface.co/xiaomi-research/dggt/resolve/main/model_latest_waymo.pt?download=true'
expected_size=5411266466
partial="${target}.part"

command -v curl >/dev/null
mkdir -p "$(dirname "$target")"

if [[ -e "$target" ]]; then
    actual_size=$(stat -c '%s' "$target")
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "existing checkpoint has wrong size: $actual_size" >&2
        exit 2
    fi
else
    curl --fail --location --retry 5 --continue-at - \
        --output "$partial" "$source_url"
    actual_size=$(stat -c '%s' "$partial")
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "downloaded checkpoint has wrong size: $actual_size" >&2
        exit 2
    fi
    mv "$partial" "$target"
fi

sha256sum "$target" | tee "${target}.sha256"
printf 'source=%s\nsize_bytes=%s\n' "$source_url" "$expected_size" \
    > "${target}.manifest.txt"
echo "DGGT checkpoint ready: $target"
