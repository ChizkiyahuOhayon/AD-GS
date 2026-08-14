#!/usr/bin/env bash
set -euo pipefail

filename=individual_files_validation_segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord
gcs_object=validation/segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord
source_uri="gs://waymo_open_dataset_v_1_4_1/individual_files/$gcs_object"
target_dir=${1:-"$HOME/dy/nas/Trust4D-GS/waymo/raw"}
target="$target_dir/$filename"
partial="${target}.part"

command -v gcloud >/dev/null
gcloud auth print-access-token >/dev/null
mkdir -p "$target_dir"

gcloud storage objects describe "$source_uri" \
    --format='json(size,md5Hash,crc32c,generation,updateTime)' \
    > "${target}.remote.json"
remote_size=$(gcloud storage objects describe "$source_uri" --format='value(size)')

if [[ -e "$target" ]]; then
    local_size=$(wc -c < "$target" | tr -d '[:space:]')
    if [[ "$local_size" != "$remote_size" ]]; then
        echo "existing TFRecord size mismatch: local=$local_size remote=$remote_size" >&2
        exit 2
    fi
else
    gcloud storage cp "$source_uri" "$partial"
    local_size=$(wc -c < "$partial" | tr -d '[:space:]')
    if [[ "$local_size" != "$remote_size" ]]; then
        echo "downloaded TFRecord size mismatch: local=$local_size remote=$remote_size" >&2
        exit 2
    fi
    mv "$partial" "$target"
fi

sha256sum "$target" | tee "${target}.sha256"
printf 'source=%s\nsize_bytes=%s\n' "$source_uri" "$remote_size" \
    > "${target}.manifest.txt"
echo "Waymo scene006 TFRecord ready: $target"
