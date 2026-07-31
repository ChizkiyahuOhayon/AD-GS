"""Extract fixed sparse held-out LiDAR samples without altering scene data."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.geometry_metrics import select_sparse_actor_depth


OPENCV_TO_DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecord")
    parser.add_argument("scene_dir")
    parser.add_argument("output_dir")
    parser.add_argument("actor_config")
    parser.add_argument("--first_frame", type=int, default=0)
    parser.add_argument("--last_frame", type=int, default=102)
    parser.add_argument("--camera_index", type=int, default=0)
    return parser.parse_args()


def camera_projection(frame, camera_name, points, intrinsic):
    calibration = next(
        item for item in frame.context.camera_calibrations if item.name == camera_name
    )
    camera_to_vehicle = np.asarray(calibration.extrinsic.transform).reshape(4, 4)
    vehicle_to_camera = np.linalg.inv(camera_to_vehicle @ OPENCV_TO_DATASET)
    camera_xyz = (
        vehicle_to_camera[:3, :3] @ points.T + vehicle_to_camera[:3, 3:4]
    ).T
    K = np.array(
        [[intrinsic[0], 0.0, intrinsic[2]],
         [0.0, intrinsic[1], intrinsic[3]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected = (K @ camera_xyz.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return uv, camera_xyz[:, 2]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(manifest_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.actor_config) as handle:
        actor_config = json.load(handle)
    stable_actor_ids = actor_config["actor_ids"]

    scene_dir = Path(args.scene_dir)
    cameras = np.load(scene_dir / "cameras.npz")
    intrinsics = cameras["K"]
    timestamps = cameras["time_stamps"]
    validation = cameras["is_val_list"]
    camera_name = args.camera_index + 1

    records = []
    dataset = tf.data.TFRecordDataset(args.tfrecord, compression_type="")
    for absolute_fid, data in enumerate(dataset):
        if absolute_fid < args.first_frame:
            continue
        if absolute_fid > args.last_frame:
            break
        relative_fid = absolute_fid - args.first_frame
        image_indices = np.flatnonzero(timestamps == relative_fid)
        if args.camera_index >= len(image_indices):
            raise ValueError("camera index is absent at frame {}".format(relative_fid))
        image_id = int(image_indices[args.camera_index])
        if not bool(validation[image_id]):
            continue

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        range_images, camera_projections, _, top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        point_groups, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            top_pose,
            ri_index=0,
        )
        points = np.concatenate(point_groups, axis=0)
        uv, depth = camera_projection(
            frame, camera_name, points, intrinsics[image_id]
        )
        actor_map = np.load(
            scene_dir / "semantic" / "mask_{:06d}.npy".format(image_id)
        )
        samples = select_sparse_actor_depth(
            uv, depth, actor_map, stable_actor_ids
        )
        sample_path = output_dir / "{:06d}.npz".format(image_id)
        if sample_path.exists():
            raise FileExistsError("refusing to overwrite {}".format(sample_path))
        np.savez_compressed(
            sample_path,
            image_id=image_id,
            relative_fid=relative_fid,
            absolute_fid=absolute_fid,
            image_shape=np.asarray(actor_map.shape, dtype=np.int64),
            **samples
        )
        record = {
            "image_id": image_id,
            "relative_fid": relative_fid,
            "sample_count": int(samples["depth"].size),
            "actor_ids": sorted(int(value) for value in np.unique(samples["actor_id"])),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    manifest = {
        "tfrecord": str(Path(args.tfrecord).name),
        "scene": actor_config["scene"],
        "camera_index": args.camera_index,
        "stable_actor_ids": stable_actor_ids,
        "frame_count": len(records),
        "sample_count": sum(item["sample_count"] for item in records),
        "observed_actor_ids": sorted(
            {actor_id for item in records for actor_id in item["actor_ids"]}
        ),
        "frames": records,
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
