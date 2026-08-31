import numpy as np


def validate_camera_split(selected_camera_ids, train_camera_ids=None):
    selected = tuple(dict.fromkeys(int(camera_id) for camera_id in selected_camera_ids))
    if not selected:
        raise ValueError("at least one Waymo camera must be selected")
    if any(camera_id < 0 for camera_id in selected):
        raise ValueError("Waymo camera IDs must be non-negative")

    if train_camera_ids is None:
        return selected

    train = tuple(dict.fromkeys(int(camera_id) for camera_id in train_camera_ids))
    if not train:
        raise ValueError("at least one selected camera must be used for training")
    unknown = sorted(set(train) - set(selected))
    if unknown:
        raise ValueError("training cameras are not selected: {}".format(unknown))
    return train


def is_validation_camera(frame_is_validation, camera_id, train_camera_ids):
    return bool(frame_is_validation) or int(camera_id) not in train_camera_ids


def load_camera_ids(metadata, image_count, fallback_num_cameras):
    if "camera_ids" in metadata:
        camera_ids = np.asarray(metadata["camera_ids"], dtype=np.int64)
    else:
        if fallback_num_cameras < 1:
            raise ValueError("fallback_num_cameras must be positive")
        camera_ids = np.arange(image_count, dtype=np.int64) % fallback_num_cameras
    if camera_ids.shape != (image_count,):
        raise ValueError("camera_ids must have one entry per image")
    return camera_ids


def infer_frame_gap(time_stamps):
    unique_times = np.unique(np.asarray(time_stamps))
    if unique_times.size < 2:
        raise ValueError("at least two Waymo timestamps are required")
    return 1.0 / unique_times.size


def summarize_camera_metrics(records_by_camera):
    summary = {}
    for camera_id, records in sorted(records_by_camera.items()):
        if not records:
            continue
        metric_names = tuple(records[0])
        if any(tuple(record) != metric_names for record in records):
            raise ValueError("camera metric records must share the same fields")
        summary[str(camera_id)] = {
            "count": len(records),
            **{
                name: float(np.mean([record[name] for record in records]))
                for name in metric_names
            },
        }
    return summary
