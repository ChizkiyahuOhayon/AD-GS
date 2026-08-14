#!/usr/bin/env python3
"""Validate all released AD-GS inputs for Waymo scene006 training."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .validate_waymo_scene006 import file_sha256, validate_scene
except ImportError:
    from validate_waymo_scene006 import file_sha256, validate_scene


PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply_vertices(path, required_properties):
    path = Path(path)
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"{path.name} does not start with a PLY header")
        header_line_count = 1
        format_name = None
        vertex_count = None
        properties = []
        in_vertices = False
        for raw_line in stream:
            header_line_count += 1
            fields = raw_line.decode("ascii").strip().split()
            if not fields:
                continue
            if fields[0] == "format":
                format_name = fields[1]
            elif fields[0] == "element":
                in_vertices = fields[1] == "vertex"
                if in_vertices:
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and in_vertices:
                if fields[1] == "list":
                    raise ValueError(f"{path.name} has an unsupported vertex list property")
                properties.append((fields[2], fields[1]))
            elif fields[0] == "end_header":
                data_offset = stream.tell()
                break
        else:
            raise ValueError(f"{path.name} is missing end_header")

    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"{path.name} must contain vertices")
    property_names = {name for name, _ in properties}
    missing = sorted(set(required_properties) - property_names)
    if missing:
        raise ValueError(f"{path.name} is missing vertex properties: {missing}")

    if format_name == "ascii":
        values = np.loadtxt(path, skiprows=header_line_count)
        values = np.atleast_2d(values)
        if values.shape != (vertex_count, len(properties)):
            raise ValueError(f"{path.name} ASCII vertex payload has the wrong shape")
        vertices = {name: values[:, index] for index, (name, _) in enumerate(properties)}
    elif format_name in {"binary_little_endian", "binary_big_endian"}:
        order = "<" if format_name == "binary_little_endian" else ">"
        try:
            dtype = np.dtype(
                [(name, order + PLY_DTYPES[data_type]) for name, data_type in properties]
            )
        except KeyError as error:
            raise ValueError(f"{path.name} uses an unsupported PLY scalar type") from error
        expected_bytes = data_offset + vertex_count * dtype.itemsize
        if path.stat().st_size < expected_bytes:
            raise ValueError(f"{path.name} vertex payload is truncated")
        payload = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=vertex_count)
        vertices = {name: payload[name] for name, _ in properties}
    else:
        raise ValueError(f"{path.name} has unsupported PLY format: {format_name}")

    ranges = {}
    for name in required_properties:
        values = np.asarray(vertices[name])
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} property {name} contains nonfinite values")
        ranges[name] = [float(values.min()), float(values.max())]
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "format": format_name,
        "vertex_count": vertex_count,
        "vertex_properties": [name for name, _ in properties],
        "ranges": ranges,
    }, vertices


def array_record(path, array):
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def validate_flow(path, height, width, source_timestamp):
    with np.load(path, allow_pickle=True) as archive:
        if "flow" not in archive.files:
            raise ValueError(f"{path.name} is missing the flow key")
        records = archive["flow"]
    if len(records) == 0:
        raise ValueError(f"{path.name} contains no target records")

    targets = []
    for record in records:
        if len(record) != 6:
            raise ValueError(f"{path.name} flow record must contain six fields")
        target, intrinsic, rotation, translation, flow, visibility = record
        target = float(target)
        arrays = [
            np.asarray(intrinsic),
            np.asarray(rotation),
            np.asarray(translation),
            np.asarray(flow),
            np.asarray(visibility),
        ]
        expected_shapes = [(3, 3), (3, 3), (3,), (2, height, width), (height, width)]
        if [value.shape for value in arrays] != expected_shapes:
            raise ValueError(f"{path.name} flow record has incompatible shapes")
        if not np.isfinite(target) or not all(np.isfinite(value).all() for value in arrays):
            raise ValueError(f"{path.name} flow record contains nonfinite values")
        if not 0 <= target <= 85 or target == source_timestamp:
            raise ValueError(f"{path.name} has an invalid target timestamp: {target}")
        targets.append(target)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "record_count": len(records),
        "target_timestamps": targets,
    }


def validate_baseline_scene(scene_path):
    scene = Path(scene_path).expanduser().resolve()
    base = validate_scene(scene)
    image_shapes = [(item["height"], item["width"]) for item in base["images"]]
    prior_records = {"depth": [], "semantic": [], "sky": []}
    semantic_positive = []
    mask_positive = {"semantic": 0, "sky": 0}

    for index, (height, width) in enumerate(image_shapes):
        stem = f"{index:06d}"
        for family in ("depth", "semantic", "sky"):
            filename = f"{stem}.npy" if family == "depth" else f"mask_{stem}.npy"
            path = scene / family / filename
            if not path.is_file():
                raise ValueError(f"missing {family} prior: {path}")
            array = np.load(path, allow_pickle=False)
            expected_shape = (height, width, 1) if family == "depth" else (height, width)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{family} prior {filename} shape must be {expected_shape}, got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{family} prior {filename} contains nonfinite values")
            if family == "depth":
                if array.min() < 0 or array.max() > 1 or array.max() <= array.min():
                    raise ValueError(f"depth prior {filename} is not a nonconstant [0,1] map")
            else:
                if not np.issubdtype(array.dtype, np.integer) or array.min() < 0:
                    raise ValueError(f"{family} prior {filename} must be nonnegative integers")
                positive = int(np.count_nonzero(array))
                mask_positive[family] += positive
                if family == "semantic":
                    semantic_positive.append(positive)
            prior_records[family].append(array_record(path, array))

    for family, count in mask_positive.items():
        if count == 0:
            raise ValueError(f"{family} priors contain no positive pixels in the scene")

    points_summary, points = read_ply_vertices(
        scene / "points3d.ply", {"x", "y", "z", "t", "obj"}
    )
    obj = np.asarray(points["obj"])
    static_count = int(np.count_nonzero(obj <= 0.5))
    object_count = int(np.count_nonzero(obj > 0.5))
    if static_count == 0 or object_count == 0:
        raise ValueError("points3d.ply must contain both static and object points")
    points_summary.update(static_vertex_count=static_count, object_vertex_count=object_count)

    flow_records = []
    training_indices = set(base["training_indices"])
    for index, positive in enumerate(semantic_positive):
        if index not in training_indices or positive == 0:
            continue
        path = scene / "flow" / f"{index:06d}.npz"
        if not path.is_file():
            raise ValueError(f"dynamic training frame {index} is missing flow: {path}")
        height, width = image_shapes[index]
        flow_records.append(validate_flow(path, height, width, index))

    colmap_summary, _ = read_ply_vertices(scene / "colmap.ply", {"x", "y", "z"})
    return {
        "scene": str(scene),
        "base_scene": base,
        "priors": prior_records,
        "positive_mask_pixels": mask_positive,
        "points3d": points_summary,
        "flow": flow_records,
        "required_dynamic_training_flow_count": len(flow_records),
        "colmap": colmap_summary,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = validate_baseline_scene(args.scene)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
