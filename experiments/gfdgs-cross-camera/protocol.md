# GF-DGS Waymo cross-camera protocol

Status: locked before the first cross-camera training run
Primary development scene: `scene090`

## Purpose

The official AD-GS split measures interpolation on the FRONT camera and is
nearly insensitive to the scale--range gauge targeted by GF-DGS. This protocol
adds a matched, ground-truth extrapolation test without synthesizing cameras:
train only on Waymo FRONT images and evaluate the recorded FRONT_LEFT and
FRONT_RIGHT images at their calibrated poses.

## Data construction

Create a separate dataset directory; never overwrite the official one-camera
baseline data.

```bash
python scripts/waymo/waymo.py <scene090.tfrecord> <cross-camera-scene090> \
  --first_frame 0 --last_frame 102 --select_camera 0 1 2 \
  --train_cameras 0 --use_color --use_depth
```

- Camera IDs are Waymo zero-based IDs: `0=FRONT`, `1=FRONT_LEFT`,
  `2=FRONT_RIGHT`.
- FRONT keeps the official every-fourth-frame validation split.
- Every side-camera image is test-only, including timestamps whose FRONT image
  is used for training.
- Only non-validation FRONT images contribute to point visibility and color
  during point-cloud construction. Side-camera pixels must not influence model
  initialization, pseudo-label training, or optimization.
- `cameras.npz` stores `camera_ids` explicitly; no modulo inference is allowed
  for this protocol.

Run the leakage audit before generating priors or training:

```bash
python scripts/waymo/verify_cross_camera.py <cross-camera-scene090> \
  --output <evidence-dir>/scene090-cross-camera-audit.json
```

## Matched comparison

- Baseline: pristine AD-GS at the locked upstream commit.
- Treatment: the selected GF-DGS commit.
- Both arms use the same cross-camera scene, seed, iteration budget, resolution,
  priors, renderer, and evaluator.
- FRONT_LEFT and FRONT_RIGHT never enter either arm's optimizer.
- Development runs do not replace the official one-camera 60k baseline table.

## Metrics

`render.py` writes the unchanged aggregate `results.json` and an additional
`results-test-by-camera.json`. Report:

1. FRONT interpolation PSNR / SSIM / LPIPS;
2. FRONT_LEFT and FRONT_RIGHT metrics separately;
3. the unweighted mean of the two side-camera metrics;
4. training time, peak memory, and Gaussian count.

The first run establishes the cross-camera baseline; no absolute target is
invented before observing it. GF-DGS advances only if both side-camera PSNRs
improve in the matched comparison and FRONT interpolation does not materially
degrade under the thresholds already fixed in the GF-DGS research plan.
