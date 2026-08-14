# Trust4D-GS experiment ledger

This is the single source of truth for experiments run on the A40 server.
Protocols are committed before execution. Results are appended only from saved
commands, logs, JSON metrics, and artifacts returned from the server.

## Project contract

- Host method: official AD-GS commit
  `9a208512e49c8ddbaa20387921d9648adcd21cb4`.
- Foundation teacher: official DGGT commit
  `a3276d2bbe4cbb03bcc117830b1836110a27adeb`.
- Compute: one NVIDIA A40 48 GB.
- Server data root: `~/dy/nas/`; the exact dataset subpaths must be discovered,
  not guessed.
- DGGT and AD-GS use separate environments. DGGT exports an offline cache and
  exits before AD-GS starts. They never share GPU memory.
- Primary rendering metrics: PSNR, SSIM, and LPIPS(VGG) under the released AD-GS
  split and evaluator. Dynamic-only metrics are reported only after fixing the
  exact mask protocol.
- A DGGT paper number is never compared directly with an AD-GS number unless
  dataset, input views, split, target frames, masking, and evaluator match.
- Evaluation labels may diagnose a prior but may not enter training, query
  selection, intervention selection, checkpoint selection, or hyperparameter
  tuning.
- Every treatment has a same-commit, same-scene, same-seed AD-GS control.

## Research hypothesis

Foundation-model motion is useful to scene-specific dynamic Gaussian
optimization only where its prediction is stable under temporally equivalent
observations. Original/reversed/shifted/sparse clips should expose unreliable
motion; an anisotropic consensus should outperform direct distillation and a
scalar confidence weight, especially on hard dynamic regions.

This is stronger than “add DGGT to AD-GS.” It predicts a measurable relation
between intervention disagreement and teacher error before any AD-GS loss is
changed.

## Code-level literature findings

### DGGT

- `VGGT.forward(images, query_points=None)` returns image features, pose
  encoding, world points and confidence, Gaussian map and confidence, dynamic
  logits, semantic logits, depth and confidence, and input images.
- The built-in track head runs only when `query_points` is supplied.
- Released `inference.py` uses predicted depth and camera parameters to build
  its point map by default; it does not use `world_points` by default.
- Interpolation mode separately loads TAPIP3D from a placeholder checkpoint
  path. Therefore, “released inference produces complete 3D tracks” is not an
  accepted assumption.
- The released training path supervises the dynamic head with dataset dynamic
  masks. DGGT is an annotation-assisted frozen teacher even if Trust4D-GS does
  not expose those labels to AD-GS training.
- DGGT encodes OpenCV camera-from-world transforms. Its released Waymo loader
  places the first ego pose at the world origin, so reversing a clip changes
  the prediction reference. Raw original/reverse 3D coordinates are not
  comparable without chronological reordering and common-frame alignment.

### DynamicVGGT

- Its useful transferable idea is current/future point-map consistency and
  multi-horizon motion, not its full feed-forward renderer.
- Its paper reports strong point-map/depth results but lower Waymo full-image
  NVS than the optimization-based methods in its own table.
- No public code URL is provided in the local paper. It is not a v1 dependency.

## Existing baseline anchors (historical, not rerun on A40 yet)

The previous server completed the released 60k protocol on three Waymo scenes:

| Scene | PSNR | SSIM | LPIPS(VGG) |
|---|---:|---:|---:|
| scene006 | 34.9363 | 0.95216 | 0.18436 |
| scene026 | 31.7305 | 0.91144 | 0.26387 |
| scene090 | 30.7210 | 0.91145 | 0.24252 |

These are useful regression anchors. They do not constitute an eight-scene
table, and the new A40 setup must reproduce at least one scene before treatment
training.

## Experiment index

| ID | Protocol state | Execution state | Decision |
|---|---|---|---|
| EXP-000 | locked | complete (local, zero GPU) | DGGT is suitable only as an offline, separately-environmented teacher; inspect actual outputs before integration |
| EXP-001 | locked | waiting for A40 data-path inventory | DGGT single-clip output/VRAM contract |
| EXP-002 | draft; do not run | blocked by EXP-001 | intervention disagreement versus evaluation-only motion error |

## EXP-000 — source and protocol audit

### Inputs

- `new_papers/DGGT.pdf`, SHA-256
  `108f7ddb1ffce9b16a812e3f92a04413996a128fba0f902c334694fdb0e8104e`
- `new_papers/DynamicVGGT.pdf`, SHA-256
  `87b5886162281ee6711bd63e6ad471f55b5c80b0d11b44c0570fee2ac58f503a`
- `new_papers/Trust4D_GS_Code_Level_Research_Plan.md`, SHA-256
  `64a42e0d16f05e23a883895f46f06d3f75be405c05941d93a8aa0c2abf164b39`
- DGGT source at the commit recorded above.

### Result

The plan is directionally strong but too large for a first implementation. The
first falsifiable dependency is the teacher output contract, not an AD-GS loss.
Track extraction, coordinate alignment, intervention consensus, and motion
transfer remain unproven. We therefore proceed to EXP-001 only.

## EXP-001 — A40 DGGT single-clip contract probe

### Question

Can the released one-view Waymo DGGT checkpoint run on one A40 for a four-frame
front-camera clip and produce finite, shape-consistent outputs with enough
memory margin for intervention export?

### Locked input

- The first four AD-GS training frames of Waymo `scene006`, front camera, in
  chronological order.
- Training membership is selected by pairing sorted files in `image/` with
  `cameras.npz['is_val_list']` and taking entries where the flag is false. The
  image count must equal the metadata length; filename heuristics are not used.
- Images only; no validation/test frame and no evaluation label is read.
- Image paths will be recorded after the `~/dy/nas/` inventory identifies the
  real processed-data layout. The probe accepts only the JSON manifest emitted
  by `select_waymo_training_frames.py`; before loading the model it re-derives
  the selection from `cameras.npz` and verifies all four file hashes.
- DGGT checkpoint: official `model_latest_waymo.pt`, with its byte size and
  SHA-256 recorded before execution.
- Source metadata observed before download: Hugging Face repo revision
  `735ac9a6486057b1eb886c33a8c6dc79e0b43214`, linked size
  `5,411,266,466` bytes, and linked ETag
  `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def`.
  The downloaded file must match the size; its locally computed SHA-256 is the
  integrity record rather than an assumed interpretation of the ETag.

### Locked execution

- DGGT commit: `a3276d2bbe4cbb03bcc117830b1836110a27adeb`, with an
  empty working tree. A different commit or any tracked/untracked source change
  fails preflight before GPU work begins.
- The downloaded checkpoint size must be exactly `5,411,266,466` bytes before
  its SHA-256 is computed and model construction begins.
- PyTorch 2.4.1 / torchvision 0.19.1, separate `dggt` environment.
- Pin `gsplat==1.5.3`, the last PyPI release preceding the locked DGGT source
  commit. Add `scikit-learn` explicitly because `SkyGaussian.__init__` imports
  it although the released `requirements.txt` omits it. The CUDA wheel index is
  selected only after recording the A40 driver version.
- One forward pass after one warm-up pass, `model.eval()` and
  `torch.inference_mode()`.
- No diffusion, TAPIP3D, AD-GS, training, or query points.
- Record `nvidia-smi`, package versions, input filenames and tensor shape,
  dtype, device, finite fraction, min/max for every returned tensor; record
  wall time and `torch.cuda.max_memory_allocated()`.

### Pass gate

- Process exit code is zero.
- Source commit, clean-tree state, checkpoint byte size, and re-derived image
  manifest pass preflight.
- All numeric output tensors are finite.
- Required keys exist: `pose_enc`, `world_points`, `world_points_conf`,
  `gs_map`, `gs_conf`, `dynamic_conf`, `depth`, and `depth_conf`.
- Sequence dimension equals four for every per-frame output.
- Peak allocated memory is below 44 GiB, leaving at least 4 GiB operational
  margin on the A40.

Failure triggers diagnosis or a smaller clip/resolution probe; it does not
authorize changing AD-GS.

### Result

Pending.

## EXP-002 — intervention reliability diagnostic (draft, not authorized)

Only after EXP-001 passes, lock exact query selection, original/reverse/shifted
clips, coordinate canonicalization, evaluation-only motion target, Spearman,
AUROC, calibration bins, minimum track support, and pass thresholds. Run at
least three preselected scenes; do not choose scenes after seeing the score.
Reverse outputs must first be restored to chronological order, and each
intervention must be aligned to one common physical reference before computing
3D disagreement.

## Result return template

For each server run, return one directory or archive containing:

```text
experiment_id
git_remote
git_commit
git_status.txt
command.sh
stdout.log
stderr.log
exitcode.txt
nvidia-smi.txt
environment.txt
metrics.json
artifacts.sha256
wall_time_seconds
peak_gpu_memory_mib
```

Do not paste only the best metric. Include the exact command and complete log so
the next code change can be attributed to evidence rather than guesswork.
