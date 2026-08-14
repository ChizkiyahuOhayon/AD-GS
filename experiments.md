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
| DATA-001 | locked | waiting for authenticated Waymo path/download | Prepare and validate only AD-GS Waymo scene006 |
| DATA-002 | locked | not started; blocked by DATA-001 and pseudo-label generation | Validate complete scene006 inputs before the A40 baseline |
| ENV-001 | locked | not started | Build a pinned, isolated AD-GS train/render environment |
| EXP-001 | locked | inventory complete; Waymo path unresolved; no GPU run | DGGT single-clip output/VRAM contract |
| BASE-001 | locked | not started; blocked by DATA-002 and ENV-001 | Reproduce the released 60k AD-GS scene006 baseline on one A40 |
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

## DATA-001 — Waymo scene006 acquisition and AD-GS preprocessing

This preparation is locked before downloading or preprocessing on the new
server. It is not a rendering experiment and consumes no evaluation metric.

### Locked source and output

- Source object:
  `gs://waymo_open_dataset_v_1_4_1/individual_files/individual_files_validation_segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord`.
- Access requires the user's accepted Waymo license and authenticated Google
  Cloud account; credentials and access tokens are never committed or logged.
- Raw and processed data live below `~/dy/nas/Trust4D-GS/waymo/`, outside both
  Git repositories. Downloads record remote size and local SHA-256.
- Preprocessor: `scripts/waymo/waymo.py`, SHA-256
  `f87905fb7c867d572679a6b7ea92dbe4b085d5a4a695f70ee1776c2058188bd6`.
- Exact arguments: `--first_frame 0 --last_frame 85 --use_color`; front camera
  remains the released default.
- A separate CPU-only Python 3.10 environment uses TensorFlow 2.11.0 and
  `waymo-open-dataset-tf-2-11-0==1.6.1`, plus CPU-only PyTorch 1.13.1 for the
  released color projection's `grid_sample`; it never shares the DGGT or
  AD-GS training environment.
- Install the sole official Linux Waymo 1.6.1 wheel directly, require exact
  size `4,041,796` bytes and SHA-256
  `2d0d4a4bbc59fe2fc8e19e9958f560383e81454e1c0662ad2e642d0797ce7f6e`,
  and retain the released README's `--no-dependencies` policy. This pins the
  compiled dataset reader without importing the wheel's full historical
  dependency set into the deliberately minimal preprocessing environment.

### Data pass gate

- Exactly 86 decodable images and 86 entries in `R`, `T`, `K`, `time_stamps`,
  and `is_val_list`.
- Exact metadata shapes are `R: (86,3,3)`, `T: (86,3)`, `K: (86,9)`, and
  one-dimensional `(86,)` timestamps and boolean split flags.
- Timestamps are exactly 0 through 85; validation indices are exactly
  `4, 8, ..., 84`, matching released `get_val_frames(..., test_every=4)`.
- Camera arrays contain only finite values. `points3d.ply` is nonempty and has
  a positive vertex count, a nonempty payload, and `x`, `y`, `z`, and `t`
  properties.
- The first four chronological training images are selected only through the
  locked split selector and every artifact SHA-256 is recorded.

Failure stops DATA-001; a partial scene cannot enter EXP-001 or baseline
training.

## DATA-002 — complete AD-GS Waymo baseline-input gate

DATA-001 is sufficient for the image-only DGGT probe but not for AD-GS
training. After the released depth, semantic/sky, flow, point segmentation,
and COLMAP stages run, one validator must establish baseline readiness before
any 60k optimization starts.

### Locked gate

- Preserve every DATA-001 image/camera/split invariant.
- Require exactly one finite `depth/NNNNNN.npy`,
  `semantic/mask_NNNNNN.npy`, and `sky/mask_NNNNNN.npy` for each of the 86
  images. Each prior must match its source image height and width. Depth has
  shape `(H,W,1)`, lies in `[0,1]`, and has positive within-frame range;
  semantic and sky masks are nonnegative integer arrays. Each mask family has
  at least one positive pixel over the scene.
- Require `points3d.ply` to retain finite `x,y,z,t` values and contain the
  segmentation property `obj`, with both static and object points present.
- For every training frame whose semantic mask is nonempty, require a
  corresponding `flow/NNNNNN.npz` with at least one finite target record and
  finite flow/visibility arrays at the image resolution. A missing flow on a
  dynamic training frame is a failed pseudo-label stage, not a warning to
  ignore. Validation-frame flow is not required.
- Because released Waymo configuration defaults to `use_colmap=True`, require
  a nonempty `colmap.ply` with finite `x,y,z` vertices.
- Record file sizes and SHA-256 digests in one JSON result. Any failure stops
  baseline training; the gate is not relaxed after observing metrics.

## ENV-001 — pinned AD-GS baseline runtime

The released `environment.yaml` pins Python 3.7 and PyTorch 1.13.1, while
PyTorch3D v0.7.2 and later officially require Python 3.8--3.10 and the last
release supporting Python 3.7 (v0.7.1) lists support only through PyTorch
1.12. Installing an unversioned 2026 PyTorch3D `main` is not a reproduction.

### Locked resolution

- Use a new Conda environment named `trust4d-adgs-baseline`; never reuse an
  inventory environment based only on its name.
- Change only Python 3.7 to Python 3.8. Preserve released PyTorch 1.13.1,
  torchvision 0.14.1, and CUDA 11.7 runtime wheels. Python minor version is
  treated as infrastructure, not a treatment variable.
- Pin PyTorch3D v0.7.4 to commit
  `297020a4b1d7492190cb4a909cafbd2c81a12cb5`. It is the first stable choice in
  the released Torch generation whose documented Python range includes 3.8;
  only `pytorch3d.ops.knn_points` is used by AD-GS.
- Compile the vendored `simple-knn` tree
  `d5b756edadeef66644510a23633c23803d6b61db` and
  `depth-diff-gaussian-rasterization` tree
  `b78a10882e6a99927b74303a83cb2c107666cdd3` from temporary copies so build
  artifacts do not dirty the research checkout. Use system CUDA 11.8 only as
  the extension compiler, with A40 architecture `8.6`; the PyTorch runtime
  remains CUDA 11.7.
- Install only packages imported by train/render and their resolved
  dependencies, using the released versions where specified. COLMAP, DPT,
  Grounded-SAM-2, CoTracker, and DGGT remain separate preparation/teacher
  environments.
- Populate and test the official evaluator cache before training. Torchvision
  0.14.1 must fetch its declared AlexNet `IMAGENET1K_V1` file
  `alexnet-owt-7be5be79.pth` and VGG16 `IMAGENET1K_V1` file
  `vgg16-397923af.pth`; torchvision enforces the SHA-256 filename prefixes and
  the evidence records the full local digests. Pin LPIPS v0.1 weights to
  PerceptualSimilarity commit
  `082bb24f84c091ea94de2867d34c4544f68e0963`: `alex.pth` has SHA-256
  `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`
  and `vgg.pth` has SHA-256
  `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868`.
  Execute both released LPIPS paths on fixed tensors and require finite,
  nonnegative outputs. This preserves the released evaluator while removing
  its runtime dependency on a mutable `master` URL.
- Before acceptance, require exact critical package versions, import both CUDA
  extensions, execute `pytorch3d.ops.knn_points` and `simple_knn.distCUDA2` on
  physical GPU 0, and retain `pip freeze`, compiler/GPU inventory, build logs,
  evaluator-cache hashes, and smoke-test JSON files. Failure stops baseline
  training.

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
- Use the official PyTorch CUDA 12.1 wheel index. Install the precompiled
  CPython 3.10 Linux wheel
  `gsplat-1.5.3+pt24cu121-cp310-cp310-linux_x86_64.whl`, SHA-256
  `0493bab68ed5fc71f4ce8bfc2be03b584d8a41a06a6d9362e09a795340f8c488`,
  rather than compiling against the server's CUDA 11.8 `nvcc`.
- Remaining packages come from locked DGGT `requirements.txt`, SHA-256
  `33428d6d7da037a7a344ea230819901493d2dbf9ca3c2b264fdeab689722854f`;
  the complete resolved `pip freeze` is retained by the evidence runner.
- `probe_dggt.py` rejects any installed PyTorch, torchvision, or gsplat version
  that differs from the pins above (a CUDA local-version suffix is allowed),
  and rejects a missing scikit-learn installation.
- The probe is launched only through `run_exp001.sh` from a clean AD-GS
  worktree. The runner refuses an existing output directory, creates the
  four-frame manifest itself, and records both repositories, the command,
  environment, GPU snapshot, stdout, stderr, exit code, wall time, metrics, and
  artifact hashes in one result directory.
- The runner invokes every Python command through the named
  `trust4d-dggt-exp001` Conda environment; it does not depend on the caller's
  active shell environment. The deserialized CPU checkpoint is released as
  soon as strict state-dict loading succeeds.
- One forward pass after one warm-up pass, `model.eval()` and
  `torch.inference_mode()`.
- Bind the process to physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`; the probe
  verifies that logical `cuda:0` is an NVIDIA A40 and records its runtime total
  memory.
- No diffusion, TAPIP3D, AD-GS, training, or query points.
- Record `nvidia-smi`, package versions, input filenames and tensor shape,
  dtype, device, finite fraction, min/max for every returned tensor; record
  wall time and `torch.cuda.max_memory_allocated()`.

### Pass gate

- Process exit code is zero.
- Source commit, clean-tree state, checkpoint byte size, and re-derived image
  manifest pass preflight.
- The locked package versions pass runtime preflight.
- All numeric output tensors are finite.
- Required keys exist: `pose_enc`, `world_points`, `world_points_conf`,
  `gs_map`, `gs_conf`, `dynamic_conf`, `depth`, and `depth_conf`.
- Sequence dimension equals four for every per-frame output.
- Runtime total memory minus peak PyTorch reserved memory is at least
  `4096 MiB`. Peak allocated memory remains recorded but is not substituted for
  reserved memory in this safety gate.

The inventory showed `46068 MiB` total, so the earlier fixed `<44 GiB` test
would have left only about `1012 MiB`, not the claimed 4 GiB. This dynamic gate
is the pre-GPU correction for decimal GB versus binary GiB and allocator cache.

Failure triggers diagnosis or a smaller clip/resolution probe; it does not
authorize changing AD-GS.

### Result

Preflight inventory received at `2026-08-14T15:59:23+08:00` and recorded in
`server.md`. The driver (`535.309.01`) is compatible with a PyTorch CUDA 12.1
wheel, physical GPU 0 was idle, and DGGT was at the locked commit. The server
AD-GS checkout was still at `aa28448` and must be updated. The bounded NAS
search found no authentic Waymo TFRecord or processed `scene006`; its
`scene0060_00` hits are ScanNet and are invalid for this experiment. EXP-001
therefore has not executed and remains pending data-path resolution.

## BASE-001 — released AD-GS Waymo scene006 reproduction

### Question

Does the released AD-GS implementation reproduce its historical scene006
result on the new A40 closely enough to serve as a matched control for later
treatments?

### Locked source, input, and runtime

- Train and render with the official AD-GS source at commit
  `9a208512e49c8ddbaa20387921d9648adcd21cb4`. The research branch may add only
  `experiments.md`, `server.md`, `scripts/trust4d/`, and `tests/`; any modified
  or deleted official file fails preflight.
- Require a clean research worktree, a DATA-002 pass result produced from the
  exact scene path, and an ENV-001 smoke-test pass in the named
  `trust4d-adgs-baseline` environment.
- Use the released `arguments/waymo.py`, all default optimization parameters,
  one front camera, the released validation split, and exactly 60,000
  iterations. Do not override resolution, losses, densification, or sampling.
- Use physical GPU 0 only via `CUDA_VISIBLE_DEVICES=0`; inside the process the
  device is `cuda:0`. The runner verifies an NVIDIA A40 before training.
- The released `safe_state` fixes Python, NumPy, and Torch RNG seeds to zero.
  No checkpoint selection or repeated-seed best-run selection is allowed.
- Execute the released sequence:
  `python train.py -c arguments/waymo.py -s <scene006> -m <new-run-dir>
  --data_device cuda:0`, followed by
  `python render.py -c arguments/waymo.py -m <run-dir> --data_device cuda:0
  -v`. Both commands run through the pinned Conda environment.
- The run and evidence directories must not exist before launch. Record exact
  commands, both repository states, source-diff audit, DATA-002 and ENV-001
  results, package freeze, GPU snapshots, complete stdout/stderr, exit codes,
  wall times, metric JSON files, and SHA-256 manifests.

### Locked pass gate

- Training and rendering both exit zero; the saved checkpoint is exactly
  `point_cloud/iteration_60000/{point_cloud.ply,env.pth}` and the evaluator
  reports an `ours_60000` record in both `results.json` and
  `results-train.json`.
- Test PSNR, SSIM, LPIPS(VGG), LPIPS(ALEX), and FPS are finite; PSNR and FPS
  are positive, SSIM lies in `[0,1]`, and both LPIPS values are nonnegative.
- Compare only against the pre-existing scene006 historical anchor: PSNR
  `34.9363`, SSIM `0.95216`, LPIPS(VGG) `0.18436`. Reproduction requires
  absolute deviations no greater than `0.50 dB`, `0.010`, and `0.020`,
  respectively. These limits are fixed before the A40 result is observed and
  are not relaxed after execution.
- A metric outside tolerance is a failed reproduction requiring diagnosis,
  not evidence against or for the research hypothesis. Treatment training
  remains unauthorized until BASE-001 passes.

The tolerance gate checks baseline fidelity only. A later SOTA claim requires
same-budget paired controls across all eight released Waymo scenes and then
the predeclared KITTI/nuScenes transfer evaluation; a scene006-only gain is
insufficient.

### Result

Not executed. DATA-002 and ENV-001 must pass first.

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
