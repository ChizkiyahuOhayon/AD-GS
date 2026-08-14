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
| DATA-003 | locked | manifest/downloader and multi-scene preprocessor ready at `2625abb`; not executed | Acquire exact three-scene diagnostic subset first; defer five main-table scenes until the research gate passes |
| DATA-002 | locked | runner ready at `80ade52`; server execution blocked by DATA-001 and ENV-002 | Validate complete scene006 inputs before the A40 baseline |
| ENV-002 | locked | setup and smoke code ready at `eaae6cc`; not executed on the server | Build pinned, separate depth/segmentation/flow/COLMAP preparation runtimes |
| ENV-001 | locked | setup/smoke code ready; official-source audit passes at `13d679d`; not executed | Build a pinned, isolated AD-GS train/render environment |
| EXP-001 | locked | inventory complete; Waymo path unresolved; no GPU run | DGGT single-clip output/VRAM contract |
| BASE-001 | locked | not started; blocked by DATA-002 and ENV-001 | Reproduce the released 60k AD-GS scene006 baseline on one A40 |
| EXP-002 | locked | train-only query-manifest builder ready at `13d679d`; execution blocked by EXP-001 and three-scene semantic preparation | intervention disagreement versus evaluation-only actor motion error |

The source audit is compiled into the Seal-L1 research artifact at
`research/ara/trust4d_teacher_reliability/` (commit `55c9e54`). Its executable
reference fixes the reverse-query and per-invocation Sim(3) conventions used
by EXP-002; it does not modify or claim a result for AD-GS.

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
  `gs://waymo_open_dataset_v_1_4_1/individual_files/validation/segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord`.
  The released README's longer
  `individual_files_validation_segment-...` string is the required local
  filename, not the GCS object key. Keep the two names distinct.
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

## DATA-003 — cost-gated official Waymo scene manifest

This acquisition protocol is locked before implementing the multi-scene
downloader. It preserves the final eight-scene AD-GS comparison while avoiding
unnecessary data preparation before the research hypothesis passes its cheap
gates.

### Authoritative manifest

- Derive scene names, local filenames, and inclusive frame ranges directly
  from the unchanged official `scripts/waymo/prepare-waymo.sh` at AD-GS commit
  `9a208512e49c8ddbaa20387921d9648adcd21cb4`.
- Use bucket prefix
  `gs://waymo_open_dataset_v_1_4_1/individual_files` and the official object
  form `validation/segment-...tfrecord`. The longer
  `individual_files_validation_segment-...` form remains only the local
  filename expected by AD-GS.
- Lock the following eight records:

| Scene | Inclusive frames | GCS object basename |
|---|---:|---|
| scene006 | 0--85 | `segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord` |
| scene026 | 0--100 | `segment-12374656037744638388_1412_711_1432_711_with_camera_labels.tfrecord` |
| scene090 | 0--102 | `segment-17612470202990834368_2800_000_2820_000_with_camera_labels.tfrecord` |
| scene105 | 20--186 | `segment-1906113358876584689_1359_560_1379_560_with_camera_labels.tfrecord` |
| scene108 | 20--115 | `segment-2094681306939952000_2972_300_2992_300_with_camera_labels.tfrecord` |
| scene134 | 106--198 | `segment-4246537812751004276_1560_000_1580_000_with_camera_labels.tfrecord` |
| scene150 | 96--197 | `segment-5372281728627437618_2005_000_2025_000_with_camera_labels.tfrecord` |
| scene181 | 0--160 | `segment-8398516118967750070_3958_000_3978_000_with_camera_labels.tfrecord` |

### Cost gate and download evidence

- Phase A contains exactly `scene006`, `scene026`, and `scene090`. It supports
  BASE-001 and EXP-002. Download no other scene before EXP-002 authorizes a
  teacher treatment and the first same-budget treatment comparison is
  positive.
- Phase B contains `scene105`, `scene108`, `scene134`, `scene150`, and
  `scene181`. It is required for an eight-scene Waymo main table, but remains
  blocked by the Phase-A research gates rather than by availability.
- The downloader requires explicit scene IDs and rejects unknown or duplicate
  IDs. It queries remote object metadata before transfer and validates exact
  byte size plus the official base64 MD5 when supplied by GCS.
- An existing final file is reused only after the same checks. Downloads stage
  under the raw-data directory and are atomically promoted to the AD-GS local
  filename; corrupt staged or final files cause failure and are never silently
  overwritten.
- A new evidence directory records the selected manifest subset, remote
  metadata, commands, local SHA-256, GCS generation/ETag/checksums, exit code,
  and artifact hashes. Credentials and access tokens are neither arguments nor
  evidence artifacts; authentication remains in the user's `gcloud` profile.
- Multi-scene preprocessing requires the successful download receipt for the
  same explicit scene subset, rechecks each raw SHA-256, and invokes the
  unchanged official `scripts/waymo/waymo.py` with that scene's complete
  inclusive frame range and `--use_color`. Scenes execute serially in the
  isolated `trust4d-waymo-prep` CPU environment.
- For each scene let `N = last_frame - first_frame + 1`. Require exactly the
  consecutive images `000000.jpg` through `N-1`, camera-array shapes
  `R:(N,3,3)`, `T:(N,3)`, `K:(N,9)`, one-dimensional `time_stamps` and
  `is_val_list`, timestamps `0..N-1`, and boolean validation indices
  `4,8,... < N`. Require finite camera arrays and a nonempty `points3d.ply`
  with positive vertex count and `x,y,z,t` properties.
- Every processed scene path and the preprocessing evidence directory must be
  new. A failed scene stops later scenes and retains its partial directory and
  logs for diagnosis; it is never passed onward or silently reused.
- Raw records and evidence live under `~/dy/nas/Trust4D-GS/`, outside Git.
  Downloading is CPU/network work and does not reserve the A40.

Any mismatch stops acquisition. The official preprocessing frame ranges are
not shortened to fit only the preregistered EXP-002 anchors.

## DATA-002 — complete AD-GS Waymo baseline-input gate

DATA-001 is sufficient for the image-only DGGT probe but not for AD-GS
training. After the released depth, semantic/sky, flow, point segmentation,
and COLMAP stages run, one validator must establish baseline readiness before
any 60k optimization starts.

### Locked generators and runtimes (ENV-002)

- Preserve the released AD-GS generator files byte-for-byte:
  `run-dpt.py` SHA-256
  `f02545c46410b3fe8bc6b4a527ddf99a9383901eabe272c2a4e0e7450e605a1d`,
  `semantic.py` SHA-256
  `8429c19c6b50591103342d1205391e805866f1745a2d7cb4109e364f2911198f`,
  `flow.py` SHA-256
  `5d4725c7f25d077aaef31fbade55f0e9ba58b26282321b87f0bde14b5969c083`,
  `segment_pcd.py` SHA-256
  `acf98743fff4572cfbb0ea14a624398544b4b1eadaa3e2d2e9ea5c68802cdb57`,
  and `colmap.py` SHA-256
  `070437ba06cbeaf784ddc0508ebbd016936047c71519f588175ca9ecf7cd1cca`.
- Depth uses a separate Python 3.11 environment with PyTorch 2.4.1 / CUDA
  11.8 and Depth-Anything-V2 commit
  `e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1`, the latest upstream commit
  preceding the AD-GS code release. Pin the ViT-L checkpoint repository to
  revision `cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9`; the file is
  `1,341,395,338` bytes with SHA-256
  `a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345`.
- Segmentation uses a separate Python 3.10 environment with PyTorch 2.4.1 /
  CUDA 11.8 and Grounded-SAM-2 tag-v1 commit
  `dd4c5141b75e4838dd486c64f773c43b4db3a07b`. Compile its connected-component
  CUDA extension against the observed system CUDA 11.8 for A40 architecture
  8.6. Use `sam2.1_hiera_large.pt`, exactly `898,083,611` bytes and SHA-256
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
  Pin the Hugging Face Grounding DINO base snapshot to revision
  `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`; its preferred
  `model.safetensors` is `933,400,872` bytes with SHA-256
  `5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21`.
  The released script uses Transformers' model and does not import the local
  GroundingDINO extension, so that unrelated README installation is omitted.
- Flow uses a separate Python 3.10 / PyTorch 2.4.1 / CUDA 11.8 environment and
  CoTracker3 commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`, the
  upstream head already present when AD-GS was released. Populate an isolated
  `TORCH_HOME` cache so the released, otherwise-unpinned `torch.hub.load`
  resolves only that checkout. Pin `scaled_offline.pth` to Hugging Face
  revision `bf55ea50d4390e1820a267f131cd6587240fb2c5`; it is `101,890,938`
  bytes with SHA-256
  `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`.
- COLMAP uses a CPU-only Python 3.10 environment with the released
  `colmap=3.7` binary. It does not claim GPU 0 while a neural prior is running.
- Each runtime must retain its repository commit/status, complete resolved
  package list, checkpoint sizes/hashes, command, logs, and a real smoke result
  before scene processing. The three neural runtimes must identify physical
  GPU 0 as an A40; no environment name alone is accepted as evidence.

### Locked scene006 generation

- Refuse pre-existing `depth`, `semantic`, `sky`, `flow`, or `colmap`
  directories and a pre-existing `colmap.ply`; partial or silently reused
  priors are not a reproduction.
- Run released depth inference at its default input size 518 for all 86 images.
- Run released segmentation serially from an isolated working directory:
  first `--text sky. --name sky`, then
  `--text car.bus.truck.van.human. --name semantic`, both with `--step 1` and
  physical GPU 0. They cannot run concurrently because the released script
  deletes its relative `./outputs` directory.
- Before point segmentation, preserve the DATA-001 `points3d.ply` with its
  hash. Run released `segment_pcd.py` once on GPU 0, retain the unsegmented
  backup, and record the new segmented hash.
- Run released Waymo CoTracker flow with `--downsample 1 --step 4` on physical
  GPU 0. Then run released COLMAP with `--cam 1` and its default CPU feature
  extraction/matching path.
- Record a stage exit code and wall time for every command. Only after all
  stages exit zero may the complete DATA-002 validator run. Generator failure
  is diagnosed from retained stage logs; no later stage may conceal it.

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
  `experiments.md`, `server.md`, `scripts/trust4d/`, `tests/`, and the
  source/code-grounding artifact under
  `research/ara/trust4d_teacher_reliability/`; any modified or deleted official
  file fails preflight. The added research artifact is documentation and
  CPU-testable reference mathematics only; it is neither imported by nor
  available on the AD-GS train/render path.
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

## EXP-002 — intervention reliability diagnostic

### Question and decision scope

Before changing AD-GS, test whether DGGT disagreement under equivalent temporal
observations identifies its own motion errors. This experiment may authorize a
teacher-loss implementation; it cannot establish a rendering improvement. Run
only after EXP-001 confirms the released output shapes and A40 memory margin.

### Locked scenes, windows, and leakage boundary

- Use Waymo `scene006`, `scene026`, and `scene090`, selected before results.
- In every scene use anchor indices `5`, `25`, `45`, and `65`. For anchor `a`,
  the original intervention is `[a,a+5,a+10,a+14]`, reverse is the exact
  reversed order, sparse is `[a,a+10,a+14]`, and interior-shifted is
  `[a,a+4,a+9,a+14]`. All indices are non-validation frames under the released
  every-fourth-frame split. No held-out RGB enters DGGT.
- Query selection uses only the released Grounded-SAM object prior at frame
  `a`. Apply the exact DGGT width-518 resize and center crop to the mask with
  nearest-neighbor sampling. Define five-pixel erosion as five iterations of a
  full `3 x 3` binary neighborhood with false padding (equivalently, every
  retained pixel has Chebyshev distance greater than five from the transformed
  mask boundary). Intersect the result with the centers of fixed `16 x 16`
  output cells: `x = 8 + 16j`, `y = 8 + 16i`, while each coordinate is in
  bounds. These erosion and grid-origin choices are project conventions because
  the source code does not specify a unique mask-query grid. Sort candidates by
  `(y,x)` and, when more than 128 remain, take the nearest integer ranks from
  `linspace(0, N-1, 128)` using NumPy round-to-nearest-even; this includes both
  endpoints and must yield 128 unique ranks. If fewer than 16 remain, the
  window is invalid and retained as a coverage failure.
- Before any evaluation label is opened, one CPU manifest builder must verify
  the exact three scene names, official frame counts, every-fourth validation
  flags, and that every intervention frame is train-only. It records hashes of
  `cameras.npz`, all referenced RGB files, the four anchor semantic masks, the
  complete crop transform, candidate/support counts, selected integer `(x,y)`
  queries, and the exact intervention indices. The output directory must be
  new and outside Git; its manifest and artifact hashes become immutable
  inputs to teacher export.
- The builder and six deterministic contract tests were implemented in commit
  `13d679d`. The full local suite passed `72` tests plus `2` subtests, and the
  clean-tree official-source audit confirmed that every released AD-GS file is
  byte-identical to base commit `9a208512`. This is implementation readiness,
  not an EXP-002 result; no server query manifest or teacher output exists yet.
- Waymo laser labels, object IDs, boxes, and velocities are evaluation-only.
  They may assign an already-selected query to an actor and define its target
  motion after every teacher output is frozen; they may not select queries,
  interventions, checkpoints, scenes, windows, or thresholds.

### Locked track and coordinate canonicalization

- Use the same DGGT commit, checkpoint, preprocessing, packages, and physical
  A40 as EXP-001. The primary 3D lift is bilinear sampling of the released
  `world_points`; depth unprojection is recorded only as a secondary diagnostic.
- DGGT's track head always anchors queries in sequence position zero. Original,
  sparse, and interior-shifted runs therefore receive the same frame-`a` query
  coordinates. The reverse run receives the original run's predicted endpoint
  coordinates at physical frame `a+14`; its returned displacement is reordered
  to physical chronology and sign-corrected. Directly reusing frame-`a` pixels
  as reverse queries is prohibited because it addresses different image points.
- Retain a query only when all four interventions are in bounds, finite, and
  have endpoint visibility at least `0.5`. Record rather than threshold DGGT
  track confidence, world-point confidence, and dynamic confidence.
- Decode every run's predicted OpenCV camera-from-world poses. For each frame,
  convert `cameras.npz` through the released AD-GS convention back to an
  explicitly tested OpenCV camera-from-world matrix, then form the world-frame
  rotation candidate
  `A_i = R_adgs_i^T R_dggt_i`; average candidates on SO(3) by SVD, then fit one
  positive scale and translation to predicted and AD-GS camera centers by
  least squares. Apply that Sim(3) to all lifted points. A run is invalid when
  its median camera rotation residual exceeds `10 degrees`, center-alignment
  RMSE exceeds `0.5 m`, fitted scale is nonpositive/nonfinite, or any aligned
  point is nonfinite. Invalid runs are coverage failures, never silently
  removed from the denominator.

### Locked evaluation target and analysis unit

- Project Waymo laser boxes into the anchor front camera only after inference.
  A selected query is evaluation-matched when it lies in exactly one projected
  actor box whose stable ID is present at both endpoints. Ambiguous queries are
  unmatched, not reassigned.
- Mark actors dynamic using the DGGT appendix thresholds: planar speed above
  `0.5 m/s` for vehicles/cyclists and above `0.2 m/s` for pedestrians. Convert
  box centers through frame ego poses into the same first-frame metric world.
- The primary independent unit is one actor-window. For every intervention,
  aggregate its matched query displacement by a coordinate-wise median. The
  target is the actor box-center displacement over the same physical horizon.
  Query-level results are exploratory and cannot decide the gate.
- Require at least 30 valid actor-window units, at least five per scene, and at
  least 50% of preselected windows valid. Otherwise EXP-002 is inconclusive;
  do not weaken these support requirements after seeing data.

### Locked metrics

For actor-window `j`, let `d_j^k` be the displacement from intervention `k`,
`d_j^gt` the Waymo displacement, and `d_j^O` the original prediction.

- Primary error: `e_j = ||d_j^O - d_j^gt||_2` in meters.
- Scalar disagreement:
  `u_j = median_{k<l} ||d_j^k - d_j^l||_2` in meters.
- Directional disagreement: covariance of the four `d_j^k` vectors and its
  largest-eigenvalue eigenvector `v_j`. Directional error capture is
  `r_j = (v_j^T(d_j^O-d_j^gt))^2 / (||d_j^O-d_j^gt||_2^2 + 1e-12)`.
- Report Spearman correlation between `u` and `e`; AUROC for `u` predicting the
  top quartile of `e`; median error in four equal-count `u` bins; and median
  `r`. Compare AUROC against each preregistered scalar baseline
  `1-min_endpoint_track_conf`, `-min_endpoint_world_point_conf`, and
  `1-min_endpoint_sigmoid(dynamic_conf)` after actor-level median aggregation.
- Use 10,000 seed-zero bootstrap resamples over stable `(scene, actor_id)`
  blocks for 95% intervals. Do not treat multiple pixels or windows of one
  actor as independent.

### Go / no-go gate

- **Anisotropic GO:** support gate passes; Spearman `rho >= 0.35` with lower
  95% bound above zero; disagreement AUROC is at least `0.70` and exceeds the
  best scalar-confidence AUROC by at least `0.03`; the top disagreement
  quartile has at least `1.5x` the median error of the bottom quartile; and
  median directional capture is at least `0.50` with lower 95% bound above
  the isotropic reference `1/3`.
- **Scalar-only pivot:** the correlation and AUROC gates pass but either the
  scalar-confidence margin or directional gate fails. Do not implement an
  anisotropic precision loss; test only the simpler scalar consensus control.
- **One audit allowed:** `0.20 <= rho < 0.35` or
  `0.60 <= AUROC < 0.70` triggers one source/coordinate/coverage audit with
  unchanged scenes, windows, queries, metrics, and thresholds. It is not a
  license to tune the experiment.
- **Stop:** `rho < 0.20`, AUROC below `0.60`, a reversed relationship, or a
  repeated borderline outcome after the audit stops Trust4D teacher-loss work.
  No AD-GS treatment training or alternative-scene search follows.

### Required evidence

Record raw and processed input hashes, query manifests before label matching,
all four teacher outputs, decoded/aligned cameras, Sim(3) residuals, label-match
coverage, actor-window rows, bootstrap indices, scalar baselines, final metrics,
plots, commands, environments, GPU snapshots, exit codes, and artifact hashes.
The report must keep invalid/unmatched counts and reasons; a high score on a
small silently filtered subset is a failed experiment.

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
