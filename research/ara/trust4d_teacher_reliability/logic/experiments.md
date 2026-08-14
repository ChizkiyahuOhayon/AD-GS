# Declarative Experiments

Exact numerical outcomes belong in evidence files. Expected outcomes here are
directional only.

## E01: Released source contract audit
- **Verifies**: C01
- **Setup**:
  - Model: DGGT at the pinned released commit.
  - Hardware: CPU source inspection.
  - Dataset: none.
  - System: clean upstream checkout.
- **Procedure**:
  1. Trace `VGGT.forward`, `TrackHead`, pose decoding, dense geometry outputs, and released inference.
  2. Record conditional outputs, tensor semantics, coordinate conventions, and implicit dependencies.
- **Metrics**: Resolved output keys, dimensions documented in source, and code-path reachability.
- **Expected outcome**: The audit distinguishes 2D tracks from metric 3D motion and identifies every required adapter step.
- **Baselines**: Paper-only interpretation.
- **Dependencies**: none

## E02: Protocol comparability audit
- **Verifies**: C02
- **Setup**:
  - Model: DGGT, DynamicVGGT, and AD-GS.
  - Hardware: none.
  - Dataset: the released Waymo protocols.
  - System: papers, appendices, and source scripts.
- **Procedure**:
  1. Extract scenes, cameras, input/target frames, supervision, iteration budgets, and metrics.
  2. Mark a comparison valid only when every field matches.
- **Metrics**: Number and type of matched or mismatched protocol fields.
- **Expected outcome**: Headline cross-paper rendering values are rejected as paired treatment evidence.
- **Baselines**: Naive headline comparison.
- **Dependencies**: none

## E03: Intervention reliability diagnostic
- **Verifies**: C03, C04
- **Setup**:
  - Model: frozen DGGT and no AD-GS treatment.
  - Hardware: one A40; teacher invocations run serially.
  - Dataset: three preregistered Waymo scenes and train-only windows.
  - System: exact protocol in root `experiments.md`, EXP-002.
- **Procedure**:
  1. Freeze query manifests before accessing Waymo evaluation labels.
  2. Run original, reverse, sparse, and interior-shifted clips.
  3. Transfer reverse queries to the original endpoint and align each run to AD-GS metric world coordinates.
  4. Aggregate actor-window motion and compare disagreement with evaluation-only box motion.
- **Metrics**: Coverage, pose residuals, Spearman correlation, error AUROC, disagreement-bin calibration, confidence-baseline AUROC, and directional error capture.
- **Expected outcome**: Reliable association and directional capture authorize anisotropic trust; partial association authorizes only a scalar pivot; weak association stops teacher-loss work.
- **Baselines**: Released track, world-point, and dynamic scalar confidences.
- **Dependencies**: E01, E05

## E04: Same-budget teacher-loss controls
- **Verifies**: C04
- **Setup**:
  - Model: AD-GS control, direct DGGT, scalar consensus, and anisotropic consensus.
  - Hardware: one A40, never shared with DGGT.
  - Dataset: same scenes, seeds, priors, and iteration budget.
  - System: cached teacher outputs only.
- **Procedure**:
  1. Run the unchanged host control.
  2. Run direct, scalar, and anisotropic treatments with identical non-teacher settings.
  3. Compare standard rendering, dynamic-region rendering, motion, cost, and failure coverage.
- **Metrics**: Paired PSNR, SSIM, LPIPS, actor-motion error, wall time, peak memory, and prior coverage.
- **Expected outcome**: Anisotropic trust improves hard motion without degrading the matched host control and outperforms simpler teacher controls.
- **Baselines**: AD-GS, direct distillation, scalar confidence, scalar disagreement.
- **Dependencies**: E03

## E05: Single-A40 DGGT contract probe
- **Verifies**: C05
- **Setup**:
  - Model: pinned one-view DGGT checkpoint.
  - Hardware: physical GPU 0, NVIDIA A40.
  - Dataset: four preregistered front-camera training images.
  - System: pinned source and Conda environment.
- **Procedure**:
  1. Verify source, checkpoint, runtime packages, and input hashes.
  2. Run warm-up and measured inference without query points.
  3. Record all tensors, finite fractions, elapsed time, and allocator peaks.
- **Metrics**: Required output contract, finite values, sequence dimensions, and reserved-memory margin.
- **Expected outcome**: The teacher fits the offline-export safety contract or fails before any treatment work.
- **Baselines**: none
- **Dependencies**: E01
