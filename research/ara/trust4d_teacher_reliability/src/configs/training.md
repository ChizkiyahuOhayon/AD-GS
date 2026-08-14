# Training and Evaluation Configuration

## DGGT source recipe

- Waymo split: 798 training scenes and 202 test scenes.
- Input timestamps: `0, 5, 10, 15`; targets span frames `0..19` from three cameras.
- Source training: eight NVIDIA H200 GPUs for about 24 hours and about 5,000 iterations.
- Per-scene optimization baselines in the source comparison use 5,000 iterations.
- Dynamic-mask supervision is derived from LiDAR 3D boxes, tracking identifiers, and category-specific speed thresholds.
- The source zero-shot protocol is separately reported on nuScenes and Argoverse; those results are not AD-GS treatment controls.

## DynamicVGGT source recipe

- Architecture scale: 12 motion-aware temporal-attention blocks; 1.4B parameters in total, with about 800M fine-tuned parameters.
- Stage 1: synthetic dynamic pretraining for 10 epochs, learning rate `1e-6`, warmup `0.5` epoch.
- Stage 2: joint training for 50 epochs, learning rate `5e-5`.
- Images are resized with maximum long side 518 pixels.
- Future offsets are `delta in {1,2,3}` and the effective batch is 18 images.
- Reported loss weights are `0.01, 0.1, 0.1, 0.01` for the auxiliary objectives described in the source.
- Training mixes synthetic motion, real video, point-map depth distillation, scene flow, and rendering supervision.

## Trust4D project protocol

- Host baseline: released AD-GS, unchanged, 60,000 iterations, every-fourth-frame validation split.
- Hardware: one A40; frozen DGGT export and AD-GS optimization execute serially.
- EXP-001 uses one four-frame front-camera clip and seed 0 for the runtime/contract probe.
- EXP-002 uses scenes 006, 026, and 090; anchors `5, 25, 45, 65`; four locked interventions per anchor.
- Query manifests are generated from train-only images and Grounded-SAM masks before evaluation labels are accessed.
- Statistical bootstrap seed is 0 with 10,000 actor-ID block resamples.
- Exact coverage, alignment, association, AUROC, calibration, direction, pivot, and stop thresholds are authoritative in root [`experiments.md`](../../../../../experiments.md).

No source training number is treated as a Trust4D result. All project outcomes remain pending until server artifacts are returned.
