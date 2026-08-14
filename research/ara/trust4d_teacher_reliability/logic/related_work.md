# Typed Related-Work Graph

## RW01: Xu et al., AD-GS, 2025
- **DOI**: ICCV 2025; paper title “AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving”
- **Type**: baseline
- **Delta**:
  - What changed: Trust4D proposes an offline reliability-aware motion prior for AD-GS object Gaussians.
  - Why: AD-GS has scene-specific motion capacity but relies on noisy pseudo priors.
- **Claims affected**: C02, C04, C05
- **Adopted elements**: Released split, renderer, losses, B-spline motion, and same-budget control.

## RW02: Chen et al., DGGT, 2025
- **DOI**: arXiv:2512.03004
- **Type**: imports
- **Delta**:
  - What changed: Trust4D uses DGGT as a frozen diagnostic/export teacher rather than a feed-forward renderer.
  - Why: DGGT exposes dense geometry, dynamic logits, poses, and query-conditioned tracking.
- **Claims affected**: C01, C03, C04, C05
- **Adopted elements**: Pinned checkpoint, track head, world points, and confidences.

## RW03: He et al., DynamicVGGT, 2026
- **DOI**: arXiv:2603.08254
- **Type**: extends
- **Delta**:
  - What changed: Trust4D adopts multi-horizon/future-point consistency as a later loss hypothesis, not the 1.4B architecture.
  - Why: DynamicVGGT separates current/future point maps and explicit Gaussian motion.
- **Claims affected**: C04
- **Adopted elements**: Future-point concept and complementary implicit/explicit motion levels.

## RW04: Wang et al., VGGT, 2025
- **DOI**: CVPR 2025
- **Type**: imports
- **Delta**:
  - What changed: DGGT extends VGGT with Gaussian, dynamic, lifespan, and motion heads; Trust4D consumes the released extension.
  - Why: VGGT provides the static geometry and pose backbone.
- **Claims affected**: C01, C05
- **Adopted elements**: Alternating-attention geometry backbone and pose encoding.

## RW05: Yang et al., STORM, 2025
- **DOI**: ICLR 2025
- **Type**: baseline
- **Delta**:
  - What changed: DGGT predicts pose and richer trajectories; DynamicVGGT adds future points and motion-aware temporal attention.
  - Why: STORM is pose-dependent and uses a different feed-forward task.
- **Claims affected**: C02
- **Adopted elements**: Motion and feed-forward evaluation metrics.

## RW06: Zhang et al., TAPIP3D, 2025
- **DOI**: arXiv:2504.14717
- **Type**: imports
- **Delta**:
  - What changed: DGGT initializes its motion head from persistent 3D tracking work; released interpolation separately loads TAPIP3D.
  - Why: It supplies correspondence machinery, but it is not the default DGGT forward output.
- **Claims affected**: C01
- **Adopted elements**: Tracking architecture lineage only.

## RW07: Karaev et al., CoTracker3, 2024
- **DOI**: not specified in provided inputs
- **Type**: bounds
- **Delta**:
  - What changed: AD-GS uses CoTracker3 pseudo flow; EXP-002 evaluates DGGT independently rather than treating CoTracker as ground truth.
  - Why: A second learned tracker cannot serve as metric motion truth.
- **Claims affected**: C03
- **Adopted elements**: Existing host flow prior only.

## RW08: Waymo Open Dataset, 2020
- **DOI**: CVPR 2020
- **Type**: imports
- **Delta**:
  - What changed: Trust4D separates train-only RGB inference from evaluation-only tracked boxes.
  - Why: The labels enable falsification without leaking into the teacher.
- **Claims affected**: C03, C04
- **Adopted elements**: Camera calibration, ego poses, tracked boxes, categories, and velocities.

## Remaining citation footprint

The complete bibliographies remain in the two source PDFs. Background groups
covered there include optimization-based dynamic NeRF/3DGS, feed-forward
Gaussian reconstruction, pose-free geometry, scene flow, diffusion refinement,
segmentation, driving simulation, and cross-domain driving datasets. They do
not create additional executable dependencies for EXP-001/EXP-002; any later
novelty claim requires a dedicated, current literature search rather than
treating this compilation as a scoop check.
