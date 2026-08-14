# Architecture

```text
train-only RGB + fixed object prior
              |
              v
      Query Manifest Builder
              |
      +-------+-------+----------------+
      |               |                |
  original         sparse        interior-shifted
      |                                |
      +---- endpoint query ----> reverse
              |
              v
  frozen DGGT: poses + 2D tracks + world points + confidences
              |
              v
     Metric Sim(3) Canonicalizer
              |
              v
   Actor Aggregator (labels evaluation-only)
              |
              v
 disagreement magnitude + covariance direction
              |
        EXP-002 decision gate
              |
      stop / scalar / anisotropic
```

## Query Manifest Builder
- **Purpose**: Produce deterministic train-only queries before evaluation labels are opened.
- **Inputs**: AD-GS split metadata, images, Grounded-SAM object masks.
- **Outputs**: Hashed frame and query manifests.
- **Interactions**: Feeds identical anchor queries to non-reverse interventions.
- **Key design choices**: Exact DGGT crop transform, erosion, fixed grid, and deterministic rank subsampling.

## Frozen DGGT Teacher
- **Purpose**: Export dense geometry, predicted cameras, optional 2D tracks, and scalar confidences.
- **Inputs**: One intervention clip and query pixels.
- **Outputs**: `pose_enc`, `world_points`, `track`, `vis`, confidence tensors.
- **Interactions**: The original endpoint initializes the reverse query.
- **Key design choices**: No fine-tuning, no held-out RGB, serial A40 execution.

## Metric Sim3 Canonicalizer
- **Purpose**: Remove each invocation's coordinate gauge.
- **Inputs**: Predicted and AD-GS camera-from-world poses plus lifted points.
- **Outputs**: Metric points and pose-fit diagnostics.
- **Interactions**: Rejects invalid runs before disagreement computation.
- **Key design choices**: Rotation first, then positive scale and translation; explicit residual gates.

## Actor Aggregator
- **Purpose**: Avoid treating many pixels from one actor as independent evidence.
- **Inputs**: Frozen query outputs and evaluation-only Waymo tracked boxes.
- **Outputs**: One displacement vector per actor-window and intervention.
- **Interactions**: Feeds scalar and directional statistics.
- **Key design choices**: Unique projected-box match and coordinate-wise median.

## Reliability Decision Gate
- **Purpose**: Decide whether any AD-GS teacher loss deserves implementation.
- **Inputs**: Coverage, association, scalar-baseline, and directional metrics.
- **Outputs**: Stop, scalar-only pivot, one audit, or anisotropic go.
- **Interactions**: Controls access to treatment experiment E04.
- **Key design choices**: Thresholds are locked in root `experiments.md` before GPU results.
