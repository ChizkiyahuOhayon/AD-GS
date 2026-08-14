# Concepts

## Temporal intervention
- **Notation**: `k in {O,R,S,I}`
- **Definition**: A fixed transformation of the observed training-frame sequence that preserves one physical endpoint motion while altering temporal order or interior context.
- **Boundary conditions**: Interventions must contain no held-out RGB and must be restored to the same physical chronology before comparison.
- **Related concepts**: Same-point transfer, intervention displacement.

## Same-point transfer
- **Notation**: `q_end = track_O(q_anchor)`
- **Definition**: Initializing a reverse run at the original run's predicted physical endpoint because DGGT always anchors queries at sequence position zero.
- **Boundary conditions**: It measures cycle-style consistency and is invalid if the original endpoint is out of bounds or invisible.
- **Related concepts**: Temporal intervention, track visibility.

## Metric Sim(3) canonicalization
- **Notation**: `X_adgs = s A X_dggt + b`
- **Definition**: Rotation averaging from camera orientations followed by positive scale and translation fitting between predicted and known training-camera centers.
- **Boundary conditions**: Requires valid decoded poses, positive finite scale, and bounded pose residuals.
- **Related concepts**: Camera-from-world pose, gauge.

## Intervention displacement
- **Notation**: `d_j^k`
- **Definition**: The coordinate-wise median metric 3D displacement of matched queries for actor-window `j` under intervention `k`.
- **Boundary conditions**: All contributing queries must be finite, in bounds, and visible at both physical endpoints.
- **Related concepts**: Scalar disagreement, directional disagreement.

## Scalar disagreement
- **Notation**: `u_j = median_{k<l} ||d_j^k-d_j^l||_2`
- **Definition**: Robust pairwise spread of intervention displacement vectors in meters.
- **Boundary conditions**: Descriptive reliability score; not a probability or calibrated variance.
- **Related concepts**: Intervention covariance, teacher error.

## Directional disagreement
- **Notation**: `Sigma_j`, `v_j`
- **Definition**: Sample covariance of intervention displacements and the eigenvector associated with its largest eigenvalue.
- **Boundary conditions**: Four samples give a noisy local direction; eigenvalues must be clamped before any later precision construction.
- **Related concepts**: Directional error capture, anisotropic trust.

## Directional error capture
- **Notation**: `r_j = (v_j^T e_vec_j)^2/(||e_vec_j||_2^2+eps)`
- **Definition**: Fraction of squared original-teacher error aligned with the largest disagreement direction.
- **Boundary conditions**: Undefined in the zero-error limit except through the fixed epsilon; compared with isotropic reference `1/3` only in EXP-002.
- **Related concepts**: Directional disagreement, teacher error.

## Evaluation-only actor target
- **Notation**: `d_j^gt`
- **Definition**: Waymo tracked-box center displacement used only after teacher outputs and query manifests are frozen.
- **Boundary conditions**: It cannot affect input selection, model selection, thresholds, or training.
- **Related concepts**: Leakage boundary, actor-window.
