# Model Contracts

## Released DGGT

The pinned model consumes a batch of RGB sequences. Its dense path exposes pose encodings, depth, depth confidence, world points, world-point confidence, dynamic scores, Gaussian parameters, and lifespan-related outputs. The released inference path normally reconstructs points by unprojecting predicted depth; EXP-002 therefore treats dense `world_points` as the primary lifting source and depth unprojection as a separately reported secondary adapter.

The optional track head is query-conditioned. Query points name 2D pixels in sequence position zero, and the output `track` remains 2D image coordinates with visibility and confidence. Metric 3D motion is not a direct return value: the Trust4D adapter must crop-transform queries, lift corresponding positions with dense geometry, and align each invocation's gauge.

Pinned source commit: `a3276d2bbe4cbb03bcc117830b1836110a27adeb`.

## DynamicVGGT

The paper augments VGGT with motion-aware temporal attention, current and future point-map prediction, an explicit dynamic Gaussian head, and multi-horizon future offsets. The reported model is not a Trust4D runtime dependency. It contributes a representation hypothesis—future-point consistency—only after the cheaper DGGT reliability gate succeeds.

## Trust4D intervention adapter

- Inputs: one locked four-frame intervention, deterministic query pixels, and known train-camera extrinsics.
- Original/sparse/interior-shifted runs use anchor-frame queries; reverse uses valid predicted original-endpoint queries.
- Per-run outputs: predicted camera extrinsics, dense points, 2D tracks, visibility, and released scalar confidences.
- Canonical output: metric actor displacement in the AD-GS world gauge plus alignment residuals.
- Diagnostic output: robust scalar disagreement, covariance eigenvectors/eigenvalues, and directional error capture.
- No learned parameter is introduced before EXP-002 passes.

The executable reference in [`intervention_reference.py`](../execution/intervention_reference.py) implements only the gauge and diagnostic mathematics. It intentionally does not duplicate DGGT inference, query construction, label matching, or AD-GS training.
