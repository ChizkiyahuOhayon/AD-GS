# G2 geometry evaluator lock

This operational definition is locked before either 5k arm is run.

## Sparse held-out dynamic LiDAR

- Scene and camera: Waymo scene090, front camera.
- Frames: the official AD-GS validation flags, corresponding to relative frame
  IDs 4, 8, ..., 100.
- LiDAR: first return, projected with the same pinhole calibration convention
  used by AD-GS preprocessing. If multiple points round to one pixel, retain
  the nearest positive camera-z depth.
- Dynamic support: ground-truth semantic-mask pixels whose instance ID is in
  `scene090-stable-actors.json`. The IDs and stability thresholds were fixed
  from training-only initialization LiDAR before G1.
- Rendered depth: `opacity / inverse_depth`, because the AD-GS rasterizer
  alpha-composites inverse depth without opacity normalization.
- Paired validity: finite positive ground-truth and predictions, with opacity
  at least 0.1 in both arms. Exactly this common pixel set and actor set is
  used for both arms.
- Support gate: at least 100 common pixels from at least 5 stable actors.
- Primary error: mean absolute relative depth error (AbsRel). The locked G2
  improvement is `(baseline - treatment) / baseline >= 0.10`.

## Actor contact residual

- Evaluate the same stable actor IDs at the same held-out times in both arms.
- An actor's lower support is the fixed 5% weighted quantile of Gaussian
  centre-z minus two rotation-aware vertical standard deviations.
- The target is the linearly interpolated training-only oracle height track.
- Baseline uses its rendered (unprojected) actor support; treatment uses its
  rendered support after the locked hard contact projection.
- Primary error is mean absolute residual over the common actor-time pairs.
  Both arms must report the same pair count and actor IDs. The locked G2
  improvement is at least 50%.

No metric definition, opacity threshold, support threshold, actor list, or
frame list may change after a 5k arm starts.
