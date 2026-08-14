# Algorithm

## Camera-guided metric canonicalization

For predicted and AD-GS OpenCV camera-from-world rotations
`R_i^p, R_i^g`, each frame implies a world rotation

`A_i = (R_i^g)^T R_i^p`.

Project the arithmetic mean of `A_i` to SO(3) with an SVD, yielding `A`.
Let predicted and target camera centers be `C_i^p` and `C_i^g`. Fit

`s = argmin_{s>0,b} sum_i ||C_i^g - (s A C_i^p + b)||_2^2`

and `b = mean(C^g) - s A mean(C^p)`. Canonicalize every point with
`X^g = s A X^p + b`.

## Intervention statistics

For an actor-window with displacement vectors `{d^k}_{k=1}^K`, define

`u = median_{k<l} ||d^k-d^l||_2`.

Let `Sigma` be their unbiased sample covariance and `v` its largest-eigenvalue
eigenvector. For original error vector `e_vec = d^O-d^gt`, define

`r = (v^T e_vec)^2 / (||e_vec||_2^2 + 1e-12)`.

## Pseudocode

```text
for scene in preregistered_scenes:
  for window in preregistered_windows:
    queries = build_manifest(train_mask_at_anchor)
    O = run_dggt(original, queries)
    R = run_dggt(reverse, endpoint_queries(O))
    S = run_dggt(sparse, queries)
    I = run_dggt(interior_shifted, queries)

    for run in [O, R, S, I]:
      transform, residuals = fit_metric_sim3(run.poses, adgs_training_poses)
      reject_window_if_alignment_gate_fails(residuals)
      lift_and_transform_tracks(run, transform)

    freeze_teacher_artifacts_and_query_manifest()
    matched = evaluation_only_actor_match(waymo_boxes)
    rows += robust_actor_displacements(matched)

require_coverage(rows)
metrics = block_bootstrap(rows, group=(scene, actor_id), seed=0)
decision = apply_preregistered_gate(metrics)
```

## Complexity

Teacher inference dominates. With `K` interventions, `S` frames, `Q` queries,
and model cost `F(S,Q)`, export costs `O(K F(S,Q))`. Canonicalization is
`O(KS + KQ)`. Actor aggregation is `O(KQ log Q)` because of medians. The
`3 x 3` eigendecomposition is constant per actor-window. Cached exports keep
AD-GS training cost independent of DGGT memory.
