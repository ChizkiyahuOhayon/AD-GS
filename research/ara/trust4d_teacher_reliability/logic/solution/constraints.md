# Constraints and Limitations

## Source boundaries

- DGGT uses Waymo LiDAR-box-derived dynamic supervision; describing the teacher as annotation-free would be false.
- DynamicVGGT uses synthetic geometry, pretrained geometry, depth distillation, and scene-flow supervision. Only its temporal representation idea transfers to v1.
- DynamicVGGT has no validated local code/checkpoint contract in this project and is not a runtime dependency.

## Coordinate and correspondence boundaries

- DGGT pose encodings are OpenCV camera-from-world transforms.
- Raw coordinates from separate invocations are incomparable until each gauge is aligned.
- The track head always samples query features from sequence position zero. Reverse queries must be transferred to the physical endpoint.
- Sim(3) camera alignment can fail when pose predictions are poor; such failures are coverage evidence, not removable outliers.
- Four intervention vectors cannot establish a calibrated covariance distribution.

## Evaluation boundaries

- DGGT Table 1 and AD-GS main-table results use incompatible tasks and cannot establish superiority.
- Waymo labels enter only after teacher inputs, queries, outputs, and thresholds are frozen.
- Box-center displacement evaluates actor translation, not articulated or per-surface scene flow.
- Actor-window rows from one tracked identity remain dependent and are block-resampled together.

## Hardware boundaries

- DGGT and AD-GS never share GPU residency.
- The new server result must identify physical GPU 0 as an A40 and retain the preregistered allocator margin.
- DynamicVGGT's reported 1.4B model and training recipe do not imply feasible one-A40 training.

## Appendix coverage

- DGGT Appendix A.1 is represented in the leakage boundary and dynamic-speed thresholds.
- DGGT Appendix A.2 is represented in protocol comparability and source environment/config files.
- DGGT Appendix B.1 is represented by raw Table 2 and cross-domain constraints.
- DGGT Appendix B.2 and Figures 7-8 are qualitative; they support no numerical Trust4D claim.
