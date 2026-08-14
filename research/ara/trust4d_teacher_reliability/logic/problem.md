# Problem Specification

## Observations

### O1: DGGT motion is strong but annotation-assisted
- **Statement**: DGGT reports 3D motion EPE3D `0.183 m`, while its appendix says dynamic masks are constructed from Waymo LiDAR 3D boxes and tracking identifiers.
- **Evidence**: DGGT Table 5 and Appendix A.1.
- **Implication**: A frozen DGGT prior is useful evidence, but it is not an annotation-free source model.

### O2: The released default forward pass does not produce tracks
- **Statement**: `VGGT.forward(images, query_points=None)` returns dense geometry and confidence outputs; `track`, `vis`, and `conf` appear only when query points are supplied.
- **Evidence**: `DGGT/dggt/models/vggt.py` at commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`.
- **Implication**: A Trust4D adapter must define query selection and 3D lifting explicitly.

### O3: DGGT and AD-GS evaluate different tasks
- **Statement**: DGGT's main Waymo task uses four input times, three cameras, 20-frame output, 202 scenes, and 5,000-iteration per-scene baselines; AD-GS uses a front-camera, every-fourth-frame held-out split and 60,000 optimization iterations on eight scenes.
- **Evidence**: DGGT Appendix A.2 and the released AD-GS protocol.
- **Implication**: Cross-paper PSNR values cannot be used as a treatment comparison.

### O4: DynamicVGGT improves point maps more than rendering
- **Statement**: DynamicVGGT reports improved point-map metrics over VGGT/StreamVGGT, while its image-only Waymo full-frame NVS result remains below camera-supervised feed-forward baselines in its Table 2.
- **Evidence**: DynamicVGGT Tables 1, 2, and 4.
- **Implication**: Its transferable value is temporal geometry design, not a drop-in rendering baseline.

### O5: Both sources expose motion failure modes
- **Statement**: DGGT names inaccurate dynamic masks and heavily occluded tracking as limitations; DynamicVGGT reports sparse real LiDAR supervision degrades smooth dense geometry and therefore distills its point-map depth.
- **Evidence**: DGGT Section 5 and DynamicVGGT Sections 3.5 and Figure 4.
- **Implication**: Direct teacher distillation can transmit structured errors.

## Gaps

### G1: No validated reliability signal
- **Statement**: Neither source establishes that its scalar confidence predicts actor-level metric motion error under the AD-GS data protocol.
- **Caused by**: O1, O2, O3, O5.
- **Existing attempts**: Per-output confidence and direct motion supervision.
- **Why they fail**: Confidence can be miscalibrated under domain shift and does not encode directional uncertainty.

### G2: Temporal interventions change the model gauge
- **Statement**: Reversing or shifting a DGGT clip changes sequence position zero, which is both the track query frame and the predicted world reference.
- **Caused by**: O2 and released tracker/camera code.
- **Existing attempts**: Direct comparison of raw forward/reverse coordinates.
- **Why they fail**: Raw coordinates and query pixels refer to different frames and possibly different physical points.

### G3: Future-point inspiration is not an available dependency
- **Statement**: The local DynamicVGGT paper describes future-point prediction, but the current project has no validated public implementation/checkpoint contract for it.
- **Caused by**: O4.
- **Existing attempts**: Treat the paper module as a ready teacher.
- **Why they fail**: Paper architecture alone cannot guarantee reproducible tensors or A40 feasibility.

## Key Insight

- **Insight**: Treat temporal interventions as a pre-training diagnostic: compare metric-aligned predictions for the same actor motion, and authorize anisotropic supervision only if disagreement predicts both error magnitude and error direction better than released scalar confidences.
- **Derived from**: O2, O3, O5, G1, and G2.
- **Enables**: A cheap go/no-go test before any AD-GS treatment run.

## Assumptions

- A1: Known training-camera poses may align teacher outputs because AD-GS already consumes those poses; held-out RGB and evaluation labels may not enter teacher inference.
- A2: Waymo tracked boxes provide an evaluation-only actor displacement target.
- A3: A four-intervention covariance is only a local diagnostic, not a calibrated posterior covariance.
- A4: Offline teacher export and AD-GS optimization never occupy the A40 simultaneously.
