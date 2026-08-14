# Implementation Heuristics

## H01: Transfer reverse queries through the original endpoint
- **Rationale**: DGGT anchors every track at sequence position zero; frame-anchor coordinates name a different image point in a reversed clip.
- **Sensitivity**: high
- **Bounds**: Reverse is valid only when the original endpoint is finite, in bounds, and visible.
- **Code ref**: [src/execution/intervention_reference.py]
- **Source**: DGGT `BaseTrackerPredictor.forward` and root EXP-002 protocol.

## H02: Fit rotation before metric scale
- **Rationale**: Camera centers in a driving clip may be nearly collinear, making center-only 3D rotation estimation unstable.
- **Sensitivity**: high
- **Bounds**: Project the mean rotation candidate to SO(3); require positive scale and locked pose residuals.
- **Code ref**: [src/execution/intervention_reference.py]
- **Source**: DGGT pose convention and Trust4D canonicalization design.

## H03: Use actor-window rather than query pixels as the unit
- **Rationale**: Dense pixels on one actor are strongly dependent and would create false statistical precision.
- **Sensitivity**: high
- **Bounds**: Aggregate by coordinate-wise median and block-resample stable actor IDs.
- **Code ref**: [src/execution/intervention_reference.py]
- **Source**: Root EXP-002 protocol.

## H04: Clamp future precision eigenvalues
- **Rationale**: With few interventions the covariance is low-sample and may be singular.
- **Sensitivity**: medium
- **Bounds**: No inverse covariance is used in EXP-002; any later loss must add a fixed variance floor and clamp eigen-precision.
- **Code ref**: [src/execution/intervention_reference.py]
- **Source**: Trust4D-GS research plan, anisotropic precision section.

## H05: Record confidence without prefiltering
- **Rationale**: Filtering on teacher confidence would make its comparison with disagreement circular and hide coverage failures.
- **Sensitivity**: high
- **Bounds**: Only finite, in-bounds, and fixed visibility gates remove queries; scalar confidences remain comparison scores.
- **Code ref**: [src/execution/intervention_reference.py]
- **Source**: Root EXP-002 leakage and metric protocol.
