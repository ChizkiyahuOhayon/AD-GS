# Derived DGGT runtime contract

**Source**: DGGT released code at commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`, especially `dggt/models/vggt.py`, the track predictor/head, camera utilities, and inference entry point.

**Extraction type**: derived_code_audit.

| Output or behavior | Condition | Released semantics | Trust4D consequence |
|---|---|---|---|
| `pose_enc` | Dense forward | Encoded OpenCV camera-from-world prediction | Decode and fit each invocation independently |
| depth + confidence | Dense forward | Per-view camera-space depth | Secondary 3D lifting adapter only |
| `world_points` + confidence | Dense forward | Dense predicted points in the model world gauge | Primary EXP-002 3D lifting source |
| dynamic output | Dense forward | Per-pixel learned dynamic evidence | Record as scalar baseline; do not prefilter |
| Gaussian/lifespan outputs | Dense forward | Feed-forward dynamic rendering representation | Not needed for EXP-001/EXP-002 |
| `track`, `vis`, `conf` | Only when query points are provided | 2D tracks, visibility, and confidence | Adapter must lift tracks; no direct metric trajectory |
| Query reference | Track predictor | Sequence position zero | Reverse must receive original predicted endpoint |
| Released inference geometry | Inference script | Depth is unprojected with decoded cameras | Report separately from direct `world_points` path |

This table is a code-derived contract, not a source-paper table and not proof that the proposed reliability signal works.
