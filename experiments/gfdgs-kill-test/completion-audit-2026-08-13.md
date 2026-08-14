# GF-DGS completion audit — 2026-08-13

This audit maps the stated research objective to evidence that is available
locally. It is intentionally stricter than a progress summary: code or results
that exist only on an unreachable server are not counted as verified, and a
unit-tested component is not counted as an integrated paper method.

Status vocabulary:

- **PROVEN** — supported by locally archived code and/or raw evidence.
- **PARTIAL** — some required cases or modules are proven, but the requirement
  is not complete.
- **MISSING** — no qualifying evidence exists.
- **BLOCKED** — the next locked check exists but currently needs the remote GPU.

## Objective-level verdict

The full objective is **not complete**. The repository contains a defensible
theory probe, reusable contact/rigidity components, a fixed-ID oracle treatment,
and a passed 1k engineering gate. It does not yet contain the annotation-free
GF-DGS method, a completed geometry decision, a complete AD-GS main-table
reproduction, or multi-dataset/cross-seed evidence. The present result is a
budgeted feasibility study, not a CVPR-ready method or result.

## Requirement-to-evidence map

| Requirement | Status | Qualifying evidence | Missing evidence |
|---|---|---|---|
| Correct released AD-GS implementation and Waymo preprocessing | **PROVEN** | Locked release/dependency manifests in `analysis/manifests/`; all eight Waymo scenes were reported with complete flow and finite COLMAP outputs in `../experiment_logs/ADGS-GPU-260730-001.md` | None for the eight-scene preprocessing contract |
| Official Waymo 60k baseline table | **PARTIAL** | Locally recorded completed results for scene006, scene026, and scene090; scene090 is PSNR 30.7210, SSIM 0.91145, LPIPS(VGG) 0.24252 | Five of eight scenes. The interrupted scene105 half-run has no checkpoint and is excluded |
| KITTI baseline reproduction | **MISSING** | None | Formal released-protocol runs and metrics |
| nuScenes baseline reproduction | **MISSING** | None | Formal released-protocol runs and metrics |
| PandaSet evaluation | **MISSING** | None | Data contract, baseline, and treatment runs |
| Gauge claim separated into projection, rendering, and host-parameterization statements | **PROVEN** for the scoped probes | `analysis/gauge_orbit.py` and `analysis/results/gauge_orbit_synthetic.json` prove the projected-moment orbit and record visibility/ordering and multi-camera counterexamples; `analysis/basis_closure.py` records the host-basis closure condition | A paper-level theorem and experiments over all claimed host parameterizations |
| Real-camera host-basis audit | **PARTIAL** | Archived `basis-closure-006-026-090.json`; scene006/026/090 training RMS is 3.59/7.54/2.55 mm | Required broader host/parameterization study and formal cross-seed instability analysis |
| Actor-rigid and contact-tie primitives | **PROVEN** as isolated components | `models/actor_rigid.py`, `models/contact_tie.py`, and their unit tests, including hard extent tie and free-offset control behavior | Integration of these primitives into the released training pipeline |
| Annotation-free GF-DGS method | **MISSING** | None | Learned actor assignment, road chart/B-spline, integrated hard tie, release mechanism, and end-to-end training path |
| Fixed-ID oracle contact upper bound | **PROVEN** as a feasibility treatment only | Commit `4cda4e8`; `models/oracle_contact.py`; training integration in `scene/gaussian_model.py`; stable actor IDs locked in `scene090-stable-actors.json` | It uses initialization LiDAR/instance IDs and must not be presented as the final annotation-free method |
| G0 zero-GPU feasibility | **PROVEN** | Protocol `ad9e938`; scene090 camera-basis RMS 2.55 mm; 17 supported IDs, 15/17 without a >10 m consecutive jump; 13 strict actor IDs retained | None for the locked G0 gate |
| G1 paired 1k smoke | **PROVEN** | `g1-results.md` and raw logs: baseline/treatment wall time 261/319 s, peak memory 7455/7615 MiB, PSNR 26.48436/26.44652, LPIPS(VGG) 0.417574/0.418056 | None for G1; it establishes feasibility, not geometry |
| Locked held-out geometry data and evaluator | **PROVEN** as preparation | `g2-evaluator.md`, `evaluate_geometry.py`, 25 locally archived NPZ files, 46,903 unique sparse dynamic pixels, all 13 actors, manifest SHA-256 `8de40dae05bd426b18ce39da07772ea9d502261d1473ac4c4396ccd5e7879fb9` | Actual paired checkpoint evaluation output |
| Minimal geometric falsification | **BLOCKED** | The existing 1k checkpoints and evaluator are the only authorized next GPU check | Server endpoint refuses the SSH connection; no locally archived 1k geometry result exists |
| Locked 5k G2 decision | **MISSING** and currently **not authorized** | `budget-audit.md` accounts for at least 928 s already used and at most 1772 s remaining | Conservative paired 5k lower bound is 2192 s before geometry/LPIPS, exceeding the locked 0.75 GPU-hour budget by at least 420 s |
| Lane-shift 1/2/3 m evaluation | **MISSING** | None | Baseline/treatment renders and metrics at each shift |
| Free-offset matched-capacity control | **PARTIAL** | Its isolated absorption behavior is unit tested | End-to-end matched-capacity training and evaluation |
| Five-seed and ablation evidence | **MISSING** | None | Cross-seed statistics, actor/road/contact/release ablations, and uncertainty reporting |
| Complete multi-dataset comparison | **MISSING** | None | Full baseline/treatment tables on the planned Waymo, KITTI, nuScenes, and PandaSet suites |
| CVPR 2027 submission package | **MISSING** | A research plan and auditable preliminary evidence exist | Complete method, positive formal results, comparisons, ablations, paper, figures, and reproducibility package |

## Locked next decision

The only in-protocol next GPU action is to verify that the two scene090 1k
checkpoints still exist and run `evaluate_geometry.py` on them. This result is
exploratory because G2 was specified for 5k. Apply the locked thresholds without
tuning:

- dynamic-object sparse LiDAR AbsRel improves by at least 10%;
- actor contact residual improves by at least 50%;
- the common pixel and actor support gates remain satisfied.

If either improvement threshold or either support gate fails, stop the GF-DGS
idea under the current budget. If all pass, report only
**promising-but-unconfirmed**; do not start 5k, 60k, or another scene unless the
budget and protocol are explicitly expanded before execution.

## Reproducibility anchors

- Local verification on 2026-08-13: `python3 -m unittest discover -s tests -v`
  ran 31 tests, with 30 passing and the CUDA-only actor-ID lifecycle test
  skipped; there were no failures.
- Baseline protocol: `analysis/manifests/adgs_waymo_baseline_protocol.json`
- Kill protocol: `experiments/gfdgs-kill-test/protocol.md` at `ad9e938`
- Oracle implementation: `4cda4e8`
- G1 runner: `9a1ee6d`
- G1 result: `66a4e0a`
- Geometry evaluator lock/fix: `3ff7ce5`, `682e2fe`
- Budget stop: `1a26761`
- Raw local archive:
  `../experiment_logs/raw/ADGS-GPU-260730-001/gfdgs-kill-test/`
