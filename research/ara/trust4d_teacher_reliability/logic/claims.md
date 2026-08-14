# Claims

## C01: Released DGGT is not a ready-made 3D-track exporter
- **Statement**: At the pinned release, DGGT returns 2D tracks only when explicit query pixels are supplied, so metric 3D trajectories require an adapter and dense-point lifting.
- **Status**: supported
- **Falsification criteria**: A clean pinned forward pass without query points returns documented metric 3D trajectories with stable identities.
- **Proof**: [E01]
- **Evidence basis**: The released `VGGT.forward` conditionally adds only `track`, `vis`, and `conf`; the tracker output is 2D image coordinates.
- **Interpretation**: Trust4D must test its adapter rather than cite the paper's phrase “3D tracking” as an implementation contract.
- **Dependencies**: none
- **Tags**: DGGT, code-contract, tracking

## C02: Cross-paper rendering numbers are not treatment evidence
- **Statement**: DGGT, DynamicVGGT, and AD-GS use different inputs, splits, supervision, iteration budgets, and evaluators, so their headline PSNR values are not directly comparable.
- **Status**: supported
- **Falsification criteria**: The released protocols are shown to use identical scenes, views, target frames, masks, budgets, and evaluator implementations.
- **Proof**: [E02]
- **Evidence basis**: DGGT Appendix A.2, DynamicVGGT Sections 4.1-4.4, and the released AD-GS scripts specify incompatible tasks.
- **Interpretation**: A paper claim must use same-budget paired AD-GS controls.
- **Dependencies**: none
- **Tags**: evaluation, comparability, protocol

## C03: Intervention disagreement predicts teacher error
- **Statement**: After same-point query transfer and metric Sim(3) alignment, actor-level temporal-intervention disagreement positively predicts original DGGT actor-motion error.
- **Status**: hypothesis
- **Falsification criteria**: The locked EXP-002 support gate fails, or disagreement has weak, null, or reversed association with evaluation-only actor error.
- **Proof**: [E03]
- **Evidence basis**: No source result establishes this; it is the project’s preregistered causal dependency.
- **Interpretation**: Passing would justify a reliability-aware teacher prior, not yet a rendering gain.
- **Dependencies**: C01
- **Tags**: intervention, reliability, motion-error

## C04: Directional disagreement supports anisotropic trust
- **Statement**: The leading eigenvector of intervention displacement covariance captures more teacher-error energy than an isotropic direction and adds signal beyond scalar confidence.
- **Status**: hypothesis
- **Falsification criteria**: EXP-002 passes scalar association but fails its preregistered directional or scalar-margin gates.
- **Proof**: [E03, E04]
- **Evidence basis**: The papers provide motion and confidence outputs but no directional reliability experiment.
- **Interpretation**: Failure requires a scalar-only pivot rather than an anisotropic loss.
- **Dependencies**: C03
- **Tags**: covariance, anisotropy, trust-region

## C05: Offline DGGT export fits the single-A40 project contract
- **Statement**: The pinned one-view DGGT checkpoint can export finite four-frame outputs on one A40 while retaining the locked safety margin.
- **Status**: hypothesis
- **Falsification criteria**: EXP-001 fails source, tensor, finite-value, GPU-identity, or reserved-memory gates.
- **Proof**: [E05]
- **Evidence basis**: The local code is prepared, but no new-server GPU result exists.
- **Interpretation**: Passing establishes feasibility only, not scientific utility.
- **Dependencies**: C01
- **Tags**: A40, memory, feasibility
