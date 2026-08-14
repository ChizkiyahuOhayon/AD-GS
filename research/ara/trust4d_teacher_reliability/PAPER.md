---
title: "Teacher Reliability for Trust4D-GS: A Source- and Code-Grounded Research Artifact"
authors:
  - "Xiaoxue Chen et al. (DGGT source)"
  - "Zhuolin He et al. (DynamicVGGT source)"
  - "Trust4D-GS project synthesis"
year: 2026
venue: "Research synthesis from arXiv papers and released repositories"
doi: "arXiv:2512.03004; arXiv:2603.08254"
ara_version: "1.0"
domain: "dynamic autonomous-driving reconstruction"
keywords: [4D reconstruction, Gaussian splatting, motion prior, teacher reliability, temporal intervention, uncertainty, autonomous driving]
claims_summary:
  - "Released DGGT exposes dense geometry and optional 2D tracking, but not ready-made metric 3D trajectories."
  - "DGGT and DynamicVGGT results are not directly comparable with the released AD-GS per-scene protocol."
  - "Temporal-intervention disagreement is a falsifiable candidate reliability signal, not an established result."
abstract: "This artifact compiles DGGT, DynamicVGGT, their released code contracts, and the Trust4D-GS proposal into an executable research map. It separates reported facts from project hypotheses, identifies coordinate and query-reference hazards, and preregisters the minimum diagnostic required before modifying AD-GS."
---

# Teacher Reliability for Trust4D-GS

## Overview

DGGT supplies a frozen image-only geometry model with an optional track head;
DynamicVGGT motivates future-point and multi-horizon motion modeling. Neither
source proves that its motion can safely supervise a scene-specific AD-GS model.
The project therefore tests reliability under temporal interventions before any
teacher loss is implemented.

## Layer Index

### Cognitive Layer (`logic/`)

| File | Description |
|---|---|
| [problem.md](logic/problem.md) | Observations, gaps, assumptions, and key insight |
| [claims.md](logic/claims.md) | Five falsifiable source/project claims |
| [concepts.md](logic/concepts.md) | Formal definitions and boundary conditions |
| [experiments.md](logic/experiments.md) | Source audit, contract probe, reliability test, and downstream control |
| [solution/architecture.md](logic/solution/architecture.md) | Offline-teacher component graph |
| [solution/algorithm.md](logic/solution/algorithm.md) | Canonicalization and disagreement mathematics |
| [solution/constraints.md](logic/solution/constraints.md) | Leakage, hardware, protocol, and model limitations |
| [solution/heuristics.md](logic/solution/heuristics.md) | Bounded implementation rules |
| [related_work.md](logic/related_work.md) | Typed dependency graph |

### Physical Layer (`src/`)

| File | Description | Claims |
|---|---|---|
| [configs/training.md](src/configs/training.md) | Exact source and project training/evaluation settings | C02, C05 |
| [configs/model.md](src/configs/model.md) | DGGT, DynamicVGGT, and intervention model contracts | C01, C03 |
| [execution/intervention_reference.py](src/execution/intervention_reference.py) | Minimal Sim(3) and disagreement reference | C03, C04 |
| [environment.md](src/environment.md) | Source and project environments | C05 |

### Exploration Graph (`trace/`)

| File | Description |
|---|---|
| [exploration_tree.yaml](trace/exploration_tree.yaml) | Eleven-node source-grounded research DAG |

### Evidence (`evidence/`)

| File | Description |
|---|---|
| [README.md](evidence/README.md) | Index of paper tables, code audit, and derived figure data |
