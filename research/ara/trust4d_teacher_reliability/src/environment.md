# Environment

## Source environments

- AD-GS official environment: Python 3.7-era package pins, PyTorch 1.13.1, CUDA 11.x custom rasterizers, COLMAP, separate Depth Anything V2 and Grounded-SAM-2 environments.
- DGGT project checkout: pinned commit `a3276d2bbe4cbb03bcc117830b1836110a27adeb`; its runtime is isolated from AD-GS.
- DynamicVGGT: paper-only dependency for this artifact; no unverified code or checkpoint is imported.

## Current server contract

- Host: `ubuntu-server`, Linux 5.15.
- GPU allocated to this project: NVIDIA A40, 46,068 MiB, physical GPU 0.
- Driver: 535.309.01; driver-reported CUDA capability 12.2.
- Compiler toolkit: CUDA 11.8 (`nvcc 11.8`).
- Base Python/Conda: Python 3.11.5, Conda 24.11.3.
- Dataset storage: network mount under `~/dy/nas`; project checkout under `~/dy/work`.
- Complete inventory and timestamps live in root [`server.md`](../../../../server.md).

## Execution isolation

DGGT export and AD-GS training must not be concurrent on the project GPU. The DGGT probe records GPU identity, tensor contract, elapsed time, allocated/reserved memory, and finite fractions. Cached artifacts are hashed before any evaluation-only labels are opened. The reference module requires only PyTorch and is CPU-testable.
