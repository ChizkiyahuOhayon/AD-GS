# A40 server record

This file records observed server state for the Trust4D-GS experiments. Facts
copied from server output are separated from decisions and unresolved items.

## Provenance

- Observation time: `2026-08-14T15:59:23+08:00`.
- Host: `ubuntu-server`; user shown in the shell prompt: `smbu`.
- Source: console output pasted by the user from
  `scripts/trust4d/inventory_server.sh`.
- The generated `server_inventory.txt` itself has not been transferred, so a
  file SHA-256 is not yet available.

## Operating system

- Kernel: Linux `5.15.0-185-generic`.
- Build: `#195-Ubuntu SMP Fri Jun 19 17:11:50 UTC 2026`.
- Architecture: `x86_64`.

## GPU snapshot

- NVIDIA driver: `535.309.01`.
- Driver-advertised maximum CUDA compatibility: `12.2`.
- Four NVIDIA A40 devices were visible. The research allocation remains one
  GPU; visibility does not authorize using the other three devices.
- Each device reported `46068 MiB` total memory.

| Physical GPU | Memory used | Utilization | Temperature | Power | Observed state |
|---:|---:|---:|---:|---:|---|
| 0 | 9 MiB | 0% | 32 C | 28/300 W | available except 4 MiB graphics process |
| 1 | 5334 MiB | 98% | 51 C | 215/300 W | occupied by another Python process |
| 2 | 14415 MiB | 75% | 51 C | 130/300 W | occupied by several processes |
| 3 | 18690 MiB | 0% | 39 C | 77/300 W | memory occupied by another Python process |

Observed GPU processes:

- GPU 0: `UrbanJapan-Linux-Shipping`, 4 MiB.
- GPU 1: `python`, 5326 MiB.
- GPU 2: `python`, 5326 MiB; `UrbanJapan-Linux-Shipping`, 2256 MiB;
  `apex/bin/python`, 4710 MiB and 2020 MiB.
- GPU 3: `qwen3vl/bin/python`, 18682 MiB.

The experiment must set `CUDA_VISIBLE_DEVICES=0`; inside the isolated process,
physical GPU 0 is addressed as `cuda:0`. No process on GPUs 1--3 is part of
this project.

## CUDA toolchains

- System `nvcc`: CUDA `11.8`, build `V11.8.89`.
- NVIDIA driver compatibility reported by `nvidia-smi`: CUDA `12.2`.

These numbers describe different layers and are not a contradiction. The DGGT
environment will use the official PyTorch 2.4.1 CUDA 12.1 wheel, which is within
the observed driver compatibility. Any package that compiles against the
system toolkit must be checked separately rather than assuming it uses 12.1.

## Python and Conda

- Active environment during inventory: `base`.
- Python: `3.11.5`.
- Conda: `24.11.3`.
- Existing named environments: `apex`, `axle`, `beliefgauss`, `blender`, `cc`,
  `dyslam`, `embodiedocc`, `explore-eqa`, `gaussian_grouping`, `gf2`, `gs`,
  `gspl`, `gt`, `havlnce`, `langsplat`, `map`, `meshsplat`, `openeqa`,
  `osmloc`, `podgs`, `qwen3vl`, `rasterize`, `rayfronts`, `refsplat`,
  `rep_geo`, `rpgwm`, `scene_splat`, `semslam`, `sgs-slam`, `splatam`,
  `splatt3r-slam`, `uavon`, and `vlnce`; micromamba also lists `evimem` and
  `mesh_splatting`.

None of these names proves a compatible DGGT or AD-GS environment. EXP-001
uses a new isolated `dggt` environment and verifies package versions at
runtime.

## Repository state

| Repository | Server path | Observed commit | State |
|---|---|---|---|
| AD-GS fork | `~/dy/work/AD-GS-Trust4D` | `aa2844873c61281f16fb8a3125b64bf6ae3ff2b1` | clean `trust4d-main`, but behind current local work |
| official DGGT | `~/dy/work/DGGT` | `a3276d2bbe4cbb03bcc117830b1836110a27adeb` | clean detached HEAD, intentionally pinned |

Detached HEAD is correct for DGGT because the experiment requires that exact
upstream commit and makes no DGGT source edits.

## Storage and dataset discovery

- Declared NAS root: `~/dy/nas`, mounted from
  `//10.24.1.47/user1/lss/home/dy`.
- Capacity: `311 T`; used: `187 T`; available: `124 T`; utilization: `61%`.
- The bounded inventory search found no Waymo `.tfrecord`, no `cameras.npz`,
  and no `points3d.ply` or `colmap.ply` under the searched depths.
- It did not find the required processed AD-GS `scene006/image` directory.
- Paths containing `scene0060_00` belong to
  `embodiedocc/occscannet/posed_images`, i.e. ScanNet. They are not Waymo and
  must not be used for EXP-001 or an AD-GS baseline.

This is evidence that the required Waymo location is unresolved, not proof
that no Waymo data exists anywhere on the 311 T NAS. A targeted path supplied
by the user or a bounded search under a known project directory is required;
an unrestricted recursive scan of the shared NAS is inappropriate.

## Decisions and next gate

1. Pull the AD-GS fork after the pending local runner commit is pushed.
2. Locate an authentic processed Waymo `scene006` containing both `image/` and
   `cameras.npz`, or download and preprocess the locked StreetGS/AD-GS scene.
3. Keep DGGT at the observed pinned commit.
4. Create the isolated DGGT environment only after the data path is resolved;
   do not run training in `base`.
5. Run EXP-001 on physical GPU 0 only. No AD-GS modification or treatment
   training is authorized until the teacher contract passes.

No baseline reproduction, DGGT forward pass, or Trust4D-GS treatment has run
on this server as of this record.
