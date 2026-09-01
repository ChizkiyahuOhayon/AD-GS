# A40 migration gate

Move the eight preprocessed Waymo scenes directly from the source AutoDL instance to persistent A40 storage. Do not stage the dataset on the local Mac and do not copy the 4090 Python environment: CUDA extensions must be rebuilt for the A40 (`sm_86`).

## Source receipt

Run on the source instance before transfer:

```bash
python scripts/waymo/inventory_preprocessed.py \
  /root/autodl-tmp/data/waymo \
  --output /root/autodl-tmp/evidence/waymo-source-inventory.json
```

The command must exit zero. Preserve the receipt with the experiment evidence.

## Direct transfer

Use `rsync -a --partial --info=progress2` over SSH, either by pulling from the A40 or pushing from AutoDL. The exact direction depends on which endpoint accepts SSH. Transfer the contents of the `waymo` directory, not an extra nested `waymo/waymo` directory.

## Destination receipt

Run the same commit on the A40:

```bash
python scripts/waymo/inventory_preprocessed.py \
  /path/to/persistent/data/waymo \
  --output /path/to/evidence/waymo-a40-inventory.json
cmp /path/to/evidence/waymo-source-inventory.json \
    /path/to/evidence/waymo-a40-inventory.json
```

Both commands must exit zero. Matching receipts prove equal per-modality file counts, byte totals, and path/size layout digests, plus content SHA256 for `cameras.npz`, `points3d.ply`, and `colmap.ply` in every scene.

## Environment gate

Build the environment from the repository on A40 with CUDA 11.8 and `TORCH_CUDA_ARCH_LIST=8.6`. Before training, require:

- repository commit `9268b38f356fd975b4ee8a9752da885130f2ddac`;
- clean tracked worktree;
- successful PyTorch CUDA allocation on the selected A40;
- successful imports of both custom rasterization extensions;
- full CPU test suite passing and CUDA lifecycle tests passing.

Only then run the locked `scene090` baseline versus scheduled-contact 1k pair from `scheduled-contact.md`.

Generate the environment receipt with:

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/adgs/bin/python \
  scripts/server/verify_a40_environment.py \
  --repo "$PWD" \
  --expected-commit "$(git rev-parse HEAD)" \
  --output /path/to/evidence/a40-environment.json
```

This invokes real CUDA kernels from simple-knn, the differentiable rasterizer, and PyTorch3D rather than accepting imports alone.

Build the locked environment from the transferred source receipt:

```bash
bash scripts/server/install_a40_environment.sh \
  /home/smbu/anaconda3/envs/adgs-gfdgs \
  "$PWD" \
  /home/smbu/dy/nas/adgs-gfdgs/evidence/a40-migration/source-environment \
  /home/smbu/dy/nas/adgs-gfdgs/evidence/a40-migration/a40-environment
```

The installer uses the verified source dependency closure, removes stale CUDA 12 wheels, installs the official PyTorch 2.0.1 CUDA 11.8 and PyTorch3D 0.7.4 wheels, and rebuilds both local CUDA extensions only for A40 architecture `sm_86`. The separate environment verifier remains the required runtime gate.
