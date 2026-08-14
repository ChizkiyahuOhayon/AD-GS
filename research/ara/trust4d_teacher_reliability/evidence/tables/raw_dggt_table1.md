# DGGT Table 1: Waymo novel-view synthesis

**Source**: DGGT, Table 1.

**Extraction type**: raw_table.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Inference | Dynamic | Pose-free |
|---|---:|---:|---:|---:|:---:|:---:|
| EmerNeRF | 24.51 | 0.738 | 33.99 | 14 min | yes | no |
| 3DGS | 25.13 | 0.741 | 19.68 | 23 min | no | no |
| PVG | 22.38 | 0.661 | 13.01 | 27 min | yes | no |
| DeformableGS | 25.29 | 0.761 | 14.79 | 29 min | yes | no |
| LGM | 18.53 | 0.447 | 9.07 | 0.06 s | no | no |
| GS-LRM | 25.18 | 0.753 | 7.94 | 0.02 s | no | no |
| MVSplat | 20.56 | 0.697 | 10.13 | 0.08 s | no | no |
| NoPoSplat | 24.31 | 0.751 | 9.08 | 23.22 s | no | yes |
| DepthSplat | 23.26 | 0.696 | 10.05 | 0.11 s | no | no |
| STORM* | 26.05 | 0.819 | 5.91 | 0.50 s | yes | no |
| STORM | 26.38 | 0.794 | 5.48 | 0.18 s | yes | no |
| VGGT++ | 22.50 | 0.749 | 3.80 | 0.24 s | no | yes |
| DGGT (Ours) | 27.41 | 0.846 | 3.47 | 0.39 s | yes | yes |

LPIPS is reproduced in the source table's displayed scale; do not silently compare it with an evaluator that reports LPIPS in `[0,1]`.
