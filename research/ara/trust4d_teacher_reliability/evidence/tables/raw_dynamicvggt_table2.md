# DynamicVGGT Table 2: Full-image novel-view synthesis

**Source**: DynamicVGGT, Table 2.

**Extraction type**: raw_table.

| Method | Supervision | KITTI PSNR ↑ | KITTI SSIM ↑ | Waymo PSNR ↑ | Waymo SSIM ↑ |
|---|---|---:|---:|---:|---:|
| 3DGS | Full | 17.13 | 0.267 | 25.13 | 0.741 |
| DeformableGS | Full | 17.10 | 0.266 | 25.29 | 0.761 |
| GS-LRM | Camera | 20.02 | 0.520 | 25.18 | 0.753 |
| STORM | Camera | 21.26 | 0.535 | 25.03 | 0.750 |
| DynamicVGGT | Image-only | 18.07 | 0.376 | 24.07 | 0.676 |

These full-image values use the DynamicVGGT paper's feed-forward protocol and are not paired with released AD-GS runs.
