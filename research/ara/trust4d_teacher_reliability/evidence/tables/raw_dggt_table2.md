# DGGT Table 2: Cross-domain novel-view synthesis

**Source**: DGGT, Table 2 and Appendix B.1.

**Extraction type**: raw_table.

| Regime | Method | nuScenes PSNR ↑ | nuScenes SSIM ↑ | nuScenes LPIPS ↓ | Argoverse PSNR ↑ | Argoverse SSIM ↑ | Argoverse LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | MVSplat | 17.84 | 0.563 | 0.451 | 18.67 | 0.647 | 0.304 |
| Zero-shot | NoPoSplat | 19.75 | 0.545 | 0.394 | 22.00 | 0.646 | 0.237 |
| Zero-shot | DepthSplat | 19.52 | 0.601 | 0.376 | 22.05 | 0.636 | 0.280 |
| Zero-shot | STORM | 17.77 | 0.669 | 0.394 | 20.83 | 0.542 | 0.326 |
| Zero-shot | DGGT (Ours) | 25.31 | 0.794 | 0.152 | 26.34 | 0.812 | 0.155 |
| Trained | STORM | 24.54 | 0.784 | 0.267 | 24.97 | 0.791 | 0.240 |
| Trained | DGGT (Ours) | 26.63 | 0.813 | 0.122 | 26.96 | 0.831 | 0.118 |
