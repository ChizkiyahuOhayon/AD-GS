# Derived DGGT Figure 1 quality-speed data

**Source**: DGGT, Figure 1 and the quantitative values in Table 1.

**Extraction type**: derived_subset.

| Method | Waymo NVS PSNR ↑ | Reported inference time |
|---|---:|---:|
| LGM | 18.53 | 0.06 s |
| GS-LRM | 25.18 | 0.02 s |
| MVSplat | 20.56 | 0.08 s |
| NoPoSplat | 24.31 | 23.22 s |
| DepthSplat | 23.26 | 0.11 s |
| STORM* | 26.05 | 0.50 s |
| STORM | 26.38 | 0.18 s |
| VGGT++ | 22.50 | 0.24 s |
| DGGT (Ours) | 27.41 | 0.39 s |

Only feed-forward rows with second-scale inference are included; optimization methods measured in minutes are deliberately omitted. This is plotting data for the source figure relationship, not a Trust4D comparison.
