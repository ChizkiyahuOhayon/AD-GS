# DGGT Table 5: 3D motion evaluation

**Source**: DGGT, Table 5.

**Extraction type**: raw_table.

| Method | EPE3D ↓ | Acc3D Strict ↑ | Acc3D Relax ↑ | Outlier ↓ |
|---|---:|---:|---:|---:|
| NSFP | 0.698 | 42.17 | 54.26 | 0.919 |
| NSFP++ | 0.711 | 53.10 | 63.02 | 0.989 |
| STORM | 0.276 | 81.12 | 85.61 | 0.658 |
| DGGT (Ours) | 0.183 | 85.42 | 90.42 | 0.328 |

The paper's Appendix A.1 states that dynamic masks and identities are constructed from Waymo LiDAR 3D boxes and tracking IDs. These numbers therefore characterize an annotation-assisted source teacher, not a self-supervised Trust4D result.
