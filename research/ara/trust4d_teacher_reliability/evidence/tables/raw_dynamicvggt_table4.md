# DynamicVGGT Table 4: Architecture ablation

**Source**: DynamicVGGT, Table 4.

**Extraction type**: raw_table.

| Variant | Dataset | Mean Acc. ↓ | Mean Comp. ↓ | Mean NC ↑ | Median Acc. ↓ | Median Comp. ↓ | Median NC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline VGGT | KITTI | 1.489 | 0.690 | 0.918 | 1.329 | 0.535 | 0.971 |
| + TA and FPH | KITTI | 0.927 | 0.600 | 0.915 | 0.857 | 0.474 | 0.932 |
| + DGS head | KITTI | 0.901 | 0.584 | 0.939 | 0.733 | 0.464 | 0.963 |
| Baseline VGGT | Waymo | 4.635 | 2.667 | 0.561 | 2.634 | 1.734 | 0.590 |
| + TA and FPH | Waymo | 4.330 | 2.939 | 0.561 | 2.224 | 1.649 | 0.593 |
| + DGS head | Waymo | 4.021 | 2.390 | 0.562 | 1.971 | 1.564 | 0.603 |
