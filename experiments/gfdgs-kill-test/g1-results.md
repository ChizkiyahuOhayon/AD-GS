# G1 result — Waymo scene090, 1k

Protocol: `protocol.md` at commit `ad9e938`  
Treatment implementation: commit `4cda4e8`  
Runner: commit `9a1ee6d`

Both arms used the official scene090 split, seed, configuration, and 1,000
iterations. The treatment changed only the fixed-ID oracle-contact switch.

| Metric | Baseline | Oracle contact | Ratio / delta |
|---|---:|---:|---:|
| Training wall time | 261 s | 319 s | 1.222x |
| Peak training GPU memory | 7,455 MiB | 7,615 MiB | 1.021x |
| PSNR | 26.48436 | 26.44652 | -0.03784 dB |
| SSIM | 0.844503 | 0.844077 | -0.000426 |
| LPIPS(VGG) | 0.417574 | 0.418056 | +0.000482 |

Both checkpoints and all 25 held-out renders completed with finite metrics.
The wall-time and memory ratios are below the locked 1.3x limits, so G1
passes. This does not establish geometric improvement; G2 remains blocked
until the paired held-out dynamic-LiDAR and contact-residual evaluator is
implemented and verified.

Raw evidence is archived outside the nested repository at
`../../experiment_logs/raw/ADGS-GPU-260730-001/gfdgs-kill-test/`.

Instrumentation note: the parent runner was manually terminated after the
oracle renderer had already started, so its render exit code was not appended
to the summary. The renderer itself continued to natural exit and wrote all
25 images and finite `results.json` metrics. No arm was rerun.
