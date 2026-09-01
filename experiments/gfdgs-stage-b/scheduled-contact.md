# Scheduled-contact scene090 gate

## Question

Does gradually introducing the GF-DGS road-contact projection recover the image-quality loss caused by enabling hard contact from iteration 1?

This 1k run is an engineering gate, not publication evidence.

## Locked comparison

- Dataset: Waymo `scene090`
- Split: repository default, 78 train views and 25 test views
- Budget: 1,000 iterations
- Baseline: `arguments/waymo.py`
- Scheduled GF-DGS: `arguments/waymo_gfdgs_scheduled.py`
- Schedule: strength 0 through 20% of training, linear ramp to 1 over the next 60%, strength 1 for the final 20%
- Fairness: both arms must use the same clean commit, data directory, environment, and GPU

Run:

```bash
PYTHON=/path/to/adgs/bin/python \
  bash experiments/gfdgs-stage-b/run_scene090_smoke.sh \
  /path/to/data/waymo/scene090 \
  /path/to/runs \
  /path/to/evidence/gfdgs-stage-b \
  scheduled-pair
```

## Gates

Engineering pass:

- both arms finish train, save, reload, and render with exit code 0;
- no NaN, CUDA assertion, or OOM;
- the scheduled run ends with contact strength 1;
- `gauge_fix.pth` exists at iteration 1000.

Promotion to 10k requires scheduled GF-DGS to exceed the paired baseline in PSNR while not worsening LPIPS-VGG, with SSIM degradation no larger than 0.001. Otherwise stop this schedule and do not expand to other scenes.

Record PSNR, SSIM, LPIPS-VGG, LPIPS-Alex, FPS, train seconds, render seconds, peak VRAM, exact commit, and SHA256 of both `results.json` files.
