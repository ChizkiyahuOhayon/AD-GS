# GF-DGS single-scene kill test

Status: locked before GPU execution  
Scene: Waymo scene090  
Maximum exploratory GPU budget before the decision: 0.75 RTX 4090 GPU hours

## Question

Does actor-wise ground contact provide enough geometric signal to justify further GF-DGS development?

This is a feasibility test, not a paper result. It deliberately uses the simplest fixed-ID, oracle-contact Stage-B prototype. If an oracle version cannot improve geometry without damaging held-out rendering, a learned annotation-free road model is unlikely to rescue the idea within the available budget.

## Locked comparison

- Baseline: pristine AD-GS configuration and seed, stopped and saved at 1k/5k.
- Treatment: the same configuration and seed with only fixed-ID oracle contact enabled.
- Dataset and split: the already prepared official Waymo scene090 split.
- Evaluation: identical held-out cameras, standard PSNR/SSIM/LPIPS, sparse held-out LiDAR dynamic-object depth error, wall time, and peak GPU memory.
- No full eight-scene run and no hyperparameter sweep are allowed before this test is decided.

## Gates

### G0 — zero-GPU feasibility

Pass only if:

- the real camera trajectory is representable by the sampled AD-GS position basis with training RMS residual below 5 cm; and
- at least 10 scene090 instance IDs have at least 20 points over at least 5 frames, with at least 70% of those tracks avoiding a consecutive centroid jump above 10 m.

Observed before protocol lock and therefore labelled exploratory: camera-basis training RMS is 2.55 mm; 17 IDs meet the support criterion and 15/17 avoid a jump above 10 m. G0 passes.

### G1 — 1k smoke

Both arms must finish with finite losses and metrics. Treatment wall time and peak memory must be at most 1.3 times baseline. Any crash, NaN/Inf, ID-shape corruption, or substantial throughput regression kills the treatment implementation before 5k.

### G2 — 5k decision

Continue only if all conditions hold:

- held-out PSNR decreases by no more than 0.20 dB and LPIPS(VGG) increases by no more than 0.01;
- sparse held-out dynamic-object LiDAR depth error improves by at least 10%;
- actor contact residual improves by at least 50%; and
- the improvement is not caused by evaluating fewer valid actors or pixels.

If any condition fails, conclude that the idea does not work under this high-signal oracle test and stop GF-DGS. Do not run another scene or the eight-scene table.

### G3 — one formal confirmation

Only after G2 passes, run one scene090 treatment to 60k and compare it with the existing official baseline: PSNR 30.7210, SSIM 0.91145, LPIPS(VGG) 0.24252. Maximum budget is 3 GPU hours. Failure to retain the G2 geometric improvement stops the project.

## Interpretation limits

- Passing supports further work; it does not establish novelty, annotation-free operation, or CVPR readiness.
- Failing is a deliberate budget stop, not an invitation to tune thresholds or search scenes.
- The two zero-GPU audits are exploratory because they preceded this locked protocol. All GPU outcomes following this document are confirmatory with respect to the gates above.
