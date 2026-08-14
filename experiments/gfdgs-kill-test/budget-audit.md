# GF-DGS kill-test GPU budget audit

Date: 2026-08-13  
Status: derived from completed G1 evidence before any 5k run

## Locked constraint

The protocol caps all exploratory work before the decision at 0.75 RTX 4090
GPU hours, or 2,700 seconds. This audit does not change that cap.

## GPU wall time already consumed

| Operation | Seconds | Evidence |
|---|---:|---|
| Oracle 1-step preflight train | 78 | interactive shell measurement and retry log |
| Oracle 1-step checkpoint render | 92 | interactive shell measurement and render log |
| G1 baseline 1k train/evaluate/save | 261 | `scene090-baseline-1k.summary.txt` |
| G1 baseline checkpoint render | 90 | `scene090-baseline-1k.summary.txt` |
| G1 oracle 1k train/evaluate/save | 319 | `scene090-oracle-1k.summary.txt` |
| G1 oracle checkpoint render | 88 | conservative timestamp difference, 06:32:16–06:33:44 |
| **Total accounted** | **928** | **0.258 GPU hours** |

The oracle render parent-shell exit was not captured, so 88 seconds is a
conservative lower bound. The remaining budget is therefore at most 1,772
seconds (29 minutes 32 seconds).

## Lower bound for the locked paired 5k run

The baseline 1k optimization loop took 173 seconds and the oracle loop took
230 seconds. Even under the unrealistically favorable assumption that later
iterations never slow as Gaussian count grows, retaining only each run's
observed fixed loading/evaluation/save overhead gives:

- baseline 5k lower bound: `5 × 173 + (261 - 173) = 953 s`;
- oracle 5k lower bound: `5 × 230 + (319 - 230) = 1,239 s`;
- paired training lower bound: `2,192 s` (36 minutes 32 seconds).

This exceeds the remaining budget by at least 420 seconds before independent
LPIPS rendering or paired geometry evaluation. Actual 5k time should be higher
because AD-GS densifies the point set through iteration 5k.

## Decision

The paired 5k G2 experiment must not start under the locked 0.75-hour cap. It
cannot finish within the budget even in the constant-throughput lower bound.
Starting only one arm would not answer the paired question and is also
disallowed.

The only remaining in-budget GPU action is the already locked evaluator on the
existing 1k checkpoints. It requires two checkpoint renders and yields an
exploratory early geometry signal:

- if the 1k oracle misses either the 10% AbsRel or 50% contact threshold, stop
  GF-DGS immediately;
- if it passes, label it promising but unconfirmed. It is not a substitute for
  G2, and no 5k or 60k run may start without an explicitly expanded budget and
  a new pre-run protocol. Under the user's stated rule to stop when funding is
  insufficient, the default remains to stop after reporting the 1k result.
