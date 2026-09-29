# Results notes — task-free expert expansion for continual pretraining

Setup: tiny MoE GPT (22M params, 4 layers, 4 experts/layer initially, top-2
routing) trained on three domains in sequence — TinyStories → Python code →
open-web-math — 10M tokens per domain (1220 steps each, 3660 total). The model
is never told where the domain boundaries are (only the oracle arm C reads
them). Every number is mean ± std over 3 seeds on an AMD Radeon Pro W7900D
(ROCm). **Forgetting** = loss on a domain at the end of the run minus its loss
at the end of its own phase, averaged over stories and code. Lower is better
for every column.

Raw logs are gitignored; they live in the `results*.tar.gz` bundles.
`python -m experiments.summarize <results dir>` rebuilds the table below and
`python -m experiments.analyze` the figures.

## Main table

| Arm | What it does | Final avg loss | Forgetting |
|---|---|---|---|
| `toy_A` | Normal MoE, 4 experts | 4.337 ± 0.007 | 1.508 ± 0.010 |
| `toy_B` | Normal MoE, 6 experts | 4.370 ± 0.001 | 1.519 ± 0.014 |
| `toy_C` | +1 expert at each TRUE boundary (oracle) | 4.363 ± 0.025 | 1.544 ± 0.026 |
| `toy_D` | +1 expert when the loss detector fires | 4.353 ± 0.012 | 1.534 ± 0.006 |
| `toy_D_fexp` | D, then freeze old experts + their router columns | 4.569 ± 0.018 | 1.665 ± 0.026 |
| `toy_D_fall` | D, then freeze ALL old weights | 5.949 ± 0.107 | 0.958 ± 0.090 |
| `toy_D_fall_k2/k4/k8` | as above, +2 / +4 / +8 experts per boundary | 6.01 / 6.02 / 6.07 | 1.157 / 1.242 / 1.344 |
| `toy_D_femb` | freeze all old weights except token embeddings | 4.802 ± 0.019 | 1.707 ± 0.014 |
| `toy_D_seen` | freeze all old weights + embeddings of already-seen tokens | 4.774 ± 0.087 | 1.066 ± 0.072 |
| `toy_C_seen` | same, oracle boundaries | 4.708 ± 0.009 | 1.058 ± 0.029 |
| `toy_A_replay` | Normal MoE + 25% replay | **3.473 ± 0.007** | **0.106 ± 0.007** |
| `toy_D_replay` | D + 25% replay | 3.480 ± 0.011 | 0.112 ± 0.011 |
| `toy_D_seen_replay` | D + seen-token freeze + 25% replay | 4.195 ± 0.029 | 0.161 ± 0.006 |
| `toy_D_fall_replay` | D + freeze all + 25% replay | 5.358 ± 0.038 | 0.081 ± 0.022 |

Replay is task-free: a reservoir sample of all past training windows, no
domain labels. Signal arms from day 4 on use `confirm_steps=3`.

## Findings

1. **Task-free shift detection works.** With the 3-consecutive-step rule the
   loss z-score detector caught 24/24 boundaries across 12 runs, always exactly
   2 steps late, with zero false triggers (fig 3). Without the rule, 1 of 3
   seeds fired on a noisy batch (z = 4.04) 8 steps early and its 300-step
   cooldown masked the real boundary.
2. **Growth alone does not reduce forgetting.** Static (4 or 6 experts),
   oracle growth and detected growth all forget ~1.51–1.54.
3. **Partial freezing is worse than none.** Freezing only old experts: 1.665.
   Freezing everything except embeddings: 1.707. The unfrozen shared weights
   drift and the frozen parts no longer fit them.
4. **Freezing all old weights cuts forgetting (0.96) but blocks learning**
   (final 5.95; code loss at end of its phase 5.57 vs 2.63). More experts per
   boundary does not restore learning and *increases* forgetting
   (k = 1/2/4/8 → 0.96 / 1.16 / 1.24 / 1.34).
5. **Mechanism 1 — router leak.** With every old weight frozen, the only way an
   old domain can degrade is the router sending its tokens to new experts. The
   share of stories tokens routed to later-born experts rises from 32% (k=1) to
   68% (k=8) and correlates with stories forgetting at r = 0.80 over 14 runs
   (fig 4).
6. **Mechanism 2 — tied embeddings.** The token embedding is tied to the output
   head. Frozen, the model cannot learn to predict new-domain tokens (finding
   4); unfrozen, forgetting rises to 1.71 at the *same* router leak as
   freeze-all (~0.38 vs ~0.32–0.47) — ~0.7 extra forgetting through the
   embeddings.
7. **Best replay-free method: freeze embeddings of already-seen tokens only**
   (running token counts, threshold 100 — task-free). Forgetting 1.07 (−30% vs
   normal MoE) at final loss 4.77; it dominates freeze-all-but-embeddings on
   both axes. Capped because 57% of code-token occurrences are already "seen"
   after the stories phase.
8. **Replay dominates at 25%.** Forgetting 0.106 (−93%) and a *better* final
   loss (3.47). Adding growth on top changes nothing (3.48 / 0.11); adding
   freezing on top hurts learning. Replay also shrinks the router leak
   (seen-freeze: 48% → 30%).
9. **Task-free ≈ oracle** in every protection setting (e.g. seen-freeze 1.066
   vs 1.058; freeze-all 0.958 vs 1.001).
10. **Bit-exact reproducibility** on the W7900D: `toy_A_repro` (re-run of day-1
    `toy_A_s0` on day 4, new container) reproduced 4.343 / 1.518 exactly.

## Figures (`results/figures/`)

1. `1_tradeoff.png` — forgetting vs how well new domains were learned, one
   point per arm (mean ± std). Bottom-left is best; only replay gets there.
2. `2_heldout_curves.png` — held-out loss per domain over training for normal
   MoE, detected growth, seen-token freeze and replay.
3. `3_detection.png` — training loss and experts/layer; true boundaries dashed,
   detector firings at steps 1222 and 2442 in all 3 seeds.
4. `4_router_leak.png` — router leak vs stories forgetting, freeze-all runs
   only (the setting where the router is the sole forgetting path).

## Open / in progress

- **Replay-budget sweep (day 5, running):** 1%, 5%, 10% replay × {normal MoE,
  growth, growth + seen-token freeze}. Question: does growth help when the
  replay budget is small (realistic continual pretraining replays 1–5%)?
- **Generated vs random experts:** every expert added so far is a freshly
  initialised MLP (the "random expert" condition). The LLM-generated condition
  is not yet run.
- Not started: expert pruning, GPT-2-scale runs, learning-rate re-warm for the
  replay baseline.
