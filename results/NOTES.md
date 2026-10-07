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
   loss z-score detector caught 72/72 boundaries across 36 runs (with and
   without replay, at 22M and 101M params), always exactly 2 steps late, with
   zero false triggers (fig 3). Without the rule, 1 of 3
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
11. **Growth has no useful regime at any replay budget** (day 5, fig 5).
    Forgetting / final avg loss, mean of 3 seeds:

    | Replay | Normal MoE | Grow | Grow + seen-token freeze |
    |---|---|---|---|
    | 0% | 1.508 / 4.337 | 1.534 / 4.353 | 1.066 / 4.774 |
    | 1% | 0.699 / 3.795 | 0.716 / 3.807 | 0.599 / 4.457 |
    | 5% | 0.373 / 3.585 | 0.381 / 3.593 | 0.342 / 4.312 |
    | 10% | 0.235 / 3.498 | 0.241 / 3.504 | 0.256 / 4.227 |
    | 25% | 0.106 / 3.473 | 0.112 / 3.480 | 0.161 / 4.195 |

    Even 1% replay halves forgetting; 5% cuts it by 75%. Growth is within
    ~0.02 of the normal MoE at every budget, and slightly worse each time.
    Seen-token freezing lowers forgetting only at ≤5% replay (−0.10 at 1%) and
    costs so much new-domain learning (new-domain loss ~4.86 vs ~3.77) that its
    final loss is worse at every budget. Router leak falls with replay for both
    growth arms (plain growth 22% → 9%, seen-freeze 41% → 30% from 1% to 25%).
12. **The conclusions hold at 4.5× the model size** (day 6, fig 6). Same
    stream and settings with d_model 512, 8 layers, expert width 2048 (101.5M
    params vs 22.4M). Forgetting / final avg loss, 3 seeds:

    | Method | 22M | 101M |
    |---|---|---|
    | Normal MoE | 1.508 / 4.337 | 1.494 ± 0.030 / 4.110 ± 0.041 |
    | Grow on detection | 1.534 / 4.353 | 1.536 ± 0.013 / 4.143 ± 0.031 |
    | Normal MoE + 5% replay | 0.373 / 3.585 | 0.289 ± 0.008 / 3.308 ± 0.015 |
    | Grow + seen-token freeze | 1.066 / 4.774 | 0.845 ± 0.028 / 4.414 ± 0.024 |

    Growth is again slightly worse than the normal MoE; 5% replay removes 81%
    of forgetting (75% at 22M). One trend: seen-token freezing gets relatively
    better with size — 44% less forgetting than the normal MoE (29% at 22M) and
    a smaller final-loss penalty (+0.30 vs +0.44). With two sizes this is a
    trend to state carefully, not a scaling law.
13. **No growth combination beats plain replay at 101M either** (days 6b–7).
    Forgetting / final avg loss, 3 seeds:

    | Replay | Normal MoE | Grow + seen-token freeze |
    |---|---|---|
    | 1% | 0.605 ± 0.014 / 3.512 ± 0.031 | 0.432 ± 0.032 / 4.147 ± 0.019 |
    | 5% | 0.289 ± 0.008 / 3.308 ± 0.015 | 0.230 ± 0.014 / 4.019 ± 0.036 |

    Seen-token freezing still lowers forgetting on top of replay (−29% at 1%,
    −20% at 5%; at 22M it was −14% / −8%, so the retention gain grows with
    size), but its final loss is ~0.65–0.7 worse at both budgets: with replay
    already protecting old domains, freezing mostly costs new learning.
14. **LLM-generated experts perform the same as random ones of the same size**
    (day 7, fig 7). Gemma-2-2b-it, given the foundry contract and the real
    failure context, wrote 8 designs in 14 tries (all valid on the first
    attempt) — but almost all the same block: LayerNorm → Linear 256→128 →
    activation → Linear 128→256 (~66k params; one design used width 256).
    That is ~8× smaller than the 526k-param template, so a size-matched
    random-init template (hidden 128) is the control. Growth on detection,
    22M model, forgetting / final avg loss (no freezing: 6 seeds, days 7 + 7c;
    seen-token freezing: 3 seeds):

    | New expert | No freezing (n = 6) | Seen-token freezing (n = 3) |
    |---|---|---|
    | Random init, full size (526k) | 1.532 ± 0.018 / 4.351 ± 0.013 | 1.066 ± 0.072 / 4.774 ± 0.087 |
    | Gemma-written (~66k) | 1.557 ± 0.017 / 4.377 ± 0.015 | 1.074 ± 0.075 / 4.941 ± 0.048 |
    | Random init, same size (66k) | 1.576 ± 0.034 / 4.389 ± 0.024 | 1.062 ± 0.087 / 4.924 ± 0.068 |

    Welch t-tests over the 6 no-freeze seeds: **Gemma vs same-size random is
    not significant** (forgetting −0.019, p = 0.27; final −0.013, p = 0.30) —
    and the gap shrank from 3 to 6 seeds, consistent with no effect. With
    seen-token freezing the sign flips (Gemma slightly worse). **Expert size is
    significant**: full-size vs same-size template, forgetting −0.044
    (p = 0.024), final −0.038 (p = 0.010); under seen-token freezing, where new
    experts carry all new-domain learning, the full-size template learns new
    domains clearly better (new-domain loss 4.86 vs ~5.10). Answer to the
    original question: LLM-written experts are no better than random ones of
    the same size; what matters is how much capacity is added.
15. **Detection sensitivity: reliable from a 10% shift, borderline at 5%,
    zero false triggers** (day 8, fig 8). Two-phase runs: pure stories, then
    stories with a share q of code mixed in (or pure stories again as a
    no-shift control); 3 seeds each, same detector (z > 4 for 3 consecutive
    steps). Switch at step 1220:

    | q (code share) | Training-loss jump | Detected | Detection delay (steps) |
    |---|---|---|---|
    | 0% (no shift) | −0.05 | 0/3 (correct) | — |
    | 5% | +0.10 | 1/3 | 10 |
    | 10% | +0.23 | 3/3 | 3, 5, 10 |
    | 25% | +0.59 | 3/3 | 2, 2, 5 |
    | 50% | +1.15 | 3/3 | 2, 2, 2 |

    The loss jump is close to linear in q, and the z-score at firing drops
    with it (4.1–10 at 10%, 16 at 25%, 22–38 at 50%), so the miss at 5% is
    simply the jump (~0.1 nats, ~4% of the loss) sitting at the z = 4
    threshold. No trigger fired anywhere except just after a real switch
    (0 false triggers in 15 runs × 2440 steps). Together with finding 1: the
    detector is reliable for shifts that move the loss by ≳0.2 nats; smaller
    or gradual drift would need a lower threshold or a slower, averaged
    signal (not tested).

## Figures (`results/figures/`)

1. `1_tradeoff.png` — forgetting vs how well new domains were learned, one
   point per arm (mean ± std). Bottom-left is best; only replay gets there.
2. `2_heldout_curves.png` — held-out loss per domain over training for normal
   MoE, detected growth, seen-token freeze and replay.
3. `3_detection.png` — training loss and experts/layer; true boundaries dashed,
   detector firings at steps 1222 and 2442 in all 3 seeds.
4. `4_router_leak.png` — router leak vs stories forgetting, freeze-all runs
   only (the setting where the router is the sole forgetting path).
5. `5_replay_budget.png` — forgetting and final loss vs replay budget (0, 1,
   5, 10, 25%) for normal MoE, growth, and growth + seen-token freeze.
6. `6_scale.png` — the same four methods at 22M vs 101M params.
7. `7_expert_source.png` — Gemma-written vs random-init new experts (full
   size and size-matched), without and with seen-token freezing.
8. `8_detection_sensitivity.png` — loss jump and detection delay vs shift
   size (0–50% code), one dot per seed.

## Open / in progress

- Experiments complete; remaining work is the write-up.
- Not tested: gradual drift (shift ramps in slowly rather than switching).
- Not started: expert pruning, GPT-2 (pretrained, upcycled) runs, learning-rate re-warm for the
  replay baseline.
