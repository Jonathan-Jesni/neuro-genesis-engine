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
domain labels. Signal arms use `confirm_steps=3` from `seen_freeze_and_replay.txt` on
(the baselines and freezing runs used 1).

Two details that apply to every growth run:
- **Initialisation.** The model's initial experts use GPT-style init (normal,
  std 0.02, zero bias). Every *grown* expert — template, size-matched or
  Gemma-written — is built by the foundry with PyTorch's default `nn.Linear`
  init (uniform; std ≈ 0.036 for the first layer and 0.018 for the second at
  these sizes, with random biases). Comparisons between growth variants are like
  for like; comparisons between growth and static runs include this difference.
- **Statistics.** "±" is the sample standard deviation over seeds (n − 1), in
  the tables and in the figures' error bars alike.

## Findings

1. **Task-free shift detection works.** With the 3-consecutive-step rule the
   loss z-score detector caught 150/150 boundaries across all 75 three-domain
   runs that used it (with and without freezing, replay and the drift
   detector, random or Gemma-written experts, 22M and 101M params), always
   exactly 2 steps late, with zero false triggers (fig 3). The 6 pretrained
   GPT-2 growth runs add 12/12 more (finding 20). The one exception is a shift
   that *lowers* the loss, which a spike detector cannot see (finding 18). Without the rule, 1 of 3
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
10. **Bit-exact reproducibility** on the W7900D: `toy_A_repro` (re-run of the baseline
    `toy_A_s0` weeks later, in a new container) reproduced 4.343 / 1.518 exactly.
11. **Growth has no useful regime at any replay budget** (`replay_budget.txt`, fig 5).
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
12. **The conclusions hold at 4.5× the model size** (`scale_101m.txt`, fig 6). Same
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
13. **No growth combination beats plain replay at 101M either** (`scale_101m.txt`).
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
    (`generated_vs_random.txt`, fig 7). Gemma-2-2b-it, given the foundry contract and the real
    failure context, wrote 8 designs in 14 tries (all valid on the first
    attempt) — but almost all the same block: LayerNorm → Linear 256→128 →
    activation → Linear 128→256 (~66k params; one design used width 256).
    That is ~8× smaller than the 526k-param template, so a size-matched
    random-init template (hidden 128) is the control. Growth on detection,
    22M model, forgetting / final avg loss (no freezing: 6 seeds;
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
    the same size; what matters is how much capacity is added. (Design
    assignment is `(seed × 2 + growth event) mod 8`, so seeds 4–5 reuse the
    designs of seeds 0–1, and two of the 8 designs are identical: the 6 seeds
    cover 7 distinct designs.)
15. **Detection sensitivity: reliable from a 10% shift, borderline at 5%,
    zero false triggers** (`detection_sensitivity.txt`, fig 8). Two-phase runs: pure stories, then
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
    shifts would need a lower threshold.
16. **Gradual drift is never detected** (`gradual_drift.txt`, fig 9). Same two-phase setup,
    but phase 2's code share rises *linearly* from 0 to q_max over its 1220
    steps (`ramp:stories:code:<q_max>`), 3 seeds each:

    | Phase 2 | Detected | Training loss, end of phase 2 vs end of phase 1 |
    |---|---|---|
    | abrupt 25% code (finding 15) | 3/3, 2–5 steps | jump of +0.59 at the switch |
    | abrupt 50% code (finding 15) | 3/3, 2 steps | jump of +1.15 at the switch |
    | drift to 25% | 0/3 | +0.03 |
    | drift to 50% | 0/3 | +0.23 |
    | drift to 100% | 0/3 | +0.49 |

    Drifting all the way to pure code raises the loss by 0.49 — as much as
    the abrupt 25% switch that is caught every time — but spread over 1220
    steps, so the detector's 50-step rolling baseline rises with it and no
    step ever stands out. A loss-spike detector only sees *changes in the
    rate*, not slow accumulation; catching drift would need a long-horizon
    reference (e.g. a frozen baseline window or held-out-loss trend).
17. **A long-horizon drift detector catches fast drift at zero cost to
    everything else** (`drift_detector.txt`, fig 10). `DriftDetector` runs alongside the
    unchanged spike detector: it keeps an EMA of the loss (α = 0.02) and that
    EMA's lowest value since the last growth, and fires when the EMA stays
    more than 10% above it for 20 steps (200-step warm-up after each reset).
    3 seeds each:

    | Run | Fired | When |
    |---|---|---|
    | drift to 100% code | 3/3, drift signal | 309–342 steps in, at 25–28% code |
    | drift to 50% code | 2/3, drift signal | 1111–1215 steps in, at 46–50% code |
    | drift to 25% code | 0/3 | (loss rises only ~1%) |
    | no shift | 0/3 | no false alarms |
    | abrupt 10% switch | 3/3, spike signal | steps 1225 / 1230 / 1223 — identical to finding 15 |
    | normal 3-domain run | 2 growths each, spike signal | 1222 and 2442 — identical to before |

    The drift signal never fired in the abrupt or normal runs, which are
    bit-identical to the runs without it (same growth steps, same final loss),
    so adding it costs nothing there. It fires once the *accumulated* rise
    passes the 10% margin: a fast drift crosses it after ~25% of the data has
    changed, a half-speed drift only near the end of the phase, and a small
    drift never.

    **Margin sweep** (3 seeds each; margin 2% / 5% / 10% / 15%):

    | Margin | drift to 25% | drift to 50% | drift to 100% | no shift | normal 3-domain run |
    |---|---|---|---|---|---|
    | 2% | **2/3, at 19–22% code** | — | — | 0/3 (no false alarm) | spike only, identical |
    | 5% | 0/3 | **3/3, at 14–24% code** | — | 0/3 (no false alarm) | spike only, identical |
    | 10% | 0/3 | 2/3, at 46–50% code | 3/3, at 25–28% code | 0/3 | spike only, identical |
    | 15% | — | 0/3 | 3/3, at 45–50% code | — | — |

    A tighter margin catches drift earlier and more reliably, and even at 2%
    there were no false alarms anywhere: no firing on unchanged data, and
    normal 3-domain runs grew only at the two real boundaries, bit-identical
    to runs without the drift detector. At 2% it catches even the small
    25% drift (2/3), which every looser margin missed; 15% is too strict.
    The margin at which false alarms start was not reached (below 2% is
    untested) — over 2–10% the detector is safe in this setup, and tighter
    is better.
18. **Reversing the domain order changes the sizes, not the conclusions**
    (`domain_order.txt`, fig 11). The same 22M setup with the stream reversed
    (math → code → stories). Forgetting / final avg loss, 3 seeds:

    | Run | Reversed order | Normal order |
    |---|---|---|
    | Normal MoE | 2.711 ± 0.099 / 5.238 ± 0.080 | 1.508 / 4.337 |
    | Grow on detection | 2.823 ± 0.107 / 5.309 ± 0.083 | 1.534 / 4.353 |
    | Normal MoE + 5% replay | **0.504 ± 0.008 / 3.777 ± 0.006** | **0.373 / 3.585** |
    | Grow + seen-token freeze | 1.237 ± 0.062 / 4.710 ± 0.059 | 1.066 / 4.774 |

    - Forgetting is ~1.8× larger. Ending on easy stories text overwrites math
      (+2.48) and code (+2.94).
    - Growth is worse than the normal MoE in every seed (+0.10 to +0.14
      forgetting).
    - 5% replay removes 81% of the forgetting and is again clearly best.
    - New in this order: **seen-token freezing beats the normal MoE on final
      loss** (−0.53). The average is dominated by the two forgotten domains,
      which it protects (final math 6.77 vs 7.64, code 3.34 vs 5.42), at the
      cost of learning stories (4.02 vs 2.66). Its router leak is 32–37%. So
      whether seen-token freezing pays off depends on the order.
    - **Detection caught only 1 of 2 boundaries, in all 6 growth runs.**
      Code → stories was caught 2 steps late as usual; math → code was never
      caught. After the math phase the model's code loss is already *lower*
      than its math loss (4.32 vs 5.15; open-web-math is full of LaTeX and
      code-like text), so the switch makes the training loss fall. A loss-spike
      detector only sees shifts that make the data look harder. There were
      still no false triggers.
19. **LR re-warming at each domain makes forgetting worse at every replay
    budget** (`lr_rewarm.txt`). The warm-up + cosine schedule restarts at each
    true boundary (as in Ibrahim et al. 2024), normal MoE, 3 seeds.
    Forgetting / final avg loss:

    | Replay | Re-warm per domain | One schedule (main runs) |
    |---|---|---|
    | 0% | 1.630 ± 0.011 / 4.368 ± 0.003 | 1.508 ± 0.010 / 4.337 ± 0.007 |
    | 1% | 0.907 ± 0.009 / 3.890 ± 0.011 | 0.699 ± 0.014 / 3.795 ± 0.012 |
    | 5% | 0.544 ± 0.002 / 3.652 ± 0.004 | 0.373 ± 0.012 / 3.585 ± 0.007 |

    - Re-warming adds 0.12 / 0.21 / 0.17 forgetting and is worse in every seed.
    - It does learn the last domain better (math 4.69 vs 4.90), but loses more
      on stories and code than it gains.
    - With one schedule, the later domains are trained at a lower LR, which
      protects the earlier ones. So the main replay baseline was not
      handicapped by skipping re-warming.
    - Replay still cuts forgetting by two thirds under re-warming.
20. **The conclusions hold when starting from pretrained GPT-2** (`gpt2.txt`,
    fig 11).
    - Setup: GPT-2 small (124M) upcycled into an MoE. Each layer's MLP is
      copied into 4 experts and a fresh top-2 router is added (294M params
      total).
    - Training: lr 2e-4, 100 warm-up steps, 3M tokens per domain (366 steps
      per phase), normal order.

    Forgetting / final avg loss, 3 seeds:

    | Run | Forgetting | Final avg loss |
    |---|---|---|
    | Normal MoE | 0.183 ± 0.009 | 2.307 ± 0.006 |
    | Grow on detection | 0.227 ± 0.014 | 2.353 ± 0.010 |
    | Normal MoE + 5% replay | **0.042 ± 0.003** | **2.217 ± 0.001** |
    | Grow + seen-token freeze | 0.200 ± 0.021 | 2.522 ± 0.009 |

    - Detection is unchanged: 12/12 boundaries, 2 steps late, no false
      triggers.
    - Forgetting is ~8× smaller than when training from scratch, but the
      ranking is the same.
    - 5% replay removes 77% of the forgetting and has the best final loss.
    - Growth is worse in every seed, by 24% more forgetting: the largest gap
      in any setting. A likely cause, not tested: the new experts are randomly
      initialised beside four copies of a pretrained MLP.
    - **Seen-token freezing stops helping**: +9% forgetting and +0.21 final
      loss.
      - It almost removes code forgetting (0.037 vs 0.106).
      - But stories forgetting *rises* (0.363 vs 0.261) even though router
        leak is only 15%.
      - New domains are learned worse (code 1.84 vs 1.61, math 3.58 vs 3.20
        at the end of their phases).
      - A likely cause, not tested: here the "unseen" embedding rows it
        leaves trainable are pretrained and tied to the output layer. Training
        them on code shifts the output distribution for stories (the
        embedding path of finding 6).

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
9. `9_gradual_drift.png` — training loss for an abrupt switch vs gradual
   drift, with detector firings marked.
10. `10_drift_detector.png` — the same drift runs with the long-horizon drift
    detector added, against the spike detector alone.
11. `11_robustness.png` — growth, 5% replay and seen-token freezing compared
    with the normal MoE in four settings (22M, 101M, reversed order,
    pretrained GPT-2). Forgetting is shown as a % change and final loss as a
    difference, paired by seed.

## Open / in progress

- Experiments complete; remaining work is the write-up.
- Not started: expert pruning; a detector for shifts that lower the loss
  (finding 18); the GPT-2 growth and seen-freeze causes suggested in finding 20.
