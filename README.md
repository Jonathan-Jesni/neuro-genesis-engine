# Neuro-Genesis Engine

**A neural network that grows its own brain while it trains.**

Think of a hospital that hires its own specialists: when the training loss spikes — the network hitting data it can't handle — the system doesn't page a human. It packages the failure into a prompt, has a local LLM (Gemma-2-2b-it, running on-device) *write the PyTorch source code for a new expert module*, validates that generated code through an AST screen, sandboxed execution, and a smoke test, then hot-swaps the new expert into the live Mixture-of-Experts network — mid-training, without stopping the optimizer, with full rollback if anything fails. If the LLM's code is rejected, the exact rejection reason is fed back and it tries again. No human in the loop, end to end.

Built for the **AMD Developer Hackathon ACT II — Unicorn Track**.

## Verified on real AMD hardware

This is not "ROCm-compatible in theory." The system **runs and has been verified on an AMD Radeon Pro W7900D via ROCm 7.2** (AMD Developer Cloud):

- [`orchestrator_run_amd_hardware.jsonl`](orchestrator_run_amd_hardware.jsonl) is the unedited event log of a full 300-step self-expanding run executed on that GPU — every training step, loss spike, LLM generation, and expert registration, timestamped. The [visualizer](#the-visualizer) plays this exact file back.
- The Docker image is built `FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (AMD's official ROCm PyTorch image), and the **build fails** unless the full test suite *and* a live Gemma generation pass inside that ROCm image. A `torch.version.hip` tripwire in the build aborts if anything ever replaces the ROCm torch build.
- [`demo.py`](demo.py) is a runtime ROCm smoke check: it prints torch's HIP build string and the visible AMD GPU name before running a gate forward pass on it.

## Quickstart (local, no Docker)

The fast path. CPU is fine — no GPU or Hugging Face account required.

**Prerequisites:** Python 3.11 or 3.12 (the versions this repo is verified on — 3.11 locally, 3.12 in the container), `pip`, `git`.

```bash
git clone https://github.com/Jonathan-Jesni/neuro-genesis-engine.git
cd neuro-genesis-engine
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt    # torch + pytest; CPU wheel is fine
python -m core.orchestrator
```

> `python -m core.orchestrator` (run from the repo root) is the reliable cross-platform form. Plain `python core/orchestrator.py` fails with `ModuleNotFoundError: No module named 'core'` unless the repo root is on the path first:
>
> ```bash
> PYTHONPATH=. python core/orchestrator.py            # bash / zsh
> ```
> ```powershell
> $env:PYTHONPATH = "."; python core\orchestrator.py  # PowerShell
> ```

**What you'll see:** a 300-step training run on a synthetic regression task that injects an out-of-distribution batch every 40 steps. The network starts with 4 experts. When the rolling z-score detector flags a loss spike, the orchestrator generates a new expert's source code, validates it through the foundry, and registers it live — the expert count grows mid-run. At the end it prints a summary:

```
=== run summary ===
steps          : 300
spikes seen    : 7
registrations  : 7
rejections     : 0
experts        : 4 -> 11 (len(experts)=11)
final task loss: 3.2372
log written to : orchestrator_run.jsonl
```

(This is the actual output of the template-generator run, which is seeded and deterministic; counts vary when the real Gemma generator is active. The run also writes `orchestrator_run.jsonl` — drag-drop it onto the visualizer — and a checkpoint `orchestrator_demo.ckpt` every 25 steps. On a CPU-only torch install you may see a one-line `Failed to initialize NumPy` warning from torch at startup; it's harmless — nothing here uses numpy.)

**Gemma vs. template generator:** the demo *prefers* the real LLM generator and falls back gracefully. If `transformers`/`accelerate` aren't installed or no Hugging Face token is available, it prints a one-line notice and uses the dependency-free template generator instead — so this Quickstart always runs. To use the real Gemma-2-2b-it locally: uncomment the optional block at the bottom of [`requirements.txt`](requirements.txt), `pip install` again, and put an `HF_TOKEN` with access to the gated [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) in your environment or a `.env` file.

**Interrupt and resume:** Ctrl+C at any point leaves a resumable checkpoint. Continue the same run with:

```bash
python resume_demo.py
```

## Running the tests

44 offline tests: 22 for the gate/foundry core, 19 for the orchestrator, 3 for the Gemma generator's prompt/extraction logic. None of them need a GPU, a network connection, or a Hugging Face account.

```bash
pytest tests/ -v        # expected: 44 passed, 1 skipped (the opt-in live test)
```

Every test file also runs standalone, without pytest:

```bash
python tests/test_dynamic_gating.py
python tests/test_orchestrator.py
python tests/test_gemma_generator.py
```

**Offline vs. live Gemma:** one additional opt-in test loads the real 2B-parameter Gemma and drives a generated expert through the actual foundry. It needs `transformers`/`accelerate` installed (the optional block in `requirements.txt`), an `HF_TOKEN` with access to the gated model, and a GPU (or an explicit CPU override):

```bash
NGEN_RUN_GEMMA_LIVE=1 pytest tests/test_gemma_generator.py -v                          # GPU
NGEN_RUN_GEMMA_LIVE=1 NGEN_GEMMA_DEVICE=cpu pytest tests/test_gemma_generator.py -v   # CPU (slow)
```

```powershell
# PowerShell equivalents
$env:NGEN_RUN_GEMMA_LIVE = "1"; pytest tests/test_gemma_generator.py -v
$env:NGEN_RUN_GEMMA_LIVE = "1"; $env:NGEN_GEMMA_DEVICE = "cpu"; pytest tests/test_gemma_generator.py -v
```

## Docker (optional — live Gemma baked in, offline at runtime)

Not required for the submission demo — the Quickstart above is self-sufficient. The container exists for full reproducibility on AMD hardware: it ships the ROCm torch stack *and* the Gemma weights in the image, so the self-expansion loop runs with the real LLM on a machine with no network access.

**Build** — requires Docker 23+ (BuildKit) and an `HF_TOKEN` with access to the gated [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it). The token is passed as a BuildKit secret only; it never lands in an image layer or `docker history`:

```bash
export HF_TOKEN=hf_...   # your Hugging Face token
docker build --secret id=hf_token,env=HF_TOKEN -t neuro-genesis:rocm-gemma .
```

The build downloads the ~5 GB model into an image layer, then **gates itself**: it runs the full test suite plus a live CPU Gemma generation inside the ROCm image and fails if either fails. Expect the live-Gemma step to take several minutes (2B-model generation on CPU — `docker build` never has GPU access on any host).

**Run the self-expanding demo on an AMD GPU** (device and group flags are required for ROCm):

```bash
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    neuro-genesis:rocm-gemma
```

**Run the tests instead** (same image, CPU is fine, works with no GPU flags):

```bash
docker run --rm neuro-genesis:rocm-gemma pytest tests/ -v
```

**Offline after build:** `HF_HUB_OFFLINE=1` is set in the image and Gemma loads from the baked cache. Prove it with no network at all:

```bash
docker run --rm --network none neuro-genesis:rocm-gemma pytest tests/ -v
```

If the GPU is invisible inside the container, the host's `render` group GID likely differs from the image's — pass the real one: `--group-add "$(getent group render | cut -d: -f3)"` (see the notes in the [`Dockerfile`](Dockerfile)).

## The visualizer

[`orchestrator_viz.html`](orchestrator_viz.html) is a self-contained, animated playback of a real training run: the expert network growing node by node, the live loss curve, and the spike → generate → validate → register pipeline firing in sequence.

- **Open it:** double-click the file (it carries an embedded snapshot of the AMD-hardware run, so `file://` works), or serve the repo root so it live-fetches the sibling log:

  ```bash
  python -m http.server 8000
  # then open http://localhost:8000/orchestrator_viz.html
  ```

- **Load your own run:** drag-drop any `orchestrator_run.jsonl` produced by the Quickstart demo onto the page.
- **Controls:** play/pause (Space), a scrub bar to seek anywhere in the run, and 0.5×/1×/2×/4× playback speed.

If you copy the HTML elsewhere, keep `orchestrator_run_amd_hardware.jsonl` next to it — the live fetch expects a sibling file (it falls back to the embedded snapshot otherwise).

## Project structure

```
core/
  orchestrator.py            TrainingOrchestrator: spike detection, self-correcting
                             LLM retry loop, checkpoint/resume, JSONL logging,
                             and the 300-step demo (__main__)
  moe/dynamic_gating.py      DynamicNoisyTopKGate, remap_optimizer_for_expansion,
                             ExpertFoundry — the hardened core
tests/                       44 offline tests + 1 opt-in live Gemma test;
                             pytest-compatible AND directly runnable
.agents/skills/              machine-readable usage contracts (SKILL.md) for the
                             gate, foundry, and optimizer remap — written for
                             LLM agents that modify this code
demo.py                      ROCm/GPU smoke check (HIP build + gate forward)
resume_demo.py               resume an interrupted demo from its checkpoint
orchestrator_viz.html        animated run playback (self-contained, no deps)
orchestrator_run_amd_hardware.jsonl   canonical AMD-GPU run log (evidence)
Dockerfile                   ROCm container; bakes Gemma; build-gated by tests
requirements.txt             local install (torch + pytest; Gemma deps optional)
requirements-docker.txt      container deps — deliberately excludes torch
```

## Architecture

Four components, each consuming the one below through a strict contract:

**[`DynamicNoisyTopKGate`](core/moe/dynamic_gating.py)** — a Shazeer-style noisy top-k router whose expert dimension can grow *while the network is training*. `expand(n)` uses copy-on-write on the router matrices, is thread-safe against concurrent forward passes, and preserves the routing of all existing experts exactly.

**[`remap_optimizer_for_expansion`](core/moe/dynamic_gating.py)** — when the gate grows, the Adam moments for the router parameters must move onto the new, larger tensors or momentum is silently lost. The remap is two-phase: validate and build everything fallibly first, then commit infallibly — a failure can never leave the optimizer half-mutated. Invoked via `gate.expand(n, optimizer=opt)`, never standalone.

**[`ExpertFoundry`](core/moe/dynamic_gating.py)** — turns a *string of PyTorch source* into a live, validated expert: AST screening, execution in a restricted namespace, instantiation and a smoke-test forward pass, then transactional registration — optimizer group first, expert append, gate expansion last, with full rollback on any failure. The gate never sees an expert that didn't survive validation.

**[`TrainingOrchestrator`](core/orchestrator.py)** — the autonomy loop. A rolling z-score detector flags loss spikes (it adapts to the observed noise level, so healthy early-training descent doesn't misfire), packages a `FailureContext`, and calls a pluggable generator — the real `GemmaExpertGenerator` or the dependency-free template. Rejected code goes back to the LLM with the exact rejection reason for a bounded number of self-correction attempts. Every event is logged to JSONL; checkpoints store generated experts *as source*, so `from_checkpoint` can rebuild classes that never existed in the codebase.

The `.agents/skills/*/SKILL.md` files are the authoritative usage contracts for the core components — read those before modifying call sites.

## Known limitations — honest notes

- **The foundry sandbox is accident-prevention, not a security boundary.** It blocks the obvious footguns (`open`, `exec`, `eval`, imports, dunder access), but CPython `exec` cannot be made adversarially safe — equivalent capabilities remain reachable through the injected `torch` module (e.g. `torch.save`). Generated code from *your own local Gemma* is the threat model here; truly untrusted source needs an external sandbox or human review. This is documented, deliberately, in [`dynamic_gating.py`](core/moe/dynamic_gating.py).
- **`torch.func.functional_call` is incompatible with the gate.** It substitutes parameters via raw `_parameters` dict writes, which bypass the gate's staleness tracking — the forward would keep reading the real parameters, not the substituted ones. Not "use with caution"; do not combine them.
- **Lock discipline is on the caller.** `gate.expand_lock` is a plain non-reentrant lock: hold it across forward→backward→step, release it before registration (the foundry re-acquires it inside `expand`; calling it while held deadlocks). The orchestrator gets this right — custom training loops must too. The contracts in `.agents/skills/` spell this out.
- **The demo task is synthetic.** A regression problem with scheduled out-of-distribution injections, chosen so loss spikes are reproducible and the full autonomy loop is observable in a 300-step CPU run. It demonstrates the machinery, not a benchmark result.
- **Spike detection has warm-up guards.** No detection until `min_history` samples exist, and the standard deviation is floored so a near-flat loss curve can't fire on numerical jitter — meaning very early or very quiet regimes intentionally don't trigger growth.
- **Single-node.** Expert growth is copy-on-write on one device; there is no distributed or sharded expansion story yet.

## License & credits

[MIT](LICENSE). Built for the AMD Developer Hackathon ACT II (Unicorn Track).

- Gating follows Shazeer et al., [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538) (2017).
- Expert generation uses [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) (gated model, subject to Google's Gemma license).
- Container base: AMD's official [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch) image.
