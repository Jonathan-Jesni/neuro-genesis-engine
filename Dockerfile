# =============================================================================
# Neuro-Genesis Engine -- production container for AMD Developer Cloud (ROCm)
# =============================================================================
# Target: AMD Instinct / Radeon GPUs via ROCm (NOT CUDA/NVIDIA).
#
# Design in one sentence: start from AMD's official ROCm PyTorch image (which
# already contains a torch built against a matching ROCm version), add only the
# non-torch Python deps, copy the source, and fail the build if the test suite
# does not pass INSIDE the ROCm image.
#
# Build (requires an HF token for the gated gemma-2 download -- passed as a
# BuildKit secret so it NEVER lands in an image layer; BuildKit is the default
# engine in Docker 23+):
#   export HF_TOKEN=hf_...            # or set it in the shell from .env
#   docker build --secret id=hf_token,env=HF_TOKEN -t neuro-genesis:rocm .
#
# The image bakes google/gemma-2-2b-it into an image layer at build time and
# sets HF_HUB_OFFLINE=1, so the built container needs NO network access at
# runtime (verify with: docker run --rm --network none ... pytest ...).
#
# Run the demo on GPU (device + group flags are REQUIRED for ROCm -- see the
# non-root user note further down):
#   docker run --rm \
#       --device=/dev/kfd --device=/dev/dri \
#       --group-add video --group-add render \
#       --security-opt seccomp=unconfined \
#       neuro-genesis:rocm
#
# Run the test suite instead of the demo (same image, override CMD):
#   docker run --rm neuro-genesis:rocm pytest tests/ -v
# =============================================================================

# --- Base image -------------------------------------------------------------
# We use AMD's official ROCm PyTorch image rather than installing ROCm onto a
# generic Ubuntu base. Reasons: (1) the torch build and the ROCm userspace in
# these images are a known-good, AMD-tested pairing -- getting that pairing
# right by hand is the single biggest source of ROCm setup pain; (2) far faster
# builds (no compiling ROCm from source). Because torch already lives in this
# image, we NEVER pip-install torch below -- doing so risks overwriting the
# GPU-enabled ROCm wheel with a CPU/CUDA one from PyPI.
#
# The tag is an ARG so bumping it to whatever AMD Developer Cloud actually ships
# is a one-line change (or `--build-arg ROCM_PYTORCH_TAG=...` at build time).
#
# !! VERIFY THIS TAG on AMD Developer Cloud before the final submission build.
#    The exact `rocmX.Y_ubuntuAA.BB_pyA.BB_pytorch_release_A.B.C` tag string and
#    the ROCm version it pins MUST match what the cloud environment supports;
#    tags on Docker Hub (hub.docker.com/r/rocm/pytorch/tags) change over time.
ARG ROCM_PYTORCH_TAG=rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
FROM rocm/pytorch:${ROCM_PYTORCH_TAG}

# --- Environment ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # /app on the path so the package's absolute imports (`from core.moe ...`)
    # resolve without an editable install.
    PYTHONPATH=/app \
    # Single point of control for the entrypoint (override with `-e DEMO=...`).
    # Default is the Gemma-backed orchestrator demo: its __main__ prefers
    # GemmaExpertGenerator (deps + weights are baked into this image) and
    # falls back to the template generator only if construction fails. The
    # GPU smoke check remains available via `-e DEMO=demo.py`.
    DEMO=core/orchestrator.py

WORKDIR /app

# --- Non-root user ----------------------------------------------------------
# Container hygiene: never run the workload as root.
#
# ROCm gotcha a future reader needs to know: GPU access is gated by membership
# in the 'video' and (on newer kernels) 'render' groups, which own /dev/kfd and
# /dev/dri/renderD*. A non-root user WITHOUT those groups cannot see the GPU
# even when the devices are passed in with --device. We add the user to both.
#
# Caveat that cannot be fully solved in the image: the numeric GID of 'render'
# must match the HOST's render GID at runtime, and that GID is host-specific.
# If the GPU is invisible as non-root on AMD Developer Cloud, pass the host's
# real GID at `docker run` time, e.g.:
#     --group-add "$(getent group render | cut -d: -f3)" --group-add video
# (the run command in the header already includes --group-add render/video).
RUN groupadd --system render 2>/dev/null || true; \
    groupadd --system video  2>/dev/null || true; \
    useradd --create-home --shell /bin/bash ngen; \
    usermod -aG video,render ngen

# WORKDIR above created /app while we were still root, so it's root-owned.
# COPY --chown further down sets ownership on the FILES we copy in, but not
# on the /app directory itself -- so without this, ngen can write into
# existing files but cannot create new subdirectories under /app. This
# surfaces at runtime as pytest failing to create .pytest_cache with
# "Permission denied" (harmless warning, but worth eliminating cleanly).
RUN chown ngen:ngen /app

# --- Python dependencies (cached layer -- deps before source) ---------------
# Copy ONLY the non-torch lock file first so this expensive layer is cached and
# reused across source-only changes. torch is intentionally absent from this
# file; see requirements-docker.txt for the full rationale. If you ever need a
# ROCm torch installed explicitly (generic base, not this one), it must be its
# OWN step against the ROCm wheel index, e.g.:
#     pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
# and NEVER be resolved by a generic `-r requirements.txt`.
COPY --chown=ngen:ngen requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

# Tripwire: fail the build IMMEDIATELY if the pip install above (transformers/
# accelerate pull in many deps) clobbered the ROCm torch with a CUDA/CPU wheel.
# torch.version.hip is only populated on ROCm builds.
RUN python -c "import torch, transformers, accelerate; \
    assert torch.version.hip, 'ROCm torch was clobbered by a non-HIP build!'; \
    print('torch', torch.__version__, 'hip', torch.version.hip, \
          '| transformers', transformers.__version__, \
          '| accelerate', accelerate.__version__)"

# --- Bake gemma-2-2b-it into the image (build-time, offline runtime) ---------
# Placed BEFORE the source COPY so code-only changes reuse this ~5.5 GB layer.
# HF_HOME points the cache at a fixed, ngen-readable path used at runtime too.
#
# The HF token (gemma-2 is a gated model) comes in as a BuildKit SECRET mount:
# it exists only as a file during this single RUN step and is never written to
# an ENV/ARG/layer -- `docker history` must never show a token value.
# allow_patterns keeps the layer lean: config + safetensors + tokenizer only.
ENV HF_HOME=/opt/hf-cache
RUN --mount=type=secret,id=hf_token \
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download('google/gemma-2-2b-it', \
    token=open('/run/secrets/hf_token').read().strip(), \
    allow_patterns=['*.json','*.safetensors','tokenizer*','*.model'])" \
    && chown -R ngen:ngen /opt/hf-cache

# From here on -- including the build-time test gate below and all runtime --
# the HF stack is OFFLINE: loads must come from the baked cache, and any
# accidental network dependence fails loudly instead of silently downloading.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- Source (changes most often -> last, so deps stay cached) ---------------
COPY --chown=ngen:ngen core/ ./core/
COPY --chown=ngen:ngen tests/ ./tests/
COPY --chown=ngen:ngen demo.py ./
# When the real demo ships with frontend assets, add e.g.:
#   COPY --chown=ngen:ngen web/ ./web/

# Drop privileges for everything from here on (build gate + runtime).
USER ngen

# --- Build-time ROCm compatibility gate -------------------------------------
# This is the actual verification that the ROCm PyTorch stack works for THIS
# code -- not a local CUDA test. If the suite does not go green inside the ROCm
# image, the build FAILS here and no broken image is produced. The suite is
# CPU-runnable (GPU-specific paths are guarded), so this gate passes even on a
# build host without a GPU; genuine GPU execution is exercised at run time by
# demo.py. We also print torch's HIP build string as a breadcrumb in build logs.
RUN python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)" \
    && pytest tests/ -v

# --- Build-time LIVE Gemma gate ----------------------------------------------
# The point of baking the model in: prove the REAL generator works inside this
# image -- load gemma-2-2b-it from the baked cache (offline env is already
# set), generate expert source, and register it through the real foundry.
# Runs on CPU because `docker build` never has GPU access (on any host); the
# same generator's GPU path is exercised at runtime on the AMD box. Expect
# this step to take several minutes (2B-model CPU generation).
RUN NGEN_RUN_GEMMA_LIVE=1 NGEN_GEMMA_DEVICE=cpu pytest tests/test_gemma_generator.py -v

# --- Entrypoint -------------------------------------------------------------
# Runs the demo by default. Overriding the demo is a one-line change: edit the
# DEMO env above, or override CMD at runtime. Running the tests instead is just
# `docker run <image> pytest tests/ -v` (CMD is fully overridable).
#
# Shell form is used deliberately so ${DEMO} is expanded at container start,
# which is what makes the entrypoint a single configurable line.
CMD ["sh", "-c", "python ${DEMO}"]