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
# Build:
#   docker build -t neuro-genesis:rocm .
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
    # Single point of control for the entrypoint: swapping in the real Day-5
    # demo is one line here (or a `-e DEMO=...` at runtime). See CMD at bottom.
    DEMO=demo.py

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

# --- Entrypoint -------------------------------------------------------------
# Runs the demo by default. Overriding the demo is a one-line change: edit the
# DEMO env above, or override CMD at runtime. Running the tests instead is just
# `docker run <image> pytest tests/ -v` (CMD is fully overridable).
#
# Shell form is used deliberately so ${DEMO} is expanded at container start,
# which is what makes the entrypoint a single configurable line.
CMD ["sh", "-c", "python ${DEMO}"]