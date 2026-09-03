# Training container: veRL's official image plus the SageMaker /opt/ml contract.
#
# Build context is the project root, not this directory, because the image needs
# both container/ and scripts/:
#
#     docker build -f container/Dockerfile --build-arg BASE_DIGEST=<sha256 hex> .
#
# In practice `infrastructure/buildspec.yml` runs the build in CodeBuild and
# supplies BASE_DIGEST from src/grpo_sagemaker/build_image.py's resolution step.
# No local Docker daemon is involved (Requirement 6.9).

# BASE_DIGEST has no default on purpose. The base image must be referenced by
# immutable content digest, never by a mutable tag (Requirement 6.3), and the
# digest comes from container/base-image.lock.json at build time rather than being
# hardcoded here, so the pin lives in one reviewable place.
#
# BASE_DIGEST carries the full `sha256:<hex>` form, not the bare hex. That is the
# convention everywhere in this project: `build_image._require_digest` enforces the
# prefix and `ResolvedBaseImage.reference` emits it. So the FROM line must not add
# a second `sha256:` -- doing so yields `verlai/verl@sha256:sha256:<hex>`, which
# Docker rejects as an invalid reference. Omitting the argument entirely yields
# `verlai/verl@` and Docker still refuses, which is the correct fail-closed
# behaviour.
ARG BASE_DIGEST

FROM verlai/verl@${BASE_DIGEST}

# Re-declared after FROM: an ARG from the pre-FROM scope is not in the build
# stage's scope. Recorded as a label so a pulled image can be traced back to its
# base without consulting the pinning matrix.
ARG BASE_DIGEST
LABEL com.amazonaws.grpo-verl-sagemaker.base-image="verlai/verl@${BASE_DIGEST}"

# PYTHONUNBUFFERED keeps training output in CloudWatch in real time rather than
# arriving in blocks, or being lost entirely if the job is killed.
# PYTHONDONTWRITEBYTECODE avoids .pyc writes into a read-only-by-convention
# /opt/ml/code. SAGEMAKER_PROGRAM names the entry point the sagemaker-training
# toolkit executes from /opt/ml/code (Requirement 7 contract).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGEMAKER_PROGRAM=entrypoint.py

# Declare the contract layout the container writes to. SageMaker mounts over most
# of these at runtime; creating them here means verify_stack.py can assert the
# layout at build time, and a checkpoint or model write can never fail on a
# missing parent directory mid-run.
RUN mkdir -p /opt/ml/code \
             /opt/ml/input/config \
             /opt/ml/input/data \
             /opt/ml/checkpoints \
             /opt/ml/model \
             /opt/ml/output/data

# Copied and installed before the source, so editing scripts/ does not
# invalidate this layer.
#
# This step installs two things: veRL itself, and the sagemaker-training toolkit
# with its runtime closure. veRL is not optional here -- the base image's own
# final step is `pip install git+...verl.git@v0.7.1 && pip uninstall -y verl`, so
# it ships veRL's dependency closure but not veRL, and `import verl` fails in the
# unmodified base. See container/requirements-container.txt.
#
# --no-deps is load-bearing (Requirement 6.6): the base image's CUDA, torch,
# vLLM, and transformers versions must survive this install untouched, and
# letting pip resolve dependencies is how torch gets silently upgraded. The
# requirements file therefore carries every needed pin exactly and explains what
# it deliberately omits.
# The base image's package inventory is printed before the install, and this is
# not idle logging. `--no-deps` means this file must carry every dependency the
# toolkit imports -- but only the ones the base image does *not* already provide.
# Naming a package the base image installed at a different version makes pip try
# to replace it, and when the existing copy has no RECORD file pip cannot
# uninstall it and the build fails. Having the inventory in the log turns that
# class of failure into a single readable diff instead of one build per package.
COPY container/requirements-container.txt /tmp/requirements-container.txt
RUN echo "=== base image packages, before this layer ===" \
    && pip list --format=freeze 2>/dev/null | sort \
    && echo "=== installing the extension layer ===" \
    && pip install --no-cache-dir --no-deps -r /tmp/requirements-container.txt \
    && rm /tmp/requirements-container.txt

# The in-container training scripts. `scripts/` is self-contained by design: it
# imports nothing from the notebooks and reads its configuration only from the
# hyperparameters.json SageMaker writes, which is what lets the same files run
# either baked into this image or uploaded per job as SourceCode.
COPY scripts/ /opt/ml/code/
COPY container/verify_stack.py /opt/ml/code/verify_stack.py

# Overrides the base image's WorkingDir (/home/dpsk_a2a).
#
# ENTRYPOINT is deliberately left as the base image's
# ['/opt/nvidia/nvidia_entrypoint.sh'], which execs its arguments after doing
# NVIDIA device setup. SageMaker starts a training container as
# `docker run <image> train`, so the toolkit's `train` console script is still
# what runs. Replacing the entrypoint would skip that device setup. Task 12.2's
# first real build is what confirms this holds in practice.
WORKDIR /opt/ml/code

# Build-time gate (Requirements 6.7, 6.8): import verl, ray, vllm, and torch,
# assert vLLM >= 0.18.0, Python >= 3.10, and CUDA >= 12.8, and assert the
# /opt/ml layout and the sagemaker-training entry point are in place. A
# violation fails this build instead of a paid training job, and the printed
# report is what populates the pinning matrix in docs/container.md.
RUN python /opt/ml/code/verify_stack.py
