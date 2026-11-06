# LeRobot VLA training image for Amazon SageMaker Training Jobs (and HyperPod EKS).
#
# Base: official NVIDIA CUDA 13.0 image (latest 13.0.x; matches the torch cu130 wheel
# toolkit). Covers Ampere/Ada/Hopper/Blackwell (sm_120, i.e. ml.g7e RTX PRO 6000).
# LeRobot's default pin is cu128; cu130 is its documented Blackwell variant. We do not
# start from the NGC PyTorch container because it ships pre-release torch builds that
# do not match LeRobot's torch/torchcodec pins.
#
# The image contains only the runtime (LeRobot + policy extras). The training launcher
# (src/) is synced into the job at runtime by the SageMaker SDK v3 SourceCode object,
# so you rebuild this image only when you change LeRobot or CUDA versions.
#
#   docker build -t lerobot-vla:0.6.1 -f docker/Dockerfile .   (context = training-jobs/)

ARG CUDA_VERSION=13.0.3
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ARG LEROBOT_VERSION=0.6.1
# Policy extras to bake in. groot = NVIDIA GR00T N1.7, pi = pi0/pi0.5.
ARG LEROBOT_EXTRAS="groot,pi,peft,training"
# PyTorch wheel index. cu130 = CUDA 13 (LeRobot's documented Blackwell variant, driver >= 580,
# which SageMaker training hosts have: the AWS DLC pytorch-training 2.9 image is cu130). Use cu128
# for hosts on driver 570.
ARG TORCH_CUDA_INDEX=cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # LeRobot / HF caches on the (large) SageMaker volume rather than the root fs
    HF_HOME=/opt/ml/cache/huggingface \
    HF_LEROBOT_HOME=/opt/ml/cache/huggingface/lerobot \
    TORCH_HOME=/opt/ml/cache/torch \
    TRITON_CACHE_DIR=/opt/ml/cache/triton \
    # Headless rendering default (harmless for training, needed by any MuJoCo-based eval)
    MUJOCO_GL=egl \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 24.04 ships Python 3.12, which is what LeRobot >= 0.5 requires.
# ffmpeg + libav* are needed by torchcodec / pyav for LeRobotDataset video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates jq \
        build-essential cmake pkg-config ninja-build \
        ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev \
        libgl1 libglib2.0-0 libegl1 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Isolated venv; torch/torchvision from the cu128 index (LeRobot's pinned variant).
# --seed installs pip into the venv: the SageMaker SDK v3 script-mode driver requires pip3 on PATH.
RUN uv venv --seed /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX} \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}"

# ---------------------------------------------------------------------------
# AWS EFA userspace stack (libfabric + AWS-OFI-NCCL plugin) so NCCL can use
# EFA/RDMA for multi-node jobs on EFA-capable instances (p4d, p5, p6,
# g6e/g7e .48xlarge). --skip-kmod relies on the kernel module already being on
# the SageMaker / HyperPod host. On instances without EFA these libraries are
# inert and NCCL falls back to TCP, so the single-node path is unaffected.
# Same block as awsome-distributed-ai/examples/use-cases/isaac-lab.
# ---------------------------------------------------------------------------
ARG EFA_INSTALLER_VERSION=1.50.0
RUN apt-get update && cd /tmp \
    && curl -sL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && cd / && rm -rf /tmp/aws-efa-installer && apt-get clean && rm -rf /var/lib/apt/lists/*
# NCCL plugin discovery. The installer ships libnccl-net-ofi.so; NCCL (2.28 in the torch cu130
# wheels) dlopens "libnccl-net.so" by default (or libnccl-net-<NCCL_NET_PLUGIN>.so). Provide both
# names, and make the lib dir visible to the dynamic loader. Without this NCCL
# logs "NET/Plugin: Could not find: libnccl-net.so" and silently uses Socket on EFA nodes.
RUN ln -sf /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so /opt/amazon/ofi-nccl/lib/libnccl-net.so \
    && ln -sf /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so /opt/amazon/ofi-nccl/lib/libnccl-net-aws-ofi.so \
    && ln -sf /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so /opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl-net.so \
    && echo "/opt/amazon/ofi-nccl/lib" > /etc/ld.so.conf.d/aws-ofi-nccl.conf \
    && echo "/opt/amazon/efa/lib"      > /etc/ld.so.conf.d/efa.conf \
    && ldconfig
ENV NCCL_NET_PLUGIN=ofi
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}" \
    PATH="/opt/amazon/efa/bin:${PATH}"

# Defaults shared by SageMaker Training Jobs and HyperPod. The launcher (src/) is baked in at
# /opt/ml/code for HyperPod PyTorchJobs; on Training Jobs the SDK v3 SourceCode sync to
# /opt/ml/input/data/code takes precedence, so editing src/ never requires a rebuild there.
RUN mkdir -p /opt/ml/cache /opt/ml/checkpoints /opt/ml/model /opt/ml/code
COPY src/ /opt/ml/code/
RUN chmod +x /opt/ml/code/train.sh
WORKDIR /opt/ml/code

# Smoke test at build time: versions, policy imports, CLI, and EFA plugin presence.
RUN python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('lerobot', lerobot.__version__)" \
    && python -c "import lerobot.policies.groot, lerobot.policies.pi05" \
    && lerobot-train --help > /dev/null 2>&1 \
    && command -v pip3 && command -v python3 \
    && test -f /opt/amazon/ofi-nccl/lib/libnccl-net.so \
    && /opt/amazon/efa/bin/fi_info --version

CMD ["/bin/bash"]
