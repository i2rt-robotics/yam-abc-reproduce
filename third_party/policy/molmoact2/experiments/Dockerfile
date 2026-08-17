ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
ARG CUDA_VERSION_PATH=cu128
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.8.15
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        g++ \
        git \
        pkg-config \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        mesa-common-dev \
        libx11-dev \
        libxext-dev \
        libxrandr-dev \
        libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -o ~/miniconda.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh" && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Install uv and point to the conda python executable for uv operations.
ARG UV_VERSION
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON=/opt/conda/bin/python

RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y "ffmpeg>=6,<8" -c conda-forge && \
    uv pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install PyTorch core ecosystem.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.10.0
ARG TORCHAO_VERSION=0.16.0
ARG INSTALL_CHANNEL=whl
RUN uv pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} \
    torchao==${TORCHAO_VERSION} \
    torchvision \
    torchaudio

ENV TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0"

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ENV GROUPED_GEMM_CUTLASS="1"
ARG GROUPED_GEMM_SHA="f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
RUN uv pip install --no-build-isolation --no-cache-dir \
    "grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

# Install flash-attn.
ARG FLASH_ATTN_VERSION=2.8.3
RUN FLASH_ATTN_CUDA_ARCHS="80;90;100" \
    uv pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install flash-attn 3 for Hopper.
ARG FLASH_ATTN_SHA="fbe15683a881743c2625b931cc4abcc107b43154"
ARG FA3_MAX_JOBS=16
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout ${FLASH_ATTN_SHA} && \
    git submodule update --init --recursive --depth 1 && \
    cd hopper && \
    FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE \
    MAX_JOBS=${FA3_MAX_JOBS} \
    python setup.py install && \
    cd /app/build && rm -rf flash-attention

# Install flash-attn 4.
ARG FLASH_ATTN_4_VERSION=4.0.0b4
RUN uv pip install --no-cache-dir flash-attn-4==${FLASH_ATTN_4_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.8
RUN uv pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.6.4
RUN uv pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Install quack kernels if requested.
ARG QUACK_VERSION=""
RUN if [ -n "${QUACK_VERSION}" ]; then \
        uv pip install --no-build-isolation --no-cache-dir quack-kernels==${QUACK_VERSION}; \
    fi

# Install torchcodec.
ARG TORCH_CODEC_VERSION=0.10.0
RUN uv pip install --no-cache-dir torchcodec==${TORCH_CODEC_VERSION}

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY olmo/__init__.py olmo/__init__.py
COPY olmo/version.py olmo/version.py
RUN uv pip install --no-cache-dir '.[all]' && \
    uv pip uninstall ai2-molmo && \
    rm -rf *

# Install vLLM runtime stack.
ARG VLLM_VERSION=0.17.0
RUN uv pip install --no-cache-dir --pre \
    vllm==${VLLM_VERSION} \
    --extra-index-url https://wheels.vllm.ai/nightly/${CUDA_VERSION_PATH}

RUN uv pip install --no-cache-dir webdataset fasttext-numpy2 inflect

# Install molmo-utils from PyPI with torchcodec support.
RUN uv pip install --no-cache-dir 'molmo-utils[torchcodec]'

# Install coco caption eval dependencies.
RUN uv pip install --no-cache-dir pycocoevalcap && \
    /opt/conda/bin/conda install -y conda-forge::openjdk && \
    /opt/conda/bin/conda clean -ya

RUN uv pip install --no-cache-dir torch-c-dlpack-ext

ARG FLASH_LINEAR_ATTENTION_VERSION=0.4.1
RUN uv pip install --no-cache-dir flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}

ARG CAUSAL_CONV1D_VERSION=1.6.0
RUN uv pip install --no-cache-dir causal-conv1d==${CAUSAL_CONV1D_VERSION}

# Install lerobot.
ARG LEROBOT_VERSION=0.5.0
RUN uv pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    "cmake>=3.29.0.1,<4.0.0" && \
    cmake --version

RUN uv pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
    --index-strategy unsafe-best-match \
    "lerobot[all]==${LEROBOT_VERSION}"

# Re-pin the final runtime stack after LeRobot resolves its own dependencies.
RUN uv pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} \
    torchao==${TORCHAO_VERSION} \
    torchvision \
    torchaudio && \
    uv pip install --no-cache-dir --force-reinstall --no-deps \
    --index-url https://pypi.org/simple \
    torchcodec==${TORCH_CODEC_VERSION}

#########################################################################
# Release image
#########################################################################

FROM ${BASE_IMAGE} AS release

ARG CUDA_VERSION

# Install CUDA JIT compilation tools for torch.compile/Triton.
RUN CUDA_VERSION_DASH=$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr '.' '-') && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-dev-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-${CUDA_VERSION_DASH} \
        libcublas-dev-${CUDA_VERSION_DASH} \
        libnccl-dev \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        language-pack-en \
        make \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        rsync \
        vim \
        sudo \
        unzip \
        fish \
        parallel \
        zsh \
        htop \
        tmux \
        wget \
        emacs \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        apt-transport-https \
        gnupg \
        jq \
        gcc \
        git \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libglx-mesa0 \
        libglu1-mesa \
        libosmesa6 \
        libosmesa6-dev \
        patchelf \
        xvfb \
        x11-utils && \
    rm -rf /var/lib/apt/lists/*

# Install uv and point to the conda python executable for uv operations.
ARG UV_VERSION
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Install DOCA OFED user-space drivers
# See https://docs.nvidia.com/doca/sdk/doca-host+installation+and+upgrade/index.html
# doca-ofed-userspace ver 2.10.0 depends on mft=4.31.0-149
ENV MFT_VER=4.31.0-149
RUN wget https://www.mellanox.com/downloads/MFT/mft-${MFT_VER}-x86_64-deb.tgz && \
    tar -xzf mft-${MFT_VER}-x86_64-deb.tgz && \
    mft-${MFT_VER}-x86_64-deb/install.sh --without-kernel && \
    rm -rf mft-${MFT_VER}-x86_64-deb*

ENV DOFED_VER=2.10.0
ENV OS_VER=ubuntu2404
RUN wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOFED_VER}/host/doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    dpkg -i doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    apt-get update && apt-get -y install doca-ofed-userspace && \
    rm doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

ENV UV_PYTHON=/opt/conda/bin/python
ENV PATH=/opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# gsutil/gcloud
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get -y install google-cloud-cli

# Install a few additional utilities via pip.
RUN uv pip install --no-cache-dir \
    gpustat \
    jupyter

# Setup env for libero eval
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Use bash for RUN
SHELL ["/bin/bash", "-c"]

# Install Beaker CLI
RUN set -e && \
    VERSION=$(curl --header 'Accept: text/plain' --silent https://beaker.org/api/v3/release) && \
    TMP=$(mktemp -d) && \
    ARCH=$(uname -m) && \
    OS=$(uname -s) && \
    if [ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]; then ARCH=arm64; OS=darwin; \
    elif [ "$ARCH" = "x86_64" ] && [ "$OS" = "Darwin" ]; then ARCH=amd64; OS=darwin; \
    elif [ "$ARCH" = "x86_64" ] && [ "$OS" = "Linux" ]; then ARCH=amd64; OS=linux; \
    else echo "Unrecognized OS-Architecture combination: ARCH=$ARCH OS=$OS"; exit 1; fi && \
    PATTERN="beaker-cli-$OS-$ARCH-$VERSION.tar.gz" && \
    mkdir -p $TMP/assets && \
    URL="https://beaker.org/api/v3/release/cli?os=$OS&arch=$ARCH" && \
    HTTP_CODE=$(curl --retry 25 --retry-delay 1 --max-time 10 --retry-max-time 10 --silent --output $TMP/assets/$PATTERN --write-out "%{http_code}" $URL) && \
    if [ "$HTTP_CODE" -ne "200" ]; then echo "Failed to download: $URL"; exit 1; fi && \
    echo "Download succeeded; size is $(wc -c < $TMP/assets/$PATTERN) bytes." && \
    tar -zxf $TMP/assets/$PATTERN -C /usr/local/bin ./beaker && \
    chmod +x /usr/local/bin/beaker && \
    beaker --version

# LABEL org.opencontainers.image.source https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core
