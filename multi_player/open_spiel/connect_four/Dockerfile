# Base image: NVIDIA CUDA 12.1 Development (required for C++ compilation)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent Python output buffering and tzdata interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# -----------------------------
# 1. Install System Dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
        curl \
        git \
        cmake \
        clang \
        wget \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# -----------------------------
# 2. Install ML/RL Dependencies (JAX ecosystem for AlphaZero)
# -----------------------------
# Install JAX with CUDA 12 support
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install neural network libraries used by OpenSpiel's AlphaZero
RUN pip install flax optax chex tensorboard

# -----------------------------
# 3. Clone and Build OpenSpiel
# -----------------------------
# Tell setup.py to use clang++ for compiling the C++ extensions
ENV CXX=clang++

# Clone the repository natively
RUN git clone https://github.com/google-deepmind/open_spiel.git /opt/open_spiel

WORKDIR /opt/open_spiel

# FIX: Run OpenSpiel's native install script to fetch abseil-cpp and other C++ deps
RUN ./install.sh

# Install OpenSpiel's python dependencies
RUN pip install -r requirements.txt

# Build and install OpenSpiel (this triggers CMake under the hood)
RUN pip install -e .

# -----------------------------
# 4. Setup Workspace
# -----------------------------
WORKDIR /workspace
RUN pip install --upgrade "jax[cuda12]"

# -----------------------------
# 5. Expose necessary ports
# -----------------------------
EXPOSE 6006-6010


# docker build -t rl-games:open_spiel_c4 .
# docker tag rl-games:open_spiel_c4 gg8docker0/rl-games:open_spiel_c4
# docker push gg8docker0/rl-games:open_spiel_c4
# docker pull gg8docker0/rl-games:open_spiel_c4
# docker run -it --gpus all -p 6003:6003 -v $(pwd):/workspace gg8docker0/rl-games:open_spiel_c4