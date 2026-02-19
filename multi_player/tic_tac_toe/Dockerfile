# Base image: PyTorch 2.4 + CUDA 12.1 runtime
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# -----------------------------
# 1. Install Python 3.10 + pip + dev tools
# -----------------------------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-distutils \
        python3.10-venv \
        python3-pip \
        git \
        gcc \
        g++ \
        swig \
        libgl1 \
        libglib2.0-0 \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip, setuptools, wheel
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# 2. Install RL / ML dependencies
# -----------------------------
RUN pip install \
    "pettingzoo[classic]" \
    "gymnasium" \
    "numpy>=1.24.1,<2" \
    "transformers" \
    "pympler" \
    "minigrid" \
    "moviepy" \
    "pytest" \
    "line_profiler" \
    "xxhash" \
    "einops" \
    "openai" \
    "nltk" \
    "pyecharts" \
    "numba" \
    "simple_parsing"

# -----------------------------
# 3. Clone LightZero and install editable
# -----------------------------
RUN git clone https://github.com/opendilab/LightZero.git /opt/LightZero
RUN pip install -e /opt/LightZero

# -----------------------------
# 4. Expose necessary ports
# -----------------------------
EXPOSE 6000-6010


# docker build -t rl-games:tictactoe .
# docker tag rl-games:tictactoe gg8docker0/rl-games:tictactoe
# docker push gg8docker0/rl-games:tictactoe
# docker pull gg8docker0/rl-games:tictactoe
# docker run -it --gpus all -p 6000:6000 -v $(pwd):/workspace gg8docker0/rl-games:tictactoe