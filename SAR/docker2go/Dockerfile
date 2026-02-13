ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION

WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (common for all model trainers) TODO use venv instead of system wide install
COPY requirements.txt .
# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Set Python path
ENV PYTHONPATH=/app

# Copy application code
# COPY . /app

# Default command
CMD ["python", "-c", "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"]