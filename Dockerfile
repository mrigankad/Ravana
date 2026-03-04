# Ravana — Docker Deployment
# Multi-stage build: compile native + install Python deps + slim runtime

# ── Stage 1: Build native C++ library ──────────────────────────────────
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS native-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY face_swap/native/ ./native/

RUN cd native && \
    cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_OPENCV=ON \
    -DWITH_ONNXRUNTIME=OFF && \
    cmake --build build --config Release

# ── Stage 2: Python dependencies ───────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS python-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Stage 3: Final runtime image ──────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=python-deps /usr/local/lib/python3.10/dist-packages/ \
    /usr/local/lib/python3.10/dist-packages/
COPY --from=python-deps /usr/local/bin/ /usr/local/bin/

# Copy native library
COPY --from=native-builder /build/native/build/libface_swap.so \
    /usr/local/lib/libface_swap.so
RUN ldconfig

# Copy application code
WORKDIR /app
COPY . .

# Create model directory
RUN mkdir -p /app/models

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default: run the CLI
ENTRYPOINT ["python3", "-m", "demos.cli"]
CMD ["--help"]

# ── Usage ──────────────────────────────────────────────────────────────
# Build:
#   docker build -t face-swap .
#
# Run image swap:
#   docker run --gpus all -v $(pwd)/data:/data face-swap \
#       --mode image --source /data/source.jpg --target /data/target.jpg \
#       --output /data/output.jpg
#
# Run video swap:
#   docker run --gpus all -v $(pwd)/data:/data face-swap \
#       --mode video --source /data/source.jpg --target /data/input.mp4 \
#       --output /data/output.mp4
#
# Interactive shell:
#   docker run --gpus all -it face-swap bash
