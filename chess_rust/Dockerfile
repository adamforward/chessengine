# Stage 1: Build Stage
FROM rust:latest AS build

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    clang \
    llvm \
    python3 \
    python3-pip \
    python3-venv \
    libomp-dev \
    libopenblas-dev \
    liblapack-dev \
    libomp5 \
    libnuma-dev \
    gfortran \
    gcc \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python YAML module
RUN apt update && apt install -y python3-pip python3-yaml

# Set working directory
WORKDIR /app/pytorch-build

# Clone PyTorch and set up the build environment
RUN git clone -b v2.5.0 --recurse-submodules https://github.com/pytorch/pytorch.git /app/pytorch && \
    mkdir -p /app/pytorch-build

# Additional Python dependencies for PyTorch
RUN apt update && apt install -y python3-dev python3-pybind11 python3-typing-extensions

# Build LibTorch
RUN cmake -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_INSTALL_PREFIX=/app/libtorch \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCPU_CAPABILITY=SSE2 \
    -DUSE_OPENMP=ON \
    -DUSE_NCCL=OFF \
    -DUSE_MKLDNN=OFF \
    -DUSE_BLAS=OpenBLAS \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/llvm-14/lib -liomp5" \
    /app/pytorch && \
    cmake --build . --target install

# Set environment variables for LibTorch
ENV LIBTORCH=/app/libtorch
ENV LD_LIBRARY_PATH=/app/libtorch/lib:$LD_LIBRARY_PATH

# Pre-cache Rust dependencies
WORKDIR /app
COPY ./Cargo.toml /app/Cargo.toml

# Copy application source code
COPY ./build.rs /app/build.rs
COPY ./src /app/src

# Build the Rust application
RUN cargo build --release
RUN ldconfig
# Expose application ports (e.g., 8080 and 3000)
EXPOSE 8080
EXPOSE 3000

# Start an interactive shell for debugging or allow running cargo commands
CMD ["bash"]
