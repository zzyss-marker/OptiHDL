# ── OptiHDL Docker Image ──
# Complete EDA environment: Yosys + OpenSTA + Node.js (DigitalJS)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# ── 1. System packages (Yosys + all OpenSTA build deps) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    graphviz \
    xdot \
    yosys \
    # OpenSTA requires all of these
    tcl-dev \
    swig \
    bison \
    flex \
    libeigen3-dev \
    libreadline-dev \
    libffi-dev \
    libfl-dev \
    libgtest-dev \
    zlib1g-dev \
    libfmt-dev \
    # CUDD autotools build
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

RUN yosys -V

# ── 2. Node.js 20 LTS (for yosys2digitaljs) ──
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── 3. Build GTest (Ubuntu 22.04 libgtest-dev ships source only) ──
RUN cd /usr/src/googletest && cmake . && make && make install

# ── 4. Build CUDD 3.0.0 (autotools, no apt package) ──
RUN git clone --depth 1 https://github.com/The-OpenROAD-Project/cudd.git /tmp/cudd \
    && cd /tmp/cudd \
    && autoreconf -fi \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/cudd

# ── 5. Build OpenSTA from source ──
RUN git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenSTA.git /tmp/OpenSTA \
    && mkdir /tmp/OpenSTA/build \
    && cd /tmp/OpenSTA/build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCUDD_DIR=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/OpenSTA

# Verify all EDA tools
RUN yosys -V && sta -version && node -v

# ── 6. Python environment ──
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 7. DigitalJS bridge (Node.js deps) ──
COPY tools/package.json /app/tools/
RUN cd /app/tools && npm install --production

# ── 8. Application ──
COPY . /app/

RUN mkdir -p /app/models /app/data /app/logs /app/outputs /app/temp

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "web_app.app:create_app()"]
