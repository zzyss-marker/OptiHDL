# ── OptiHDL Docker Image ──
# Complete EDA environment: Yosys + OpenSTA + Node.js (DigitalJS)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# ── Stage 1: System deps + Yosys + Node.js ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    tcl-dev \
    libreadline-dev \
    libffi-dev \
    graphviz \
    xdot \
    yosys \
    # OpenSTA build dependencies
    swig \
    bison \
    flex \
    libeigen3-dev \
    libfl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN yosys -V

# ── Stage 2: Node.js 20 LTS (for yosys2digitaljs) ──
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN node -v && npm -v

# ── Stage 3: Build OpenSTA from source ──
RUN git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenSTA.git /tmp/OpenSTA \
    && mkdir /tmp/OpenSTA/build \
    && cd /tmp/OpenSTA/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/OpenSTA

# Verify EDA tools
RUN yosys -V && sta -version

# ── Stage 4: Python environment ──
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Stage 5: DigitalJS bridge (Node.js deps) ──
COPY tools/package.json /app/tools/
RUN cd /app/tools && npm install --production

# ── Stage 6: Application ──
COPY . /app/

RUN mkdir -p /app/models /app/data /app/logs /app/outputs /app/temp

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "web_app.app:create_app()"]
