# ── OptiHDL Docker Image ──
# EDA environment: Yosys + Node.js (DigitalJS)
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── 1. System packages (Yosys) ──
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
    tcl-dev \
    libreadline-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN yosys -V

# ── 2. Node.js 20 LTS (for yosys2digitaljs) ──
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verify tools
RUN yosys -V && node -v

# ── 3. Python environment ──
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 4. DigitalJS bridge (Node.js deps) ──
COPY tools/package.json /app/tools/
RUN cd /app/tools && npm install --production

# ── 5. Application ──
COPY . /app/

RUN mkdir -p /app/models /app/data /app/logs /app/outputs /app/temp

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "web_app.app:create_app()"]
