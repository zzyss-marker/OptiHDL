# OptiHDL Docker 镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置工作目录
WORKDIR /app

# 安装系统依赖和 Yosys
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# 验证 Yosys 安装
RUN yosys -V

# 配置 pip 清华源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 升级 pip
RUN pip3 install --no-cache-dir --upgrade pip

# 复制依赖文件
COPY requirements.txt /app/

# 安装 Python 依赖（使用清华源）
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件（排除大文件，见 .dockerignore）
COPY . /app/

# 创建必要的目录
RUN mkdir -p /app/models/qwen /app/models/qwen_finetuned /app/data /app/logs /app/uploads /app/temp

# 注意：models/ 目录应该通过 Docker 卷挂载，不要打包到镜像中

# 暴露端口 (Flask 默认 5000)
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 使用 gunicorn 启动 Flask 应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "web_app.app:create_app()"]
