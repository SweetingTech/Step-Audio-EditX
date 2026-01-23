FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y software-properties-common curl zip unzip git-lfs awscli libssl-dev openssh-server vim \
    && apt-get install -y net-tools iputils-ping iproute2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install --reinstall -y ca-certificates && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 3.12 (pyproject.toml 要求 >=3.12,<3.14)
RUN add-apt-repository -y 'ppa:deadsnakes/ppa' && apt update
RUN apt install python3.12 python3.12-dev python3.12-venv -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# 安装 uv 包管理器
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 复制依赖配置文件
COPY pyproject.toml uv.lock ./

# 使用 uv 安装依赖
RUN uv sync

CMD ["uv", "run", "python", "app.py", "--model-path", "/model", "--model-source", "local"]
