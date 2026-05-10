# 电商客服 Agent — 生产镜像（API + 可选 Streamlit 由 compose 启动）
# 使用 SQLite checkpoint：uvicorn 请保持 **单 worker**，避免多进程写同一库。

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# 数据与索引（仓库内 mock）；若你在构建前未生成，可在容器首次启动时挂载卷执行脚本
RUN mkdir -p /app/.cache

EXPOSE 8000 8501

# 默认入口仅用于镜像自检；实际命令由 docker-compose 覆盖
CMD ["python", "-c", "print('Use docker-compose or: uvicorn apps.api_server:app --host 0.0.0.0 --port 8000')"]
