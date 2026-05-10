#!/usr/bin/env bash
# 在无 Docker 的机器上后台启动 FastAPI + Streamlit。
# 用法：在项目根目录执行  bash scripts/run_local_services.sh
# 环境变量（可选）：BIND_HOST  API_PORT  UI_PORT  PYTHON

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .logs

if [[ -f ".logs/api.pid" ]] && kill -0 "$(cat .logs/api.pid)" 2>/dev/null; then
  echo "检测到 API 已在运行（PID $(cat .logs/api.pid)）。请先执行: bash scripts/stop_local_services.sh"
  exit 1
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONPATH="$ROOT"

HOST="${BIND_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

echo "使用 Python: $PY"
echo "API → http://${HOST}:${API_PORT}  （本机访问可把 BIND_HOST 设为 127.0.0.1）"
echo "UI  → http://${HOST}:${UI_PORT}"

nohup "$PY" -m uvicorn apps.api_server:app \
  --host "$HOST" \
  --port "$API_PORT" \
  --workers 1 \
  >>".logs/api.log" 2>&1 &
echo $! >".logs/api.pid"

nohup "$PY" -m streamlit run apps/streamlit_app.py \
  --server.address="$HOST" \
  --server.port="$UI_PORT" \
  --browser.gatherUsageStats=false \
  >>".logs/streamlit.log" 2>&1 &
echo $! >".logs/streamlit.pid"

echo ""
echo "已启动（SQLite checkpoint 请勿开多个 uvicorn worker）。"
echo "  API PID:       $(cat .logs/api.pid)    日志: .logs/api.log"
echo "  Streamlit PID: $(cat .logs/streamlit.pid)    日志: .logs/streamlit.log"
echo "停止: bash scripts/stop_local_services.sh"
