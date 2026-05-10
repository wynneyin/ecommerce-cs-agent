#!/usr/bin/env bash
# 停止 run_local_services.sh 启动的进程（按 .logs/*.pid）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for name in api streamlit; do
  f=".logs/${name}.pid"
  if [[ -f "$f" ]]; then
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "已停止 $name (PID $pid)" || true
    else
      echo "PID 文件存在但进程不存在: $f"
    fi
    rm -f "$f"
  else
    echo "无 $f，跳过"
  fi
done

echo "完成。"
