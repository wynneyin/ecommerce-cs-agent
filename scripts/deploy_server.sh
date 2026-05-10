#!/usr/bin/env bash
# 一键在 Linux 服务器上部署（无 Docker）：venv、依赖、.env、可选 mock 数据与索引，并后台启动 API + Streamlit。
#
# 用法（在项目仓库根目录执行）：
#   bash scripts/deploy_server.sh
#   bash scripts/deploy_server.sh --install-only          # 只安装，不启动
#   bash scripts/deploy_server.sh --skip-index            # 跳过检索索引构建（省时间）
#   bash scripts/deploy_server.sh --restart               # 先停再起（等价于更新后重启）
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INSTALL_ONLY=0
SKIP_INDEX=0
RESTART=0

usage() {
  cat <<'EOF'
用法: bash scripts/deploy_server.sh [选项]

  --install-only / --no-start   只安装依赖与环境，不启动服务
  --skip-index                  跳过 scripts/build_index.py（省时间 / 离线）
  --restart                     启动前先执行 stop_local_services（适合更新代码后重启）
  -h, --help                    显示帮助

示例:
  bash scripts/deploy_server.sh
  bash scripts/deploy_server.sh --restart
  bash scripts/deploy_server.sh --install-only
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-only | --no-start)
      INSTALL_ONLY=1
      ;;
    --skip-index)
      SKIP_INDEX=1
      ;;
    --restart)
      RESTART=1
      ;;
    -h | --help)
      usage 0
      ;;
    *)
      echo "未知参数: $1"
      usage 1
      ;;
  esac
  shift
done

die() {
  echo "错误: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "未找到命令「$1」，请先安装（例如 apt install $1）"
}

need_cmd python3
python3 -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"' \
  || die "需要 Python >= 3.10，当前: $(python3 -V 2>/dev/null || echo unknown)"

echo "=========================================="
echo "  ecommerce-cs-agent 一键部署（无 Docker）"
echo "=========================================="
echo "项目目录: $ROOT"
echo ""

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "[1/5] 创建虚拟环境 .venv …"
  python3 -m venv "$ROOT/.venv"
else
  echo "[1/5] 已存在 .venv，跳过创建"
fi

PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
[[ -x "$PY" ]] || die ".venv/bin/python 不可用"

echo "[2/5] 安装 / 更新依赖 …"
"$PIP" install --upgrade pip
"$PIP" install -r "$ROOT/requirements.txt"

echo "[3/5] 环境文件 …"
if [[ ! -f "$ROOT/.env" ]]; then
  if [[ -f "$ROOT/.env.example" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "       已从 .env.example 复制为 .env —— 部署后请务必编辑其中的 LLM 密钥等配置。"
  else
    echo "       警告: 未找到 .env.example，请手动创建 .env"
  fi
else
  echo "       已存在 .env，不覆盖"
fi

echo "[4/5] 数据与索引 …"
if [[ ! -f "$ROOT/data/products.json" ]] || [[ ! -f "$ROOT/data/orders.json" ]]; then
  echo "       缺少 mock 数据，运行 scripts/generate_mock_data.py …"
  "$PY" "$ROOT/scripts/generate_mock_data.py"
else
  echo "       data/products.json 等已存在，跳过生成"
fi

if [[ "$SKIP_INDEX" -eq 1 ]]; then
  echo "       已 --skip-index，跳过 build_index.py（之后可手动: $PY scripts/build_index.py）"
else
  echo "       构建检索索引（fake embedding 时较快）…"
  if "$PY" "$ROOT/scripts/build_index.py"; then
    echo "       索引构建完成"
  else
    echo "       警告: build_index.py 失败，可检查日志后手动执行上述命令"
  fi
fi

if [[ "$INSTALL_ONLY" -eq 1 ]]; then
  echo ""
  echo "[5/5] 已指定 --install-only，不启动服务。"
  echo "启动命令:  bash scripts/run_local_services.sh"
  exit 0
fi

if [[ "$RESTART" -eq 1 ]]; then
  echo "[5/5] --restart：尝试停止已有进程 …"
  if [[ -x "$ROOT/scripts/stop_local_services.sh" ]]; then
    bash "$ROOT/scripts/stop_local_services.sh" || true
  fi
  sleep 1
else
  echo "[5/5] 启动服务 …"
fi

bash "$ROOT/scripts/run_local_services.sh"

echo ""
echo "=========================================="
echo "  部署完成"
echo "=========================================="
echo "  · API 健康检查:  curl -s http://127.0.0.1:8000/health"
echo "  · Streamlit:     http://127.0.0.1:8501（外网访问请改防火墙并慎用 0.0.0.0）"
echo "  · 日志目录:      $ROOT/.logs/"
echo "  · 停止服务:      bash scripts/stop_local_services.sh"
echo "  · 再次部署重启:  bash scripts/deploy_server.sh --restart"
echo ""
