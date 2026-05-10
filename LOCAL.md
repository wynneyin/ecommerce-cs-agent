# 本机直接运行（不用 Docker）

在服务器或开发机上，用 **Python 虚拟环境 + 两个后台进程** 即可。

## 一键部署（推荐）

在已克隆的仓库根目录执行：

```bash
bash scripts/deploy_server.sh
```

会自动：创建 `.venv`、安装 `requirements.txt`、若无 `.env` 则从 `.env.example` 复制、缺少 mock 数据则生成、默认跑一次检索索引构建，最后调用 `run_local_services.sh` 后台启动 API（8000）与 Streamlit（8501）。

常用参数：

- `--install-only`：只安装与环境准备，**不启动**
- `--skip-index`：跳过 `build_index.py`（省时间或离线环境）
- `--restart`：先停再起（适合拉代码更新后）

---

## 1. 准备

```bash
cd ecommerce-cs-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # 按需填写 LLM 等变量
```

可选：生成 mock 数据、建索引（与 README「快速开始」一致）。

## 2. 一键启动（推荐）

在项目根目录执行：

```bash
bash scripts/run_local_services.sh
```

默认监听：

- **API**：`http://0.0.0.0:8000`（健康检查：`GET /health`）
- **Streamlit**：`http://0.0.0.0:8501`

日志：`.logs/api.log`、`.logs/streamlit.log`

停止：

```bash
bash scripts/stop_local_services.sh
```

### 关掉终端 / 断开 SSH 后还会运行吗？

- **用上面的 `run_local_services.sh`：** 脚本里对 uvicorn 和 streamlit 都加了 **`nohup`** 放到后台，一般会 **忽略挂断信号**。所以你关掉本地终端、或 SSH 断开后，**服务多半还会继续跑**（重新登录后用 `curl http://127.0.0.1:8000/health` 或 `ps aux | grep uvicorn` 可验证）。
- **如果你是前台运行**（终端里直接敲 `uvicorn ...`、`streamlit ...` 且没有用 `nohup`/后台）：SSH 一断或关掉终端，进程常会 **一起退出**，**不会**长期留在服务器上。
- 若你希望 **不管什么情况都稳定常驻**，更稳妥的是：**systemd**（见 `deploy/systemd/*.example`），或在 **tmux / screen** 里跑前台命令。

### 常用环境变量（启动前 `export` 或写进 `.env`）

| 变量 | 含义 | 默认 |
|------|------|------|
| `BIND_HOST` | 绑定地址；仅本机可设为 `127.0.0.1` | `0.0.0.0` |
| `API_PORT` | API 端口 | `8000` |
| `UI_PORT` | Streamlit 端口 | `8501` |

示例（只监听本机、换端口）：

```bash
export BIND_HOST=127.0.0.1 API_PORT=8000 UI_PORT=8501
bash scripts/run_local_services.sh
```

前面再挂 **Nginx** 做 HTTPS 时，通常把 `BIND_HOST=127.0.0.1`，由 Nginx 反代到这两个端口。

## 3. 手动前台运行（调试用）

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
uvicorn apps.api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

另开终端：

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
streamlit run apps/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

## 4. 长期驻留（可选）

不用 Docker 时，可用 **systemd** 托管：见 `deploy/systemd/*.example`，把路径改成你的安装目录与 `User`。

或用 **tmux** / **screen** 会话里跑上面的前台命令，避免 SSH 断开后进程退出。

## 5. 注意

- `uvicorn` 请保持 **`--workers 1`**（当前 checkpoint 为 SQLite，多 worker 会锁冲突）。
- 数据目录默认在 `.env` 里的 `CHECKPOINT_DB`、`LONG_MEMORY_DB`、`CHROMA_DIR`；持久化请定期备份对应文件。
