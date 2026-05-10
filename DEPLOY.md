# 服务器部署方案

面向单机 / 小团队：可选 **Docker Compose（推荐）** 或 **裸机 systemd + Nginx**。

## 架构说明

| 组件 | 端口（默认） | 说明 |
|------|----------------|------|
| FastAPI | 8000 | `GET /health`、`POST /chat`，适合对接自有前端或网关 |
| Streamlit | 8501 | 内置对话 / Debug 界面 |

持久化目录（checkpoint、长期记忆、Chroma）默认落在 **Docker 卷**或你挂载的 `/data`，勿与多副本并发写同一 SQLite 文件。

**重要：** 当前 checkpoint 使用 SQLite，`uvicorn` 请使用 **`--workers 1`**。若要水平扩展，需换成 Postgres checkpoint 等共享后端（需改代码与依赖）。

## 方案 A：Docker Compose（推荐）

### 前置

- 服务器已安装 Docker、Docker Compose Plugin
- 仓库内已包含 `data/` mock 数据；若首次构建镜像前未生成，可在宿主机执行一次 `python scripts/generate_mock_data.py` 后再 `docker compose build`

### 步骤

```bash
cd ecommerce-cs-agent
cp .env.example .env
# 编辑 .env：真实模型时填写 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL 等

docker compose build
docker compose up -d
```

- API：<http://服务器IP:8000/health>
- UI：<http://服务器IP:8501>

查看日志：`docker compose logs -f`

升级：`git pull && docker compose build --no-cache && docker compose up -d`

### HTTPS 与域名

生产环境不要在公网裸暴露端口，建议在主机安装 **Nginx** 或 **Caddy**，反向代理到 `127.0.0.1:8000` / `8501`。可参考仓库内 `deploy/nginx-example.conf`，证书使用 [Certbot](https://certbot.eff.org/) 或云厂商证书。

### 防火墙

仅开放 `80`、`443`（及 SSH）；API/UI 通过 Nginx 转发，无需对公网开放 8000/8501。

---

## 方案 B：裸机（venv + systemd）

适合不方便用 Docker 的环境。

```bash
sudo apt update && sudo apt install -y python3.11-venv nginx
cd /opt
sudo git clone <你的仓库> ecommerce-cs-agent
cd ecommerce-cs-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && vim .env

# 冒烟
source .venv/bin/activate && uvicorn apps.api_server:app --host 127.0.0.1 --port 8000 --workers 1
```

将 `deploy/systemd/*.example` 复制到 `/etc/systemd/system/`（去掉 `.example`），按需修改 `User`、`WorkingDirectory`，然后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ecommerce-agent.service
sudo systemctl enable --now ecommerce-agent-streamlit.service   # 可选
```

Nginx 配置同上，upstream 指向 `127.0.0.1:8000` 与 `8501`。

---

## 资源与安全建议

1. **内存**：Embedding / Chroma / sentence-transformers 较重，建议 **≥ 4 GiB**；纯 `fake` 模型可适当降低。
2. **密钥**：`.env` 权限 `chmod 600`，勿提交 Git。
3. **限流**：公网 API 建议在网关或 Nginx 做速率限制、鉴权（API Key / JWT）。
4. **备份**：定期备份挂载卷中的 `/data`（含 checkpoint、long_memory、chroma）。
5. **联网搜索**：安装 `duckduckgo-search`（已在 requirements）；出站需允许访问 DuckDuckGo。

---

## 健康检查与运维

- API：`curl -s http://127.0.0.1:8000/health`
- Compose 已为 `agent-api` 配置 `healthcheck`；Streamlit 无内置 health，可通过端口探测。

---

## 仅暴露 API（最小化攻击面）

若只需要后端对接自有前端，Compose 中可注释掉 `agent-ui` 服务，或仅用 systemd 启动 uvicorn，不对外开 Streamlit。
