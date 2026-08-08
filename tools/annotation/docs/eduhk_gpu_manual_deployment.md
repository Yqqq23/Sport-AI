# EdUHK GPU Server Manual Deployment

## 场景说明

本说明假设你会使用 MobaXterm 的 SFTP 手动上传整个项目文件夹到服务器，
然后再通过 SSH 登录服务器手动执行命令。

目标部署目录：

```bash
~/jersey-active-learning
```

## 推荐目录结构

```text
~/jersey-active-learning/
~/jersey-active-learning/backend/
~/jersey-active-learning/frontend/
~/jersey-active-learning/data/
```

## 上传后先做环境检查

```bash
cd ~/jersey-active-learning
bash scripts/check_server_env.sh
```

## 后端启动

```bash
cd ~/jersey-active-learning/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

如果还没有 `.env`，先复制示例：

```bash
cp .env.server.example .env
```

然后按服务器实际路径修改：

- `DATABASE_URL`
- `DATA_DIR`
- `CORS_ORIGINS`
- `SPORT_CV_ROOT`（如果不用可以留空）

## 前端启动

```bash
cd ~/jersey-active-learning/frontend
cp .env.server.example .env
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

推荐保持：

```env
VITE_API_BASE_URL=
VITE_PROXY_TARGET=http://127.0.0.1:8001
```

这样浏览器只访问前端 5173，Vite 会在服务器本机把 `/videos`、`/upload-video` 等请求代理到后端 `127.0.0.1:8001`，避免 `127.0.0.1` / 服务器 IP / CORS 混乱。

关键点：

- `frontend/src/api/client.js` 现在默认优先走相对路径
- `frontend/vite.config.js` 会读取 `.env` 中的 `VITE_PROXY_TARGET`
- 所以如果你通过 MobaXterm / 本地端口转发访问 `127.0.0.1:5173`，前端也仍然会把请求正确代理到服务器上的后端

## 浏览器访问

```text
http://SERVER_IP:5173
http://SERVER_IP:8001/docs
```

注意：

- `0.0.0.0` 只是监听地址，不是浏览器访问地址
- 请使用：

```bash
hostname -I
```

查看服务器真实 IP

## 首次服务器测试清单

1. 打开 `http://SERVER_IP:8001/health`
2. 打开 Swagger：`http://SERVER_IP:8001/docs`
3. 上传一个小视频
4. 确认抽帧成功
5. 点击 `Load Frames for Annotation`
6. 保存一条人工标注
7. 点击 `Download Corrected Dataset`
8. 最后再测试 `Run YOLO Pre-annotation`

## Troubleshooting

### 1. Address already in use

如果 8001 或 5173 被占用：

```bash
ss -ltnp | grep 8001
ss -ltnp | grep 5173
```

换端口或停止冲突进程。

### 2. ModuleNotFoundError: ultralytics

说明后端依赖没有装完整：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 前端无法连接后端

检查：

- `frontend/.env` 的 `VITE_API_BASE_URL` 是否为空
- `frontend/.env` 的 `VITE_PROXY_TARGET` 是否正确指向 `http://127.0.0.1:8001`
- 后端是否运行在 `0.0.0.0:8001`
- 浏览器是否使用了正确的 `SERVER_IP`

### 4. CORS error

修改 `backend/.env` 中的：

```env
CORS_ORIGINS=http://SERVER_IP:5173,http://SERVER_IP:8001
```

然后重启后端。

### 5. 0.0.0.0 不是浏览器 URL

不要在浏览器里访问：

```text
http://0.0.0.0:8001
```

正确做法是使用服务器实际 IP：

```bash
hostname -I
```

## GPU 说明

当前 `requirements.txt` 不强制安装 CUDA 版 torch。

如果你后续确认服务器 GPU/CUDA 环境正常，再单独按服务器环境安装 GPU 版 torch。
这一步不要先写死在项目 requirements 里。
