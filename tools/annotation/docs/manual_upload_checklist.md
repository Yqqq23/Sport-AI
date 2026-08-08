# Manual Upload Checklist

## 需要手动上传的主要内容

建议上传整个项目目录：

```text
jersey-active-learning/
  backend/
  frontend/
  scripts/
  docs/
  data/   (按需要)
  README.md
  start.bat
  stop.bat
```

## 可以不上传的内容

以下内容建议在上传前删除或排除：

- `backend/.venv`
- `frontend/node_modules`
- `backend/__pycache__`
- `frontend/__pycache__`
- `**/__pycache__`
- `.pytest_cache`
- `frontend/dist`（如果只是开发运行）

## data/ 是否需要上传

### 如果你想保留现有数据和历史标注

建议上传：

- `data/videos/`
- `data/frames/`
- `data/exports/`
- `data/datasets/`（如果已有版本化数据集）

### 如果你只想部署一个全新空项目

可以不上传 `data/`，让服务器首次启动时自动创建。

## SQLite 数据库是否需要上传

如果你想保留现有：

- 视频记录
- 预测记录
- 人工标注
- 标注进度
- 数据集版本记录

则需要上传 SQLite 数据库文件，例如：

- `backend/jersey_active_learning.db`

如果不上传数据库：

- 之前的标注状态会丢失
- 仅上传视频和抽帧图片不足以恢复完整状态

## 推荐上传方式

### 方式 1：直接上传项目文件夹

在 MobaXterm 的 SFTP 面板中，把本地 `jersey-active-learning` 整个目录拖到：

```bash
~/jersey-active-learning
```

### 方式 2：先压缩再上传

可先在本地压缩为 zip：

- `jersey-active-learning.zip`

上传到服务器后解压：

```bash
unzip jersey-active-learning.zip
```

## 上传前本地整理建议

1. 删除 `backend/.venv`
2. 删除 `frontend/node_modules`
3. 删除所有 `__pycache__`
4. 决定是否保留 `data/`
5. 决定是否保留 `backend/jersey_active_learning.db`

## 最推荐的最小可运行上传内容

如果你只想尽快在服务器上跑起来：

- `backend/`
- `frontend/`
- `scripts/`
- `docs/`
- `README.md`
- `backend/.env.server.example`
- `frontend/.env.server.example`

如果你还想保留已有标注和视频：

再加上：

- `backend/jersey_active_learning.db`
- `data/`
