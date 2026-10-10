# 篮球视频分析 Web 系统

系统代码在 `webapp/`：FastAPI 后端 + 原生 HTML/CSS/JavaScript 前端，由同一服务提供，不需要安装 Node.js。

## 在你的 Linux GPU 服务器启动

使用已经能够运行 `src/pipeline.py` 的 Python / CUDA 环境：

```bash
cd /home/s1143757/basketball-v2
pip install -r webapp/requirements.txt
sudo apt-get update
sudo apt-get install -y ffmpeg
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000 --workers 1
```

浏览器打开 `http://服务器IP:8000`。若服务器只能通过 SSH 访问，在自己电脑建立转发：

```bash
ssh -L 8000:localhost:8000 s1143757@服务器地址
```

然后打开 `http://localhost:8000`。交互式 API 文档是 `/docs`。

## 操作流程

1. 上传 MP4 / MOV / AVI / MKV / WebM / M4V，默认最大 2048 MB。
2. 后台排队运行现有流水线，页面显示分析和转码状态，可以打开运行日志。
3. 完成后直接播放或下载 **topdown**、**annotated** 两个视频。
4. 从识别到的球衣号码列表选一个号码，点击生成，得到第三个 **球员视频**：左侧原画面高亮该号码，右侧显示该号码累计俯视路线和当前位置。
5. 换号码后再次生成，仅渲染视频，不重新执行 YOLO/OCR；生成过的号码直接返回缓存结果。

后台执行的命令等价于：

```bash
python src/pipeline.py \
  --input outputs/webapp/<任务ID>/input.mp4 \
  --config configs/default.yaml \
  --output-dir outputs/webapp/<任务ID>/analysis \
  --max-frames 1800 \
  --jersey-backend perdigit \
  --export-trajectories
```

`--export-trajectories` 是新增的可选参数，保存 `trajectories.jsonl`（逐帧号码、身份键、原图框和球场画布坐标）、`trajectory_meta.json` 和 `court_template.png`，供第三个视频使用。原来的命令仍然可用。

## 输出和任务管理

每个上传都有独立的 `outputs/webapp/<任务ID>/`：

```text
input.mp4             上传源文件（扩展名随上传格式变化）
job.json              持久化任务状态、号码列表、球员视频状态
pipeline.log          分析日志
analysis/             原流水线输出、轨迹和球场模板
annotated.mp4         网页可播放的 H.264 MP4
topdown.mp4           网页可播放的 H.264 MP4
player_23.mp4         选择 #23 后生成的视频
```

FFmpeg 转为 H.264/yuv420p 并启用 faststart；视频接口支持 HTTP Range，浏览器可拖动进度条。输出沿用流水线帧率，不包含音轨。

使用 **单进程 `--workers 1`**，后台串行执行任务，避免多个模型任务同时占 GPU。服务重启后已完成结果继续可访问，未完成任务标为失败；重新上传可以重跑，球员视频可以重试。浏览器保存最近任务 ID，刷新后恢复查看。运行中正常关闭服务会等待队列完成。任务文件不会自动清除，可在任务结束后按目录清理。

可选环境变量（启动前设置）：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `BASKETBALL_CONFIG` | 项目 `configs/default.yaml` | 服务器配置文件，建议绝对路径 |
| `BASKETBALL_WEB_DATA` | 项目 `outputs/webapp` | 上传与结果存储目录 |
| `BASKETBALL_MAX_FRAMES` | `1800` | 每次分析帧数上限，正整数 |
| `BASKETBALL_MAX_UPLOAD_MB` | `2048` | 应用层文件大小上限 |

若配置关闭 `write_annotated` / `write_topdown`，Web 任务会因缺少所需视频失败，请保持两个输出开启。模型权重、perdigit 依赖仍使用现有服务器环境及配置。

## 轨迹含义

- 默认球场标定针对已有固定机位；换机位需要按 [球场标定说明](topdown_calibration.md) 更新配置。
- 号码来自现有 OCR / 身份逻辑，仅在逐帧输出具有号码时高亮，不把后续识别结果推测填回早期未知帧。
- 路线按独立身份键绘制，观测中断处不连线。相同号码可能属于不同队伍，当前会高亮所有匹配号码，不自动按队伍消歧。
- 没有识别到号码时仍展示两个全场视频，并提示检查模型与日志。
- 当前是单服务器工作台，无用户账号隔离；部署方式适合自己的服务器或可信局域网。

## 验证

```bash
python -m pytest tests/test_webapp.py -q
```

测试覆盖上传验证、后台命令参数、任务隔离、号码选择缓存、视频 Range 和真实小视频高亮渲染；渲染测试替换 FFmpeg 转码步骤。完整模型推理和 H.264 播放需在具备模型权重及 FFmpeg 的服务器用实际比赛视频验证。
