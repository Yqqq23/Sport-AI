# trajectory —— 按球衣号抽取球员俯视轨迹

把两边接起来，不重写任何算法：

| 来源 | 负责 |
|---|---|
| `D:\basketball-v2` | 球员检测、MOT、球衣号 OCR + 投票锁定、身份粘连 |
| 本项目 `calibration/` | **米制**单应标定（已验收：线拟合中位偏差 0.0635 m） |
| 本包 `trajectory/` | 把前者的每帧球员框投影成球场米坐标，产出轨迹 / 热区 / 动画 |

## 为什么不用 basketball-v2 自带的标定

它的 `configs/court_calibration.json` 有两个问题：

1. 球场坐标是一张 **500×940 俯视画布的像素**，不是米。没法直接算跑动距离、速度、热区密度。
2. `image_points` 明显错位 —— `corner_tl` 在 `(679, 615)`、`corner_br` 在 `(849, 637)`，两个对角挨在一起。

本包用 `calibration/court_3x3.json`，单位是米，且过了这些验收：

```
平均误差   0.1650 m   中位 0.0259 m
线拟合     11219 个球场线像素，中位偏差 0.0635 m
物理检查   球场不跨越地平线
```

## 用法

```bash
# 0. 自检（上云先跑这个，别浪费 GPU 时间）
python -m trajectory.cli check

# 1. 抽取：跑视频 -> 逐帧球场坐标 CSV（慢，跑一次）
python -m trajectory.cli extract --video game.mp4 --out outputs/run1

# 只跑开头一分钟；同时生成 outputs/minute1/annotated.mp4
python -m trajectory.cli extract --video game.mp4 --out outputs/minute1 --duration-sec 60

# 2. 看看识别到了哪些球员
python -m trajectory.cli players --csv outputs/run1/tracks.csv

# 3. 画某个号的轨迹
python -m trajectory.cli plot --csv outputs/run1/tracks.csv --jersey 7
python -m trajectory.cli plot --csv outputs/run1/tracks.csv --jersey 7 --video
python -m trajectory.cli plot --csv outputs/run1/tracks.csv --all
```

一键（固定处理视频开头 60 秒，同时输出 `annotated.mp4` 和指定号码的轨迹动画）：

```bash
./trajectory/run.sh /path/to/game.mp4 outputs/run1 7
```

该入口生成的 `tracks.csv`、`extract_summary.json`、轨迹图、热区图、
`track_num7.csv`、`annotated.mp4` 和 `trajectory_num7.mp4` 都只包含或统计
输入视频开头一分钟。需要手动处理其他时长时，仍可直接调用 `extract` 并指定
`--duration-sec`。

## 产出

| 文件 | 内容 |
|---|---|
| `tracks.csv` | 逐帧每个球员一行：frame / t / slot / identity / jersey / bbox / 脚点像素 / **court_x_m, court_y_m** / in_court / sane |
| `extract_summary.json` | 帧数、OCR 命中、出界统计、识别到的号码 |
| `annotated.mp4` | 原视频上的球员/号码、篮球、篮筐和裁判检测框 |
| `trajectory_num7.png` | 轨迹图，按瞬时速度着色，起点绿终点红 |
| `heatmap_num7.png` | 0.5 m 网格热区 |
| `trajectory_num7.mp4` | 轨迹动画（`--video`） |
| `track_num7.csv` | 该球员的清理后轨迹（平滑前后都有），方便接别的分析 |

## 三个必须知道的处理细节

### 1. 取脚，不取框中心

`CourtProjector.project_feet()` 用的是检测框**底边中点**。单应只对地平面成立，胸口不在地面上。本项目实测同一个人：

```
脚   (8.45, 1.72) m
中心 (8.63, 3.78) m     差 2.07 m
```

2 米足够把人从篮下挪到三分线外。

### 2. 先平滑再算距离

30 fps 下检测框底边每帧抖几像素，投影到球场就是几厘米。直接对逐帧原始位置累加位移，噪声会一路累积。合成测试里同一条轨迹：

```
原始逐帧累加   90.6 m
平滑后         38.9 m      （真值 38.2 m）
```

**原始值高了 2.4 倍**，是纯噪声。所以 `Track.distance_m` 一律用平滑轨迹算，`xy_raw` 只用来画点。

默认 Savitzky-Golay（保住急停急转的形状），没有 scipy 就退回滑动平均并打日志。

### 3. 同号冲突按「离上一点最近」判，不按置信度

OCR 会把别人的号看成这个号，而且那种误识别常常**置信度很高**。所以同一帧出现多条同号记录时，选离上一个已接受位置最近的那条。

起始点是个特例（没有「上一点」）。这里用开头 45 帧候选点的**中位数**当锚点 —— 中位数对少数离群点免疫。这一条是测出来的 bug：原本第一帧按置信度挑，结果高置信度的误识别点当了种子，整条轨迹被拽偏，跑动距离从 38.9 m 虚报到 47.1 m。

跳变超过 `max_step_m`（默认 2.0 m/帧）的点直接丢弃；缺口 ≤ `max_gap_frames`（默认 15 帧）线性插值补上，更长的缺口**保留断开**——1 秒的洞用直线连起来就是编造轨迹。

## 部署到云服务器

当前服务器目录布局（`PROJECT = /tmp/s1143700/basketball-v2`）：

```text
/tmp/s1143700/basketball-v2/
├── .venv/
├── src/
├── configs/
├── calibration/
├── trajectory/       ← trajectory.sh
└── jersey_train/     ← jersey_train.sh
```

**路径不写死。** `trajectory.sh` 里的 `PROJECT` 是 `dirname $(dirname $0)`，
`bridge.py` 的 `project_root` 是 `Path(__file__).parent.parent`。项目挪到任何
位置、换任何用户名，代码都不用改 —— 只有本文档里的示例路径需要跟着更新。

```bash
chmod +x /tmp/s1143700/basketball-v2/trajectory/trajectory.sh

/tmp/s1143700/basketball-v2/trajectory/trajectory.sh check
/tmp/s1143700/basketball-v2/trajectory/trajectory.sh extract --video game.mp4 --out outputs/run1
/tmp/s1143700/basketball-v2/trajectory/trajectory.sh players --csv outputs/run1/tracks.csv
/tmp/s1143700/basketball-v2/trajectory/trajectory.sh plot --csv outputs/run1/tracks.csv --jersey 7 --video
```

也可以直接运行 `cli.py`，不依赖当前工作目录：

```bash
/tmp/s1143700/basketball-v2/.venv/bin/python \
  /tmp/s1143700/basketball-v2/trajectory/cli.py check
```

如果继续使用 `python3 -m trajectory.cli`，则必须先进入包的上一级目录：

```bash
cd /tmp/s1143700/basketball-v2
source .venv/bin/activate
python -m pip install -r trajectory/requirements.txt
python -m trajectory.cli check
```

### ⚠ 项目放在 `/tmp` 下有风险

多数 Linux 发行版会在**重启时清空 `/tmp`**，有的还挂了 `tmpfs`（占内存）或者
跑 `systemd-tmpfiles` 定期删除多天未访问的文件。你的输入视频（200 MB）、
`tracks.csv`、训练数据和 checkpoint 都在里面，一次重启就没了。

跑长任务前先确认一下：

```bash
df -h /tmp                                   # 是不是 tmpfs、还剩多少
findmnt -no FSTYPE /tmp                      # tmpfs 就是内存盘
systemctl cat systemd-tmpfiles-clean.timer 2>/dev/null | head   # 有没有定期清理
```

是 tmpfs 或有清理策略的话，把项目挪到 `$HOME` 或持久卷；至少把
`outputs/`、`data/`、`train/*.pt` 软链到持久位置。

### ⚠ numpy 必须 pin 在 1.x

本机实测：numpy 2.1 装上之后，用 numpy 1.x 编译的二进制包（pyarrow / numexpr / bottleneck / 某些 psutil 轮子）全部报

```
AttributeError: _ARRAY_API not found
```

连 `import pandas` 都会一路报到底，ultralytics 直接起不来。`requirements.txt` 里已经钉了 `numpy>=1.24,<2`。

### 长视频

比赛视频 26341 帧。`extract` 是**边跑边写 CSV** 的，中途挂掉已跑完的部分不丢：

```bash
# 先用一小段验证链路通不通
python -m trajectory.cli extract --video game.mp4 --out outputs/smoke --end-frame 900

# 续跑
python -m trajectory.cli extract --video game.mp4 --out outputs/run2 --start-frame 12000

# 抽帧跑（轨迹够用，速度快 3 倍；速度指标会变粗）
python -m trajectory.cli extract --video game.mp4 --out outputs/run3 --stride 3
```

服务器上 OpenCV 常缺 mp4 编码器，出视频时报错就换：
`--fourcc MJPG`（会自动写成 `.avi`）。

## 已知限制

- **球衣号取决于 OCR**。`check` 里 ultralytics/torch 起不来时 `extract` 跑不了；OCR 后端加载不了时轨迹照样产出，但 `jersey_number` 全空，只能按 `slot` 区分球员（`plot --slot N`）。
- **只标定了一个半场**（Y 0→14 m）。球员跑到另半场，投影值会超出 14 m。`sane` 字段标记了离场地 6 m 以外的离谱结果，`build_track` 默认丢弃。
- 分辨率必须是 **1920×1080**。单应和像素系绑死，`CourtProjector.check_frame()` 不符直接抛异常，不静默算出偏移的坐标。
