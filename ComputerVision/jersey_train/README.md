# jersey_train —— 训练球衣号分类器

产出 `jersey_classifier.pt`，能被 basketball-v2 的 `LocalJerseyClassifier` 直接加载。

## 当前状态（已核查）

```
D:\basketball-v2             没有任何 .pt / .pth / .onnx / .ckpt
D:\帝国理工\Computer Vision   没有任何 .pt / .pth / .onnx / .ckpt
data/  runs/  weights/       全都不存在  → 下载和训练脚本一次都没跑过
```

`configs/default.yaml` 里两个权重路径：

| 配置项 | 值 | 状态 |
|---|---|---|
| `detector.weights` | `/home/s1143757/basketball-v2/runs/.../weight/best.pt` | **路径有三处错，且用户名/挂载点都换了** —— 见下 |
| `jersey.local.weights` | `../train/jersey_classifier.pt` | 缺失 —— 就是本包要造的东西 |

## 服务器上怎么跑

布局（`PROJECT = /tmp/s1143700/basketball-v2`）：

```text
/tmp/s1143700/basketball-v2/
├── .venv/  src/  configs/  calibration/
├── trajectory/       ← trajectory.sh
└── jersey_train/     ← jersey_train.sh
```

```bash
chmod +x /tmp/s1143700/basketball-v2/jersey_train/jersey_train.sh
J=/tmp/s1143700/basketball-v2/jersey_train/jersey_train.sh

$J check                                              # 先自检
$J fetch --source soccernet --out data/soccernet
$J mine  --video game.mp4 --out data/mined --end-frame 9000
$J label --src data/mined --out data/labelled
$J train --data data/soccernet --data data/labelled
$J verify --ckpt train/jersey_classifier.pt --numbers 0,6,12
```

**代码里没有任何写死的路径。** `jersey_train.sh` 的 `PROJECT` 是
`dirname $(dirname $0)`，`trajectory/bridge.py` 的 `project_root` 是
`Path(__file__).parent.parent`。项目挪位置、换用户名都不用改代码。

### 本包用到 basketball-v2 的地方只有一行

`jersey_train/mine.py`：

```python
from trajectory.bridge import Backend
```

路径解析全部发生在 `trajectory/bridge.py` 的 `_candidates()`，查找顺序是
`--repo` → `$BASKETBALL_V2` → 项目根 → 同级 `basketball-v2` → `D:/basketball-v2`。
在你的布局下第三条就命中了（`jersey_train` 和 `trajectory` 都在仓库里面）。

`fetch` / `train` / `verify` / `predict` **完全不依赖 basketball-v2**，
在任何装了 torch 的机器上都能单独跑。

### ⚠ 项目放在 `/tmp` 下有风险

多数 Linux 会在**重启时清空 `/tmp`**，有的挂 `tmpfs`（占内存）或跑
`systemd-tmpfiles` 定期删除多天未访问的文件。训练数据、checkpoint、
200 MB 的视频都在里面，一次重启全没。跑长任务前确认：

```bash
findmnt -no FSTYPE,SIZE,AVAIL /tmp     # tmpfs 就是内存盘
systemctl cat systemd-tmpfiles-clean.timer 2>/dev/null | head
```

是 tmpfs 或有清理策略，就把项目挪到 `$HOME`，至少把 `data/` 和
`train/*.pt` 软链到持久位置。

## 「训过但效果不好」最可能的三个原因

按我在代码里能验证的严重程度排：

### 1. tracklet 泄漏 —— 验证分数是假的

一条 tracklet / 一个槽位是同一个球员连续几十上百帧，画面几乎一样。按**图片**随机切训练/验证集，同一个人的近似重复帧会同时落在两边。测试里实测：480 张图、24 组，group-blind 切分 **16 组同时出现在训练和验证集**。

结果就是验证准确率虚高得离谱，线上完全不行。**这个现象和「效果不好」的描述完全吻合**：指标好看，实际不能用。

`data.split_items` 现在默认 `group_aware=True`，同一组整组进训练或整组进验证，绝不拆开。分组从文件名认（`t000123_...` / `slot_003_...`）。

### 2. 类别里没有你的号码

模型只能输出训练过的类。你视频里是 0、6、12 这些，如果数据集里没有，softmax **根本没有对应输出** —— 不是「不太可能」而是物理上不可能，它会高置信度地给你一个最像的错号。

```bash
python -m jersey_train.cli verify --ckpt xxx.pt --numbers 0,6,12
```

### 3. 域差距

公开数据是 224px 清晰特写，你的部署是 20–40px 广角。`augment.py` 的 `RandomDownscale` 就是针对这个：每张图压到 12–48px 再还原，逼网络学能活下来的特征。

## 数据集

| 来源 | 需要 key | 规模 | 域 |
|---|---|---|---|
| **soccernet** | 否 | 2853 条 tracklet，号码 1–99 | 足球跟踪缩略图，**小、糊、有遮挡 —— 最接近你的机位** |
| roboflow | 是 | 几千张，清洗后 41 类 | NBA 转播 224px 清晰特写 |
| mined | 否 | 看你挖多少 | 就是你的部署域 |

### SoccerNet（推荐，一行命令，无需 API key）

```bash
pip install SoccerNet
python -m jersey_train.cli fetch --source soccernet --out data/soccernet
```

内部做两件必要的事：

- **`-1` → `none`**，绝不写成 `"-1"`。SoccerNet 用 `-1` 表示「号码不可见」，而 `clean_jersey_number("-1")` 会抽出数字变成 `"1"` —— 模型说「看不清」会被 pipeline 读成**球衣 1 号**，静默且系统性。映射成 `none` 之后 `clean_jersey_number` 返回 `None`，`recognize()` 返回 `(None, conf)`，正好是 pipeline 已理解的「本帧无读数」。副作用是模型获得**弃权**能力。
- **每条 tracklet 均匀取 25 帧**（`--max-per-tracklet`），并把 tracklet id 写进文件名。不限量的话少数球员会主导整个数据集；不写 id 就没法按组切分。

最佳组合是两个一起用：

```bash
python -m jersey_train.cli train --data data/soccernet --data data/labelled
```

SoccerNet 给量和域，自己挖的数据给具体号码。

## 两条路线

### 路线 A：从**你自己的视频**挖数据（推荐）

```bash
export BASKETBALL_V2=/path/to/basketball-v2

python -m jersey_train.cli mine  --video game.mp4 --out data/mined --end-frame 9000
python -m jersey_train.cli label --src data/mined --out data/labelled
python -m jersey_train.cli train --data data/labelled --out train/jersey_classifier.pt
python -m jersey_train.cli verify --ckpt train/jersey_classifier.pt --numbers 0,6,12
```

`mine` 需要检测器权重能加载（它要跑 Tracker 才能定位球衣区域）。

### 路线 B：Roboflow 公开数据集（需 API key）

```bash
export ROBOFLOW_API_KEY=你的key
python -m jersey_train.cli download --out data/jersey_ocr
python -m jersey_train.cli train --data data/jersey_ocr --data data/labelled
```

`--data` 可以给多次，**两个数据源混用效果最好**：公开数据给量，自己的数据给域。

## 为什么强烈建议用自己的视频

公开数据集（`basketball-jersey-numbers-ocr` v7）：

- NBA 转播画面裁的 **224px 清晰特写**
- 清洗后只剩 **41 个类**，不是 0–99
- **你视频里的号码不一定在这 41 个里**

而你的部署目标是固定广角机位，号码只有 **20–40 px**，球馆灯光。纯源域训练会得到一个验证集好看、真视频上崩掉的模型 —— 而且崩得很安静：它会对着一个从没见过的号码给出高置信度的错误答案。

`verify --numbers 0,6,12` 就是用来当场查这个的：不在类别表里的号码，softmax 里**根本没有对应输出**，不是「不太可能」而是物理上不可能。

## 四个关键设计

### 1. checkpoint 契约不能破（`contract.py`）

`jersey.py:694-716` 是写死的：

```python
ckpt["classes"]        list[str]
ckpt["state_dict"]     torchvision resnet18 + Linear(512, num_classes)
ckpt.get("input_size", 96) / ckpt.get("mean") / ckpt.get("std")
```

**架构不可换。** 想换 backbone 就得同时改 jersey.py。

训练结束会自动跑 `verify_checkpoint()` —— 真的构造 resnet18 并 `load_state_dict`，还跑一次前向确认输出维度。不通过当场报错，而不是等你部署到服务器才发现。

### 2. 预处理训练/推理走同一个函数

`contract.preprocess_bgr()`，和 `jersey.py:742-748` 逐行一致：

```
cv2.resize(crop, (96,96), INTER_CUBIC)   ← 方形拉伸，不保长宽比
BGR -> RGB,  /255,  (x-mean)/std
```

分两份写迟早漂移，而这种漂移**在验证集上完全看不出来**，只在真实视频上表现为准确率崩掉。

### 3. 增强就是主线（`augment.py`）

`RandomDownscale` 把每张图压到 12–48 px 再还原，逼网络学那些经过这一趟还能活下来的特征。加上运动模糊、JPEG 伪影、框抖动（复现检测器 mAP50-95≈0.64 的定位误差）、光度扰动、小角度旋转、噪声。

**没有水平翻转。** 几乎所有默认配方里都有，这里是错的 —— 2 翻过来不是 2，但标签还写着 "2"，等于往训练集掺系统性错标。

### 4. `mine` 用的是推理时同一个裁剪函数

直接调 basketball-v2 的 `crop_jersey`，并且**推理时会被 gate 挡掉的 crop，训练里也不要** —— 那是模型永远不会见到的输入分布。自己另写一套裁剪，等于人为制造训练-推理不一致。

## 标注省力的关键

`mine` 按**槽位**分目录（`slot_003/f001234.png`）。同一个槽位在一段时间内就是同一个球员，所以 `label` 一屏显示同槽位的 40 张，看清楚是几号、输入数字、回车 —— 整批归档。几百张图往往只要十几次按键。

**看不清就按 `s` 跳过。** 错标一张的破坏力远大于少一张：分类器会老老实实把错标当真值学，而且这种错误在验证集上还会自我一致（验证集来自同样的错标），你根本看不出来。

## 类别不均衡

球衣号天然极度不均衡：主力出现几千次，替补几十次。默认开 `WeightedRandomSampler`（权重 1/该类样本数）。不加权的话模型会学成「永远猜最常见的号」，而那在总体准确率上还挺好看。

训练报告会列出验证准确率 <0.6 的弱类别 —— 那些基本都是样本太少，**补数据比调参有用得多**。

## 训完怎么启用

`configs/default.yaml`：

```yaml
jersey:
  backend: local          # 或保持 auto，auto 会优先试 local
  local:
    weights: /absolute/path/to/train/jersey_classifier.pt
```

然后拿真实 crop 抽查一下，别只看验证集分数：

```bash
python -m jersey_train.cli predict --ckpt train/jersey_classifier.pt \
    --images data/mined/slot_003 --min-confidence 0.50 --min-margin 0.15
```

门槛和 `LocalJerseyClassifier` 一致。**通过率太低说明域差距还在 —— 补自己视频的数据，别去调低阈值。**
