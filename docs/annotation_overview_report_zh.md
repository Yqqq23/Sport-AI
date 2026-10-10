# Annotation 报告：目录结构、整体 Pipeline 与训练数据

## 一、模块定位

`tools/annotation/` 是篮球视频的人工标注与数据集管理工具。用户通过网页查看视频帧和目标框，填写或修改类别、球衣号码，再将标注导出为训练数据。

它采用 **React 前端 + FastAPI 后端 + SQLite 数据库**。模型训练和比赛视频分析的主要代码位于项目根目录的 `train/`、`scripts/` 和 `src/`，并不全部放在 annotation 文件夹内。

从整体上看，项目由三部分组成：

```text
标注与数据管理                  模型训练                     视频分析
tools/annotation/   ──────→    train/、scripts/   ──────→    src/
人工生成真实标签                得到检测和号码识别权重         对比赛视频进行推理
```

上图表示各部分的数据关系；当前标注网页没有自动执行完整训练流程。

## 二、文件夹分支与文件作用

### 2.1 Annotation 总体目录

```text
tools/annotation/
├── backend/                   后端服务、数据库、接口和数据导出
│   ├── app/
│   │   ├── main.py            FastAPI 应用入口
│   │   ├── api/routes.py      上传、复核、标注、导出等接口
│   │   ├── core/config.py     后端配置
│   │   ├── db/session.py      数据库连接与表结构维护
│   │   ├── models/dataset.py  数据库表定义
│   │   ├── schemas/dataset.py 接口请求和响应的数据结构
│   │   └── services/          存储、类别管理、导出等业务功能
│   ├── tests/                 后端功能测试
│   ├── requirements.txt       后端依赖
│   ├── debug_upload.py        上传接口调试脚本
│   └── jersey_active_learning.db
│                              SQLite 数据库文件
├── frontend/                  网页标注界面
│   ├── src/
│   │   ├── main.jsx           React 启动入口
│   │   ├── App.jsx            主界面及交互逻辑
│   │   ├── api/client.js      后端 HTTP 接口调用
│   │   └── styles/            页面样式
│   └── vite.config.js         前端开发与构建配置
├── data/                      视频、图像、裁剪和导出数据
├── scripts/                   数据导入、数据库维护和服务器启动
├── docs/                      使用、部署和历史集成说明
├── README.md                  项目说明
├── start.bat                  Windows 启动脚本
├── start.sh                   Shell 启动脚本
└── stop.bat                   Windows 停止脚本
```

### 2.2 Backend：后台业务层

| 文件 | 主要作用 | 在流程中的位置 |
|---|---|---|
| `app/main.py` | 创建 FastAPI 服务，初始化数据库和数据目录，注册接口，提供视频帧静态访问；存在前端构建产物时托管网页 | 后端启动入口 |
| `app/api/routes.py` | 接收视频上传、查询视频与帧、加载复核条目、保存/更新/删除标注、生成和下载数据集 | 连接网页操作与后台功能 |
| `app/core/config.py` | 管理数据库地址、数据目录、跨域地址、裁剪尺寸等设置 | 配置读取 |
| `app/db/session.py` | 建立 SQLAlchemy 数据库连接、提供会话，并维护 SQLite 表结构 | 数据持久化基础 |
| `app/models/dataset.py` | 定义视频、帧、预测、人工标注、处理任务、数据集版本等数据库表 | 存储记录的结构 |
| `app/schemas/dataset.py` | 定义 API 请求和返回的数据格式及校验规则 | 前后端数据交换 |

这里的 `models/dataset.py` 中，“models”指数据库模型，不是 YOLO 或 ResNet 神经网络。

数据库中几个主要对象的关系是：

```text
VideoAsset：一个上传的视频
    └── VideoFrame：该视频抽出的多张帧
            └── JerseyPrediction：帧上的候选框及其来源
                    └── ManualAnnotation：人工确认的类别、框和号码

DatasetVersion：将一批人工标注组织为可导出的数据集版本
ProcessingJob：记录后台任务状态和进度
```

### 2.3 Services：存储与数据集处理

以下文件位于 `backend/app/services/`：

| 文件 | 主要作用 |
|---|---|
| `storage.py` | 管理数据目录、视频存储和解码/抽帧等文件处理 |
| `basketball_classes.py` | 从根目录 `train/basketball.yaml` 读取五类定义，并连接 `src/jersey.py` 的号码处理规则 |
| `yolo_export.py` | 提供 YOLO 导出的格式约定与校验功能；具体导出接口还位于 `routes.py` |
| `coco_export.py` | 提供 COCO 类别映射和导出数据校验 |
| `dataset_versioning.py` | 汇集多个视频的标注、划分 train/val/test、生成版本目录、manifest 和划分报告 |

该目录还保留了以下历史识别文件：

| 文件 | 文件内容与当前状态 |
|---|---|
| `yolo_preannotation.py` | 原 YOLOv8n person 预标注代码；文件头标注已停用，HTTP 入口已移除 |
| `real_recognition.py` | 原 YOLOv8n + Tesseract 识别流程；当前入口已停用 |
| `fixed7_adapter.py` | 对外部 Sport-CV/fixed7 流程的适配代码；当前诊断入口已停用 |
| `face_clustering.py` | InsightFace `buffalo_l` 特征提取及相似度聚类辅助代码，关联历史识别流程 |
| `mock_recognition.py` | 测试/调试用的模拟预测生成器 |

因此，不能仅凭某个模型文件存在，就将它列为当前主流程正在运行的模型。

### 2.4 Frontend：人工标注界面

网页部分主要集中在 `frontend/src/App.jsx`，负责视频选择、复核条目展示、边界框编辑、类别与号码填写、标注保存及导出操作。

`api/client.js` 负责发送请求。例如，用户点击保存后，前端把框坐标、类别和号码交给后端 API，后端再写入数据库。界面本身不执行 YOLO 或 ResNet 推理。

### 2.5 Data：运行数据的存放位置

```text
tools/annotation/data/
├── videos/          上传的视频
├── frames/          视频抽帧图像
├── crops/           球员/球衣号码裁剪图
├── annotations/     标注相关文件目录
├── exports/         单次导出的数据文件和归档
└── datasets/
    └── versions/    带版本号的数据集
```

人工标注的核心记录保存在 SQLite 中，`data/` 主要保存对应媒体与导出产物。

常见导出文件的含义：

| 文件或目录 | 含义 |
|---|---|
| `images/` | 用于训练的原始帧图像 |
| `labels/` | YOLO 目标检测标签 |
| `dataset.yaml`、`classes.txt` | 数据路径与类别定义 |
| `jersey_numbers.json` | 球衣号码字符串，供号码识别数据处理使用 |
| `manifest.json` | 导出图像与原视频、帧、标注记录之间的对应关系 |
| `metadata.json` | 图像数量、标注数量、来源和版本等信息 |
| `split_report.json` | train/val/test 划分结果 |

### 2.6 Scripts 与 Docs

| 文件/目录 | 主要作用 |
|---|---|
| `scripts/import_crops.py` | 将外部挖掘产生的 `manifest.jsonl` 和裁剪图导入标注数据库 |
| `scripts/migrate_db.py` | 数据库结构迁移 |
| `scripts/reset_db.py` | 重置标注数据库及相关运行数据的维护工具 |
| `scripts/check_server_env.sh` | 检查服务器环境 |
| `scripts/start_server_backend.sh` | 启动服务器后端 |
| `scripts/start_server_frontend.sh` | 启动服务器前端 |
| `docs/dataset_versioning_design.md` | 数据集版本与划分设计说明 |
| `docs/real_recognition_integration.md` 等 | 识别集成及历史方案说明 |
| 部署类文档 | GPU 服务器上传、环境与启动说明 |

部分 README 和集成说明记录的是早期流程，当前接口状态以 `routes.py` 和服务文件头说明为准。

## 三、宏观 Pipeline 与模型结构

### 3.1 标注 Pipeline

```text
比赛视频
    ↓ 上传与抽帧
视频帧 + 数据库帧记录
    ↓ 导入外部挖掘结果
候选号码框 + 裁剪图 + 来源信息
    ↓ 网页复核
人工填写类别、边界框、球衣号码
    ↓ 数据保存与导出
检测数据：图像 + 五类目标框
号码数据：号码区域 + 号码字符串
```

当前导入代码将自动号码留空、识别置信度置为 0，表示这一阶段没有运行 OCR。人工标注者负责填写号码。

代码将外部挖掘命令写为 `jersey_train mine`，但当前展开的工作区中未找到该命令的实现；本目录可直接确认的是 `import_crops.py` 的接收格式和后续标注流程。

### 3.2 模型训练 Pipeline

项目训练分为两条线路：

```text
线路 A：目标检测
篮球图像 + 五类边界框
    ↓ train/train.py
YOLO26（默认基础权重 yolo26s.pt）
    ↓
best.pt：检测模型权重

线路 B：球衣号码识别
ImageNet 预训练 ResNet18
    ↓ train/pretrain_svhn.py：SVHN 数字预训练
stage1_svhn.pt
    ↓ train/finetune_soccernet.py：SoccerNet + 合成号码图
ResNet18 三头号码网络
    ↓
stage3_soccernet.pt + stage3_soccernet.json
```

annotation 生成的人工号码标签可作为后续本场景数据来源；当前目录未提供其已经参与最终权重训练的记录。

### 3.3 视频推理 Pipeline

主项目入口是 `src/pipeline.py`，从宏观上可以理解为：

```text
输入比赛视频
    ↓
YOLO26 五类检测
    ├── player → 球员跟踪 → 地面点投影 → 米制运动轨迹
    ├── jersey → 号码区域裁剪 → 号码识别 → 跨帧号码稳定
    ├── ball、rim → 球轨迹与投篮几何判断
    └── referee → 从球员分析中排除
    ↓
整合球员身份、号码、位置和投篮记录
    ↓
annotated.mp4 / topdown.mp4 / shots.csv / shots.json
```

主要组件及对应文件如下：

| 处理环节 | 模型或方法 | 主要文件 |
|---|---|---|
| 目标检测 | Ultralytics YOLO26 五类模型 | `src/detect.py`、`src/track.py` |
| 图像空间跟踪 | Ultralytics 跟踪接口，默认配置 `tracktrack.yaml` | `src/track.py` |
| 号码识别 | 优先加载 ResNet18 逐位识别模型 | `src/jersey.py`、`train/jersey_net.py` |
| 号码与身份管理 | 跨帧统计、号码锁定及身份记录 | `src/jersey.py`、`src/identity.py` |
| 球场投影 | 已标定的单应性矩阵 | `src/calib.py` |
| 米制运动跟踪 | Kalman + Hungarian 关联 | `src/motion.py` |
| 投篮判断 | 球与篮筐的几何状态机 | `src/shot.py` |
| 输出 | 原视频标注、俯视图、统计文件 | `src/render.py`、`src/stats.py` |

其中主要需要学习权重的是目标检测器与号码识别器；球场投影、运动滤波和投篮状态机使用几何及规则方法。

### 3.4 YOLO26 检测模型

检测器接收比赛视频帧，输出边界框、类别和置信度。项目的类别表为：

| ID | 类别 | 含义 |
|---:|---|---|
| 0 | player | 球员人体 |
| 1 | ball | 篮球 |
| 2 | rim | 篮筐 |
| 3 | jersey | 球衣上号码所在区域 |
| 4 | referee | 裁判 |

训练入口默认使用 `yolo26s.pt`，输入尺寸 960，训练 100 epochs。实际运行的规格取决于训练参数及加载的 checkpoint，单凭 `best.pt` 文件名无法判断是 n、s 还是其他规格。

号码内容不是检测类别。例如，穿 23 号的球员，其号码区域仍属于 `jersey`，数字 `23` 由后续号码网络识别。

### 3.5 ResNet18 号码识别模型

```text
96×96 RGB 号码区域图像
    ↓
ResNet18 骨干
    ↓
512 维特征
    ├── 长度头：3 类（不可读 / 一位 / 两位）
    ├── 第一位数字头：10 类（0～9）
    └── 第二位数字头：10 类（0～9）
    ↓
组合为号码字符串或不可读结果
```

例如，长度输出 2、第一位输出 2、第二位输出 3，就得到号码 `23`。训练时使用三个头的交叉熵损失；不存在的数字位不参与该头的损失计算。

该网络定义在 `train/jersey_net.py`，推理封装是 `src/jersey.py` 的 `PerDigitJerseyClassifier`。权重旁的 JSON 文件记录输入尺寸、归一化参数及输出头含义。

当前 `auto` 识别后端按可用性依次尝试：

`perdigit → local → parseq → paddleocr → easyocr → pytesseract`

其中 `local` 是保留的旧整号码分类推理后端，PARSeq 等是可选识别后端。默认优先级不等于某次运行已成功加载该模型；具体使用者应以运行日志为准。

## 四、训练数据有哪些

### 4.1 数据与模型的对应关系

| 数据集或数据来源 | 服务的模型/任务 | 入口或位置 |
|---|---|---|
| Roboflow 篮球检测数据 v18 | YOLO26 五类检测训练 | `dataset/`、`scripts/prepare_dataset.py` |
| ImageNet-1K | ResNet18 骨干预训练权重来源 | `train/jersey_net.py` |
| SVHN | 号码网络的数字预训练 | `train/pretrain_svhn.py` |
| 自制合成球衣号码图 | 号码网络训练 | `scripts/gen_synthetic.py` |
| SoccerNet jersey-2023 | 号码网络真实图像训练 | `train/finetune_soccernet.py` |
| 人工标注的比赛视频帧及裁剪 | 本场景检测、号码数据的收集与导出 | `tools/annotation/` |
| Roboflow OCR v7 | 历史整号码分类训练路线 | `dataset/jersey_number/`、已停用的 `train/train_jersey.py` |

### 4.2 篮球检测数据：Roboflow v18

**名称：** `basketball-player-detection-3-ycjdo`，版本 18。

**来源：** <https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo/dataset/18>

本地实际图像数量为：

| 划分 | 数量 |
|---|---:|
| train | 464 |
| valid | 96 |
| test | 94 |
| 合计 | **654** |

原始数据有 10 类：`ball`、`ball-in-basket`、`number`、`player`、四种球员状态类、`referee`、`rim`。

`scripts/prepare_dataset.py` 将其转换为项目的五类：球及入网状态归为 `ball`，球员及其状态归为 `player`，`number` 对应 `jersey`，另外保留 `referee`、`rim`。

转换后的目录为：

```text
train/datasets/basketball/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

YOLO 标签每行记录 `class_id cx cy w h`，坐标归一化到 0～1。当前本地同时存在原始数据和转换后的数据，数量相同。

### 4.3 ImageNet-1K 与 SVHN

**ImageNet-1K** 用于提供 ResNet18 的初始权重，代码指定 `IMAGENET1K_V1`。项目不是从头训练一个 ImageNet 网络，而是在预训练骨干基础上继续训练号码任务。

**SVHN** 是真实街景门牌数字数据。当前 `pretrain_svhn.py` 使用官方 train 进行训练、test 进行验证，训练骨干和第一位数字头。

- 标准 train 约 7.3 万张。
- 当前入口没有加载 extra，因此不能将 extra 算入这条训练流程的数据量。
- 输出为 `stage1_svhn.pt` 及配套 JSON。

### 4.4 自制合成号码数据

生成脚本为 `scripts/gen_synthetic.py`，覆盖 `0`、`00`、`1`～`99`，并包含不可读样本。图像由字体渲染生成，再加入透视、遮挡、模糊、低分辨率和压缩等变化。

当前默认生成设置为：

| 类型 | 默认目标数量 |
|---|---:|
| 一位号码 | 18,000 |
| 两位号码 | 18,000 |
| 不可读 | 14,000 |

由于按号码分配时使用整数除法，按当前默认参数生成的实际总数为 49,927 张。

本地确认存在两套预览清单：

- `data/synthetic_preview/manifest.json`：606 条。
- `data/synthetic_preview_v2/manifest.json`：532 条。

预览数据数量与完整训练集数量是不同概念；本地没有发现完整的约 5 万张合成训练集。

### 4.5 SoccerNet jersey-2023

该数据由运动员轨迹片段组成，每个 tracklet 对应一个号码或不可读标签。当前加载器将轨迹展开为图像样本，默认每条轨迹最多抽取 12 帧，并从人体图像中裁出躯干区域。

训练配置为：

- SoccerNet train：参与训练。
- 合成号码图：与 SoccerNet 按 1:1 混合。
- SoccerNet test：用于验证与 checkpoint 选择。
- 输出：`stage3_soccernet.pt` 和 `stage3_soccernet.json`。

相关文件是 `scripts/download_soccernet.py`、`train/jersey_data.py` 和 `train/finetune_soccernet.py`。当前本地未发现完整 SoccerNet 数据及最终训练产物，不能给出本次实际训练使用的轨迹数和图像数。

### 4.6 人工标注数据与历史 OCR 数据

**人工标注数据**来自上传的比赛视频。标注系统记录五类框和球衣号码，能够导出检测训练数据及号码标签。当前 `basketball5_demo_v1` 是一套可见的示例导出：3 张图、12 个框，仅包含训练划分，描述为导出功能检查。

**Roboflow OCR v7** 的项目名称为 `basketball-jersey-numbers-ocr`，说明文件标称 3,615 张、224×224 图像。当前 `dataset/jersey_number/` 中保留三个 JSONL 标签文件及说明，没有配套图像。它关联的旧 41 类整号码训练脚本 `train/train_jersey.py` 已停用，当前逐位训练入口没有读取这套数据。

### 4.7 如何理解“使用了这些数据训练”

当前源码清楚给出了两条训练路线：

1. **YOLO26：Roboflow 篮球检测图像及五类标签。**
2. **ResNet18 号码网络：ImageNet 初始化 → SVHN 预训练 → SoccerNet 与合成数据混合训练。**

本地可直接核实检测数据、号码标签、合成预览和人工示例导出。最终权重和完整训练日志未出现在当前展开的工作区，因此上述列表说明的是代码定义的数据使用关系，不能单独证明服务器上的某个最终模型已用所有数据完成训练。

## 五、阅读代码的建议顺序

若要快速理解该目录，可按以下顺序阅读：

1. `frontend/src/App.jsx`：用户能做哪些操作。
2. `backend/app/api/routes.py`：每个操作调用哪些后台功能。
3. `backend/app/models/dataset.py`：视频、预测、人工标签如何保存。
4. `scripts/import_crops.py`：候选框和裁剪图如何进入系统。
5. `backend/app/services/dataset_versioning.py`：标注如何组成训练数据集。
6. 根目录 `train/train.py` 与 `train/jersey_net.py`：训练什么模型。
7. 根目录 `src/pipeline.py` 与 `src/jersey.py`：训练结果如何用于比赛视频。

**概括：Annotation 负责准备和管理带人工标签的数据；YOLO26 负责找到篮球场景中的目标；ResNet18 三头模型负责读取球衣号码；主流水线将检测、跟踪、号码、球场位置和事件结果组织为最终输出。**
