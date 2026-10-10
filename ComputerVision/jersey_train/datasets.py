"""公开数据集的下载与转换。

    # SoccerNet（推荐：一行命令，不用 API key，2853 条球员 tracklet）
    python -m jersey_train.cli fetch --source soccernet --out data/soccernet

    # Roboflow（需要 API key）
    export ROBOFLOW_API_KEY=...
    python -m jersey_train.cli fetch --source roboflow --out data/jersey_ocr

数据源对比
----------
====================  ==========  ==========  =====================================
                      需要 key    规模        域
====================  ==========  ==========  =====================================
soccernet             否          2853 条     足球转播/跟踪缩略图，小、糊、有遮挡
                                  tracklet    —— **和你的广角机位最接近**
roboflow              是          ~几千张     NBA 转播 224px 清晰特写，清洗后 41 类
mined（你自己的视频）  否          看你挖多少  就是你的部署域，最有效但要人工标
====================  ==========  ==========  =====================================

最佳组合：``--data data/soccernet --data data/labelled``。
SoccerNet 给量和域，自己的数据给具体号码。

⚠ 两个会静默毁掉结果的坑，本模块都处理了
----------------------------------------

**1. "-1" 绝对不能当类名**

SoccerNet 用 ``-1`` 表示「号码不可见」。但 basketball-v2 的
``clean_jersey_number("-1")`` 会抽出数字变成 ``"1"`` ——
模型说「看不清」会被 pipeline 读成**球衣 1 号**，而且悄无声息。

本模块映射成 ``"none"``：``clean_jersey_number("none")`` 返回 None，
``LocalJerseyClassifier.recognize`` 于是返回 ``(None, conf)``，
正好是 pipeline 已经理解的「本帧无读数」。副作用是模型获得了**弃权**能力，
比强行在几十个号码里猜一个要好得多。

**2. 同一条 tracklet 不能横跨训练/验证集**

一条 tracklet 是同一个球员连续几十上百帧，画面几乎一样。按图片随机切分，
同一个人的近似重复帧会同时出现在两边 —— 验证准确率虚高得离谱，而真实
表现完全对不上。**这很可能就是「训过但效果不好」的原因**：指标好看，
线上不行。

所以转换出来的文件名带 tracklet 前缀（``t000123_007.jpg``），
``data.split_items`` 会按这个前缀分组切分。
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: 「看不见号码」的类名。刻意取非数字，让 clean_jersey_number 返回 None。
NO_NUMBER_CLASS = "none"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# SoccerNet
# ---------------------------------------------------------------------------
def download_soccernet(out_dir: str | Path, splits=("train", "test")) -> Path:
    """下载 SoccerNet jersey-2023。不需要密码/NDA。

    https://github.com/SoccerNet/sn-jersey
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as exc:
        raise SystemExit(
            "缺少 SoccerNet 包。装一下：\n    pip install SoccerNet\n"
            f"（原始错误: {exc}）"
        ) from exc

    LOGGER.info("下载 SoccerNet jersey-2023 到 %s（splits=%s）", out_dir, list(splits))
    dl = SoccerNetDownloader(LocalDirectory=str(out_dir))
    dl.downloadDataTask(task="jersey-2023", split=list(splits))
    LOGGER.info("下载完成")
    return out_dir


def _find_split_dirs(root: Path) -> dict[str, Path]:
    """找出 train/test/challenge 目录 —— 不同版本的解压层级不一样，所以是搜不是猜。"""
    found: dict[str, Path] = {}
    for name in ("train", "test", "challenge"):
        for cand in [root / name, *root.glob(f"**/{name}")]:
            if cand.is_dir() and (list(cand.glob("*.json")) or
                                  any(d.is_dir() for d in cand.iterdir())):
                found[name] = cand
                break
    return found


def _find_gt_and_images(split_dir: Path) -> tuple[Path | None, Path | None]:
    """在一个 split 目录里定位 GT json 和图片根目录。

    官方 README 没写死文件名（只说「一个 json 和一个 image 文件夹」），
    所以这里是**发现**而不是硬编码 —— 硬编码一个我没法验证的名字，
    等于把失败推迟到服务器上。
    """
    gts = sorted(split_dir.glob("*.json"))
    gt = gts[0] if gts else None

    images = None
    for name in ("images", "image", "imgs"):
        d = split_dir / name
        if d.is_dir():
            images = d
            break
    if images is None:
        # 退路：找一个「里面全是子目录、子目录里是图片」的目录
        for d in split_dir.iterdir():
            if not d.is_dir():
                continue
            subs = [s for s in d.iterdir() if s.is_dir()]
            if subs and any(f.suffix.lower() in IMAGE_SUFFIXES
                            for s in subs[:3] for f in s.iterdir()):
                images = d
                break
    return gt, images


def prepare_soccernet(
    root: str | Path,
    out_dir: str | Path,
    *,
    splits=("train", "test"),
    max_per_tracklet: int = 25,
    include_no_number: bool = True,
    copy: bool = True,
) -> dict:
    """把 SoccerNet 的 tracklet 结构转成 ``<label>/t<id>_<n>.jpg`` 目录树。

    ``max_per_tracklet`` 是必要的：一条 tracklet 可能有几百帧近似重复的图，
    不限量的话少数几个球员会主导整个数据集。取均匀间隔的若干帧，不是取前 N 帧
    （前 N 帧往往是连续的、差异最小的）。
    """
    root, out_dir = Path(root), Path(out_dir)
    if not root.exists():
        raise FileNotFoundError(f"SoccerNet 目录不存在: {root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dirs = _find_split_dirs(root)
    if not split_dirs:
        raise FileNotFoundError(
            f"{root} 下面找不到 train/test/challenge 目录。\n"
            f"  实际内容: {[p.name for p in root.iterdir()][:20]}"
        )

    stats = {"splits": {}, "labels": Counter(), "tracklets": 0, "images": 0,
             "skipped_no_gt": 0, "no_number_tracklets": 0}

    for split in splits:
        sd = split_dirs.get(split)
        if sd is None:
            LOGGER.warning("没有 %s 这个 split，跳过", split)
            continue
        gt_path, images_dir = _find_gt_and_images(sd)
        if gt_path is None or images_dir is None:
            LOGGER.warning("%s 里找不到 GT json 或图片目录（gt=%s images=%s），跳过",
                           sd, gt_path, images_dir)
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        LOGGER.info("%s: gt=%s  images=%s  %d 条 tracklet",
                    split, gt_path.name, images_dir.name, len(gt))

        n_img = 0
        for tracklet_id, raw in gt.items():
            tdir = images_dir / str(tracklet_id)
            if not tdir.is_dir():
                stats["skipped_no_gt"] += 1
                continue

            # ⚠ 关键：-1 映射成 "none"，绝不能直接用 "-1"
            try:
                num = int(raw)
            except (TypeError, ValueError):
                continue
            if num < 0:
                if not include_no_number:
                    continue
                label = NO_NUMBER_CLASS
                stats["no_number_tracklets"] += 1
            else:
                label = str(num).lstrip("0") or "0"
                if label == "0":
                    # clean_jersey_number 会拒绝 "00"，但 "0" 是合法号码，保留
                    label = "0"

            files = sorted(f for f in tdir.iterdir()
                           if f.suffix.lower() in IMAGE_SUFFIXES)
            if not files:
                continue
            # 均匀取样，不取前 N 帧 —— 连续帧之间几乎没有差异
            if len(files) > max_per_tracklet:
                step = len(files) / max_per_tracklet
                files = [files[int(i * step)] for i in range(max_per_tracklet)]

            dest = out_dir / label
            dest.mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(files):
                # 文件名带 tracklet 前缀，data.split_items 靠它按组切分，
                # 避免同一个球员横跨训练/验证集
                target = dest / f"t{tracklet_id}_{k:03d}{f.suffix.lower()}"
                if copy:
                    shutil.copy2(f, target)
                else:
                    shutil.move(str(f), str(target))
                n_img += 1
                stats["labels"][label] += 1
            stats["tracklets"] += 1

        stats["splits"][split] = n_img
        stats["images"] += n_img

    if stats["images"] == 0:
        raise RuntimeError(
            f"一张图都没转换出来。检查 {root} 的目录结构是否是 "
            "<split>/<gt>.json + <split>/images/<player_id>/*.jpg"
        )

    (out_dir / "SOURCE.txt").write_text(
        "Derived from the SoccerNet jersey number dataset (jersey-2023 task).\n"
        "https://github.com/SoccerNet/sn-jersey   https://www.soccer-net.org/\n"
        "Cite SoccerNet if you publish anything trained on this.\n\n"
        f"tracklets={stats['tracklets']}  images={stats['images']}\n"
        f"'-1' (number not visible) was mapped to the class '{NO_NUMBER_CLASS}', "
        "NOT to '-1': clean_jersey_number('-1') would yield '1' and silently\n"
        "report an unreadable crop as jersey number 1.\n",
        encoding="utf-8")

    print(f"\n  tracklet   {stats['tracklets']}")
    print(f"  图片       {stats['images']}  {dict(stats['splits'])}")
    print(f"  类别       {len(stats['labels'])}")
    print(f"  无号码类   {stats['no_number_tracklets']} 条 tracklet -> '{NO_NUMBER_CLASS}'")
    top = sorted(stats["labels"].items(), key=lambda kv: -kv[1])
    print("  最多的 12 类: " + "  ".join(f"{k}:{v}" for k, v in top[:12]))
    thin = {k: v for k, v in stats["labels"].items() if v < 10}
    if thin:
        print(f"  ⚠ 样本 <10 的类: {thin}")
    print(f"\n  输出       {out_dir}")
    return stats


# ---------------------------------------------------------------------------
def fetch(source: str, out_dir: str | Path, **kwargs) -> Path:
    """统一入口。"""
    out_dir = Path(out_dir)
    if source == "soccernet":
        raw = out_dir / "raw"
        if kwargs.get("skip_download"):
            LOGGER.info("跳过下载，直接转换 %s", raw)
        else:
            download_soccernet(raw, splits=kwargs.get("splits", ("train", "test")))
        prepare_soccernet(
            raw, out_dir,
            splits=kwargs.get("splits", ("train", "test")),
            max_per_tracklet=kwargs.get("max_per_tracklet", 25),
            include_no_number=kwargs.get("include_no_number", True),
        )
        return out_dir
    if source == "roboflow":
        from .download import run as roboflow_run
        roboflow_run(out_dir, skip_download=kwargs.get("skip_download", False))
        return out_dir
    raise ValueError(f"未知数据源 {source!r}，可选: soccernet, roboflow")
