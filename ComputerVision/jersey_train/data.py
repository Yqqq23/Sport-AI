"""训练数据。支持两种来源，都归一到同一个 ``JerseyDataset``。

来源 A：Roboflow 数据集（``manifest.json``）
    basketball-v2 的 ``scripts/download_jersey_ocr_dataset.py`` 下载并清洗后
    写出的 manifest。需要 Roboflow API key。

来源 B：文件夹树（**推荐，尤其是给你自己这个机位用**）
    <root>/
        7/    img_0001.png  img_0002.png ...
        12/   ...
        23/   ...
    目录名就是标签。用 ``mine.py`` 从你自己的视频里挖 crop，
    再用 ``label.py`` 打标，就得到这个结构。

两种可以混用（``--data A --data B``），这也是实际最有效的做法：
公开数据给量，自己的数据给域。

⚠ 类别列表来自数据，不是硬编码 0–99
------------------------------------
Roboflow v7 清洗后只剩 41 个号码。硬开一个 100 类的头，会有 59 个类
一个样本都没有 —— 而没训练过的 logit 照样能在 argmax 里赢。
那就会对着模型从没见过的号码给出高置信度预测。

checkpoint 里记着真实类别表，``LocalJerseyClassifier.unrepresentable_numbers()``
就是拿它来明确告诉你「这个号我根本预测不出来」，而不是偷偷投给最像的那个。
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .contract import INPUT_SIZE, MEAN, STD, preprocess_bgr

LOGGER = logging.getLogger(__name__)

#: 图片后缀
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


#: 「看不见号码」的类名。取非数字，这样 basketball-v2 的
#: clean_jersey_number 会返回 None，recognize() 于是返回 (None, conf) ——
#: 正好是 pipeline 已经理解的「本帧无读数」。给模型一个弃权出口。
NO_NUMBER_CLASS = "none"


def _clean_label(raw: str) -> str | None:
    """和 basketball-v2 的 ``clean_jersey_number`` 保持一致的规则。

    有意重写而不是 import：这个包要能在没有 basketball-v2 的机器上单独训练。
    规则很简单，值得复制；但**规则必须一致**，否则训练和推理对同一个字符串
    会给出不同的标签。

    唯一的例外是 ``NO_NUMBER_CLASS``，它是我们刻意引入的弃权类。
    """
    s = str(raw).strip()
    if s.lower() == NO_NUMBER_CLASS:
        return NO_NUMBER_CLASS
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits or len(digits) > 2:
        return None
    if digits == "00":
        return None          # 和上游一致地拒绝 "00"
    return digits.lstrip("0") or "0"      # "01"->"1"，"0" 保留


def load_from_manifest(root: Path) -> list[tuple[Path, str]]:
    """来源 A：读 download_jersey_ocr_dataset.py 写的 manifest.json。"""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} 不存在。\n"
            "  先跑：python scripts/download_jersey_ocr_dataset.py --out " + str(root)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[tuple[Path, str]] = []
    for split_rows in manifest.get("splits", {}).values():
        for row in split_rows:
            p = root / row["image"]
            if p.is_file():
                items.append((p, str(row["label"])))
    LOGGER.info("manifest %s -> %d 张", manifest_path, len(items))
    return items


def load_from_folders(root: Path) -> list[tuple[Path, str]]:
    """来源 B：<root>/<label>/*.png 文件夹树。"""
    items: list[tuple[Path, str]] = []
    skipped_dirs: list[str] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        label = _clean_label(sub.name)
        if label is None:
            skipped_dirs.append(sub.name)
            continue
        n = 0
        for f in sub.rglob("*"):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                items.append((f, label))
                n += 1
        if n == 0:
            LOGGER.warning("目录 %s 里没有图片", sub)
    if skipped_dirs:
        LOGGER.info("跳过非号码目录: %s", skipped_dirs)
    LOGGER.info("文件夹树 %s -> %d 张", root, len(items))
    return items


def load_items(paths: list[str | Path]) -> list[tuple[Path, str]]:
    """把多个数据源合并。目录里有 manifest.json 就按来源 A，否则按来源 B。"""
    items: list[tuple[Path, str]] = []
    for p in paths:
        root = Path(p)
        if not root.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {root}")
        if (root / "manifest.json").is_file():
            items += load_from_manifest(root)
        else:
            items += load_from_folders(root)
    if not items:
        raise ValueError(f"这些目录里一张有标签的图都没有: {paths}")
    return items


#: 从文件名里认出「组」的前缀。组 = 同一个球员的一段连续画面。
#:   SoccerNet tracklet   t000123_007.jpg   -> t000123
#:   自己挖的槽位          slot_003_f001234.png -> slot_003
_GROUP_RE = re.compile(r"^(t\d+|slot_\d+)_")


def group_of(path: Path) -> str:
    """这张图属于哪个「组」。认不出来就每张图自成一组。"""
    m = _GROUP_RE.match(Path(path).name)
    return m.group(1) if m else Path(path).name


def split_items(items, val_fraction: float = 0.15, seed: int = 0,
                min_per_class_in_val: int = 1, group_aware: bool = True):
    """按类别分层 + **按组** 切分训练/验证。

    两件事都不能省：

    **分层** —— 简单随机切的话，稀有类可能整个落进验证集，训练集里一张都没有。
    然后你会看到一个「验证集上永远错」的类，还以为是模型的问题。

    **按组** —— 这条更要命。一条 tracklet / 一个槽位是同一个球员连续几十上百帧，
    画面几乎一样。按**图片**随机切，同一个人的近似重复帧会同时出现在训练和验证
    两边，验证准确率虚高得离谱，而真实表现完全对不上。
    「训过但效果不好」十有八九是栽在这里：指标好看，线上不行。

    所以同一组的图**整组**要么进训练要么进验证，绝不拆开。
    """
    by_class: dict[str, dict[str, list]] = {}
    for path, label in items:
        g = group_of(path) if group_aware else str(path)
        by_class.setdefault(label, {}).setdefault(g, []).append((path, label))

    rng = random.Random(seed)
    train, val = [], []
    for label, groups in sorted(by_class.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        if len(keys) == 1:
            # 这个类只有一组：全给训练集。放验证集等于这个类永远学不会，
            # 而且验证分数还会一直是 0，误导你去调参。
            train += groups[keys[0]]
            continue
        n_val_groups = max(min_per_class_in_val, int(round(len(keys) * val_fraction)))
        n_val_groups = min(n_val_groups, len(keys) - 1)     # 至少给训练留一组
        for k in keys[:n_val_groups]:
            val += groups[k]
        for k in keys[n_val_groups:]:
            train += groups[k]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class JerseyDataset(Dataset):
    """一张 crop -> (CHW float32 tensor, class index)。

    增强作用在 **BGR uint8** 上，最后统一走 ``contract.preprocess_bgr``，
    和推理时逐行一致。
    """

    def __init__(self, items, classes: list[str], transform=None,
                 input_size: int = INPUT_SIZE, mean=MEAN, std=STD):
        self.items = list(items)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.input_size = input_size
        self.mean, self.std = mean, std
        unknown = {lab for _, lab in self.items if lab not in self.class_to_idx}
        if unknown:
            raise ValueError(f"数据里有类别表之外的标签: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import torch

        path, label = self.items[i]
        # 中文路径下 cv2.imread 会静默返回 None，所以走 fromfile+imdecode
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"读不了图片（损坏或不是图片）: {path}")
        if self.transform is not None:
            img = self.transform(img)
        arr = preprocess_bgr(img, self.input_size, self.mean, self.std)
        return torch.from_numpy(np.ascontiguousarray(arr)), self.class_to_idx[label]


def _sort_key(label: str):
    """数字类按数值排，非数字类（none）排在最后。"""
    return (1, 0, label) if not label.isdigit() else (0, int(label), "")


def class_list(items) -> list[str]:
    """类别表 = 数据里实际出现过的标签。按数值排序，'none' 放末尾。"""
    return sorted({lab for _, lab in items}, key=_sort_key)


def describe(items, classes: list[str]) -> str:
    counts = Counter(lab for _, lab in items)
    groups = Counter()
    for path, lab in items:
        groups[lab] = groups[lab]
    n_groups = len({group_of(p) for p, _ in items})
    thin = {c: counts[c] for c in classes if counts[c] < 10}
    lines = [
        f"样本总数 : {len(items)}",
        f"独立分组 : {n_groups}  （tracklet / 槽位；验证集按组切分，不按图片）",
        f"类别数   : {len(classes)}",
        f"类别     : {' '.join(classes)}",
        "每类数量 : " + "  ".join(f"{c}:{counts[c]}" for c in classes),
    ]
    if NO_NUMBER_CLASS in classes:
        lines.append(
            f"含弃权类 '{NO_NUMBER_CLASS}' —— clean_jersey_number 对它返回 None，"
            "所以模型判为「看不清」时 pipeline 会当作本帧无读数，不会误报成某个号。")
    if thin:
        lines += [
            f"⚠ 样本不足 10 张的类: {thin}",
            "  这些类不管训多久都不会可靠。要么补数据，要么在推理时把它们排除。",
        ]
    if n_groups < 20:
        lines.append(
            f"⚠ 只有 {n_groups} 个独立分组。分组太少，验证集代表性很弱，"
            "分数会偏乐观。多挖几段视频 / 多下点数据。")
    return "\n".join("  " + s for s in lines)


def sampler_weights(items, classes: list[str]):
    """类别不均衡时的采样权重（1/该类样本数）。

    球衣号天然极度不均衡：几个主力出现几千次，替补几十次。
    不加权的话模型会学成「永远猜最常见的号」，而那在准确率上还挺好看。
    """
    counts = Counter(lab for _, lab in items)
    return [1.0 / counts[lab] for _, lab in items]
