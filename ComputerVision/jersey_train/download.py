"""下载并清洗 Roboflow 的球衣号数据集（CC BY 4.0）。

    export ROBOFLOW_API_KEY=你的key
    python -m jersey_train.cli download --out data/jersey_ocr

数据源
    basketball-jersey-numbers-ocr v7
    https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr
    License: CC BY 4.0 —— **要求署名**。脚本会写一份 ATTRIBUTION.txt；
    训练出来的模型如果对外发布/演示，这份署名必须跟着走。

为什么要清洗而不是直接用
------------------------
这个项目在 Roboflow 上的类型是 ``text-image-pairs``，导出的是给 VLM 微调用的
JSONL，不是分类用的目录树：

    {"image": "....jpg", "prefix": "Read the number.", "suffix": "28"}

原始导出里有三类会**悄悄污染分类器**的东西（v7 实测计数）：

    52 行 suffix 为空          —— 根本没标签，丢掉
    28 行标成 "00"             —— 上游 clean_jersey_number 会拒绝 "00"，
                                  pipeline 永远输出不了它，留着这个类只会产生
                                  永不可达的预测。丢掉，并报出来
    40 行 prefix 损坏          —— suffix 本身还是好的，分类器不用 prefix，
                                  所以**保留**，但计数报出来

类别表取自数据，不假设 0–99：清洗后只剩 41 个号码。开 100 类的头会有 59 个类
一个样本都没有，而没训练过的 logit 照样能赢 argmax。

⚠ 这个包里已经有一份同样的实现
------------------------------
basketball-v2 的 ``scripts/download_jersey_ocr_dataset.py`` 做的是同一件事。
这里重写一份，只是为了让 jersey_train 能在没有那个仓库的机器上独立跑起来
（比如只上传训练相关的目录去 GPU 机器）。两边的清洗规则必须一致 ——
不一致就意味着训练标签和推理时的 clean_jersey_number 对不上。
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)

WORKSPACE = "roboflow-jvuqo"
PROJECT = "basketball-jersey-numbers-ocr"
VERSION = 7
EXPORT_FORMAT = "jsonl"

ATTRIBUTION = f"""\
This directory contains a derivative of:

  basketball-jersey-numbers-ocr (version {VERSION})
  https://universe.roboflow.com/{WORKSPACE}/{PROJECT}
  License: CC BY 4.0  (https://creativecommons.org/licenses/by/4.0/)

CC BY 4.0 requires attribution. If any model trained on this data is published,
distributed, or demonstrated outside your team, this credit must travel with it.
"""


def clean_label(raw: str) -> str | None:
    """和 basketball-v2 的 clean_jersey_number 同规则。见 data.py 的同名函数。"""
    s = "".join(ch for ch in str(raw) if ch.isdigit())
    if not s:
        return None
    stripped = s.lstrip("0")
    if not stripped or len(stripped) > 2:
        return None
    return stripped


def download(api_key: str, out_dir: Path) -> Path:
    import requests

    meta_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/{EXPORT_FORMAT}"
    LOGGER.info("请求导出链接: %s", meta_url)
    r = requests.get(meta_url, params={"api_key": api_key}, timeout=120)
    r.raise_for_status()
    payload = r.json()
    link = (payload.get("export") or {}).get("link")
    if not link:
        raise RuntimeError(f"响应里没有导出链接: {json.dumps(payload)[:400]}")

    size = (payload.get("export") or {}).get("size")
    LOGGER.info("下载中 ~%.1f MB ...", size or 0.0)
    archive = requests.get(link, timeout=1800)
    archive.raise_for_status()

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        zf.extractall(out_dir)
    LOGGER.info("已解压到 %s", out_dir)
    return out_dir


def build_manifest(root: Path) -> dict:
    splits: dict[str, list[dict]] = {}
    dropped: Counter = Counter()
    bad_prefix = 0
    label_counts: Counter = Counter()

    for split in ("train", "valid", "test"):
        ann = root / split / "annotations.jsonl"
        if not ann.exists():
            LOGGER.warning("缺少 %s", ann)
            continue
        rows: list[dict] = []
        for line in ann.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped["unparseable_json"] += 1
                continue
            if str(rec.get("prefix", "")) != "Read the number.":
                bad_prefix += 1
            raw = str(rec.get("suffix", "")).strip()
            if not raw:
                dropped["empty_label"] += 1
                continue
            label = clean_label(raw)
            if label is None:
                dropped[f"rejected:{raw}"] += 1
                continue
            image = rec.get("image")
            if not image or not (root / split / image).exists():
                dropped["missing_image"] += 1
                continue
            rows.append({"image": f"{split}/{image}", "label": label})
            label_counts[label] += 1
        splits[split] = rows

    classes = sorted(label_counts, key=lambda s: int(s))
    return {
        "source": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                   "license": "CC BY 4.0",
                   "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}"},
        "classes": classes,
        "counts": {s: len(r) for s, r in splits.items()},
        "class_counts": {c: label_counts[c] for c in classes},
        "dropped": dict(dropped),
        "rows_with_corrupt_prefix": bad_prefix,
        "splits": splits,
    }


def run(out_dir: str | Path, *, api_key_env: str = "ROBOFLOW_API_KEY",
        skip_download: bool = False) -> Path:
    out_dir = Path(out_dir)
    if not skip_download:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(
                f"环境变量 ${api_key_env} 里没有 API key。\n"
                f"  先设置：export {api_key_env}=你的key\n"
                "  key 在 https://app.roboflow.com 的账号设置里拿。\n"
                "  没有 key 也能训 —— 用 mine + label 从自己的视频挖数据，"
                "对你这个机位反而更有效。"
            )
        download(api_key, out_dir)
    elif not out_dir.exists():
        raise SystemExit(f"给了 --skip-download 但 {out_dir} 不存在")

    (out_dir / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    manifest = build_manifest(out_dir)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = manifest["counts"]
    print(f"\n  保留    {sum(counts.values())} 行  {counts}")
    print(f"  类别    {len(manifest['classes'])} 个: {' '.join(manifest['classes'])}")
    print(f"  丢弃    {sum(manifest['dropped'].values())} {manifest['dropped']}")
    print(f"  坏prefix {manifest['rows_with_corrupt_prefix']}（保留，分类器不用 prefix）")
    rare = {c: n for c, n in manifest["class_counts"].items() if n < 10}
    if rare:
        print(f"\n  ⚠ 样本 <10 的类: {rare}")
        print("    训多久都不会可靠。")
    print(f"\n  manifest  {path}")
    return path
