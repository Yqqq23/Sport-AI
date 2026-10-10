"""checkpoint 契约 —— 这个文件是整个包最重要的一个。

训练出来的 ``.pt`` 必须能被 basketball-v2 的 ``LocalJerseyClassifier`` 原样加载。
格式对不上，练得再好也是废的。契约是从
``basketball-v2/src/jersey.py:694-716`` 逐行抄下来的：

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes    = list(checkpoint["classes"])          # 必需，list[str]
    input_size = int(checkpoint.get("input_size", 96))
    mean       = checkpoint.get("mean", (0.485, 0.456, 0.406))
    std        = checkpoint.get("std",  (0.229, 0.224, 0.225))
    model      = torchvision.models.resnet18(weights=None)
    model.fc   = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])   # 必需
    # 另外会读 checkpoint.get("val_accuracy") 打日志

所以架构**不能换**：必须是 torchvision resnet18，最后一层换成
``Linear(512, num_classes)``。想换 backbone 就得同时改 jersey.py，那是另一件事。

推理时的预处理（jersey.py:742-748），训练必须完全对齐：

    cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    BGR -> RGB
    /255.0
    (x - mean) / std

注意是**方形拉伸**，不保持长宽比。训练时如果按比例缩放加 padding，
推理时来的却是拉伸图，那就是自找的域差距。

``verify_checkpoint()`` 就是照上面这段把权重真加载一遍。训练脚本跑完会自动调它，
不通过就当场报错 —— 而不是等你部署到服务器上才发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: 必须和 jersey.OCRGateConfig.upscale_to_height 一致 —— pipeline 就是按这个
#: 尺寸把 crop 递给模型的
INPUT_SIZE = 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: checkpoint 里必须有的键
REQUIRED_KEYS = ("classes", "state_dict")


class CheckpointContractError(RuntimeError):
    """checkpoint 不符合 LocalJerseyClassifier 的加载契约。"""


@dataclass
class CheckpointInfo:
    """verify_checkpoint 的返回。"""

    path: Path
    classes: list[str]
    input_size: int
    mean: tuple
    std: tuple
    val_accuracy: Any = None
    num_params: int = 0
    extras: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"文件         : {self.path}  ({self.path.stat().st_size / 1e6:.1f} MB)",
            f"类别数       : {len(self.classes)}",
            f"类别         : {' '.join(self.classes)}",
            f"输入尺寸     : {self.input_size}x{self.input_size}",
            f"归一化       : mean={tuple(round(v, 4) for v in self.mean)} "
            f"std={tuple(round(v, 4) for v in self.std)}",
            f"参数量       : {self.num_params/1e6:.2f} M",
            f"验证集准确率 : {self.val_accuracy}",
        ]
        if self.extras:
            lines.append(f"附加信息     : {', '.join(sorted(self.extras))}")
        return "\n".join("  " + s for s in lines)


def build_model(num_classes: int):
    """按契约建模型：resnet18 + 换掉的 fc。

    ⚠ 不要改这里。这个结构是 jersey.py 加载时写死的，换了就 load_state_dict 失败。
    """
    import torch.nn as nn
    from torchvision import models

    if num_classes < 2:
        raise CheckpointContractError(
            f"类别数是 {num_classes}，至少要 2 类才谈得上分类。"
            " 数据里只有一个号码的话，模型学不到任何东西。"
        )
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(
    path: str | Path,
    *,
    model,
    classes: Sequence[str],
    val_accuracy: float | None = None,
    input_size: int = INPUT_SIZE,
    mean: Sequence[float] = MEAN,
    std: Sequence[float] = STD,
    **extras: Any,
) -> Path:
    """按契约写 checkpoint。``extras`` 里的东西 jersey.py 会忽略，纯粹留档。"""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": [str(c) for c in classes],
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_size": int(input_size),
        "mean": tuple(float(v) for v in mean),
        "std": tuple(float(v) for v in std),
        "val_accuracy": None if val_accuracy is None else float(val_accuracy),
        **extras,
    }
    torch.save(payload, str(path))
    return path


def verify_checkpoint(path: str | Path, *, strict_shape: bool = True) -> CheckpointInfo:
    """照 jersey.py 的做法把 checkpoint 真加载一遍。

    这不是「看看键在不在」，是**真的构造 resnet18 并 load_state_dict**。
    只有这样才能保证部署那一刻不会失败。
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise CheckpointContractError(f"checkpoint 不存在: {path}")

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointContractError(f"torch.load 失败: {exc}") from exc

    if not isinstance(ckpt, dict):
        raise CheckpointContractError(
            f"checkpoint 顶层应该是 dict，实际是 {type(ckpt).__name__}。"
            " jersey.py 会直接 ckpt['classes'] 取值。"
        )
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise CheckpointContractError(
            f"checkpoint 缺少必需键 {missing}（现有键: {sorted(ckpt)}）"
        )

    classes = list(ckpt["classes"])
    if not classes or not all(isinstance(c, str) for c in classes):
        raise CheckpointContractError(
            "classes 必须是非空的 list[str]。数字类型会让 jersey.py 后面的"
            " clean_jersey_number(label) 拿到 int 而炸掉。"
        )
    if len(set(classes)) != len(classes):
        raise CheckpointContractError(f"classes 里有重复项: {classes}")

    input_size = int(ckpt.get("input_size", INPUT_SIZE))
    mean = tuple(ckpt.get("mean", MEAN))
    std = tuple(ckpt.get("std", STD))
    if len(mean) != 3 or len(std) != 3:
        raise CheckpointContractError(f"mean/std 必须是 3 个数: mean={mean} std={std}")
    if any(v == 0 for v in std):
        raise CheckpointContractError(f"std 里有 0，推理时会除零: {std}")

    model = build_model(len(classes))
    try:
        model.load_state_dict(ckpt["state_dict"], strict=strict_shape)
    except Exception as exc:
        raise CheckpointContractError(
            f"load_state_dict 失败: {exc}\n"
            "  说明权重和 resnet18 + Linear(512, num_classes) 对不上。"
            " 架构不能换 —— jersey.py 是写死的。"
        ) from exc
    model.eval()

    # 真跑一次前向，确认输出维度对得上类别数
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size)
        out = model(dummy)
    if out.shape != (1, len(classes)):
        raise CheckpointContractError(
            f"前向输出形状 {tuple(out.shape)}，期望 (1, {len(classes)})"
        )

    known = set(REQUIRED_KEYS) | {"input_size", "mean", "std", "val_accuracy"}
    return CheckpointInfo(
        path=path, classes=classes, input_size=input_size, mean=mean, std=std,
        val_accuracy=ckpt.get("val_accuracy"),
        num_params=sum(p.numel() for p in model.parameters()),
        extras={k: ckpt[k] for k in ckpt if k not in known},
    )


def preprocess_bgr(crop, input_size: int = INPUT_SIZE, mean=MEAN, std=STD):
    """把一张 BGR crop 处理成模型输入 —— **和 jersey.py 推理时逐行一致**。

    训练和推理都必须走这一个函数。分成两份写，迟早会漂移，
    而这种漂移在验证集上完全看不出来，只在真实视频上表现为准确率崩掉。
    """
    import cv2
    import numpy as np

    resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return rgb.transpose(2, 0, 1)          # CHW
