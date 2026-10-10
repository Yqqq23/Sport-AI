"""训练球衣号分类器，产出 basketball-v2 能直接加载的 checkpoint。

    python -m jersey_train.cli train --data data/jersey_ocr --out train/jersey_classifier.pt

为什么是分类而不是 OCR
----------------------
球衣号是**闭集**。合法答案只有几十个数字，让模型有能力吐出任意字符串，
只是给了它更多出错的方式。闭集分类还能给出**逐号码的标定置信度**——
而 basketball-v2 的跨帧投票（``JerseyNumberStabilizer``）吃的正是这个。

架构不可换
----------
必须是 torchvision resnet18 + ``Linear(512, num_classes)``。
``LocalJerseyClassifier`` 加载时是写死这个结构的（jersey.py:701-703）。
想换 backbone 就得同时改 jersey.py，属于另一件事。

训练完自动跑 ``verify_checkpoint()``：真的构造 resnet18 并 load_state_dict，
不通过当场报错，而不是等你部署到服务器才发现。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .augment import eval_augment, train_augment
from .contract import (INPUT_SIZE, MEAN, STD, build_model, save_checkpoint,
                       verify_checkpoint)
from .data import (JerseyDataset, class_list, describe, load_items,
                   sampler_weights, split_items)

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    checkpoint: Path
    classes: list[str]
    best_val_acc: float
    epochs_run: int
    history: list[dict] = field(default_factory=list)
    per_class_acc: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def report(self) -> str:
        lines = [
            f"checkpoint   : {self.checkpoint}",
            f"类别数       : {len(self.classes)}",
            f"最佳验证准确率: {self.best_val_acc:.4f}",
            f"训练轮数     : {self.epochs_run}",
            f"耗时         : {self.elapsed_sec:.0f} s",
        ]
        weak = {c: a for c, a in self.per_class_acc.items() if a < 0.6}
        if weak:
            lines.append("弱类别（验证准确率 <0.6）:")
            for c, a in sorted(weak.items(), key=lambda kv: kv[1]):
                lines.append(f"    #{c:<4} {a:.2f}")
            lines.append("    这些多半是样本太少。补数据比调参有用得多。")
        return "\n".join("  " + s for s in lines)


def _resolve_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("要 cuda 但不可用，退回 cpu（会慢很多）")
        return torch.device("cpu")
    return torch.device(requested)


def train(
    data_dirs: list[str | Path],
    out_path: str | Path = "train/jersey_classifier.pt",
    *,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.15,
    input_size: int = INPUT_SIZE,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 0,
    aug_strength: float = 1.0,
    label_smoothing: float = 0.05,
    balance_classes: bool = True,
    pretrained: bool = True,
    early_stop_patience: int = 12,
) -> TrainResult:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(seed)
    np.random.seed(seed)

    items = load_items(list(data_dirs))
    classes = class_list(items)
    print(describe(items, classes))

    train_items, val_items = split_items(items, val_fraction=val_fraction, seed=seed)
    LOGGER.info("训练 %d 张 / 验证 %d 张", len(train_items), len(val_items))
    if not val_items:
        raise ValueError("验证集是空的。数据太少，或者每个类只有一张。")

    train_ds = JerseyDataset(train_items, classes,
                             transform=train_augment(input_size, aug_strength),
                             input_size=input_size)
    val_ds = JerseyDataset(val_items, classes, transform=eval_augment(input_size),
                           input_size=input_size)

    if balance_classes:
        w = sampler_weights(train_items, classes)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    dev = _resolve_device(device)
    model = build_model(len(classes))
    if pretrained:
        # ImageNet 预训练权重。数据量小的时候差别很大 —— 低层的边缘/笔画特征
        # 本来就是通用的，从头学纯属浪费样本。
        try:
            from torchvision import models
            ref = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            sd = ref.state_dict()
            sd.pop("fc.weight", None)
            sd.pop("fc.bias", None)     # 分类头形状不同，不能加载
            model.load_state_dict(sd, strict=False)
            LOGGER.info("已加载 ImageNet 预训练权重（fc 除外）")
        except Exception as exc:
            LOGGER.warning("拿不到预训练权重，从随机初始化开始: %s", exc)
    model = model.to(dev)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr * 10, total_steps=epochs * max(1, len(train_loader)),
        pct_start=0.25)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))

    best_acc, best_state, best_epoch = -1.0, None, 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tn, tc = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            tl += float(loss) * y.numel()
            tc += int((out.argmax(1) == y).sum())
            tn += y.numel()

        model.eval()
        vl, vn, vc = 0.0, 0, 0
        hits = np.zeros(len(classes), np.int64)
        totals = np.zeros(len(classes), np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                vl += float(criterion(out, y)) * y.numel()
                pred = out.argmax(1)
                vc += int((pred == y).sum())
                vn += y.numel()
                for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                    totals[t] += 1
                    hits[t] += int(t == p)

        tr_acc, va_acc = tc / max(tn, 1), vc / max(vn, 1)
        history.append({"epoch": epoch, "train_loss": tl / max(tn, 1),
                        "train_acc": tr_acc, "val_loss": vl / max(vn, 1),
                        "val_acc": va_acc, "lr": sched.get_last_lr()[0]})
        LOGGER.info("epoch %3d/%d  train %.4f/%.3f  val %.4f/%.3f",
                    epoch, epochs, tl / max(tn, 1), tr_acc, vl / max(vn, 1), va_acc)

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            per_class = {classes[i]: float(hits[i] / totals[i])
                         for i in range(len(classes)) if totals[i] > 0}
        elif early_stop_patience and epoch - best_epoch >= early_stop_patience:
            LOGGER.info("验证集 %d 轮没提升，提前停止", early_stop_patience)
            break

    if best_state is None:
        raise RuntimeError("一轮都没跑完，没有可保存的权重")
    model.load_state_dict(best_state)

    out_path = Path(out_path)
    save_checkpoint(
        out_path, model=model, classes=classes, val_accuracy=best_acc,
        input_size=input_size, mean=MEAN, std=STD,
        # 下面这些 jersey.py 会忽略，纯粹留档，方便日后追溯这个权重怎么来的
        trained_epochs=best_epoch, history=history, per_class_val_acc=per_class,
        data_dirs=[str(d) for d in data_dirs], aug_strength=aug_strength,
        n_train=len(train_items), n_val=len(val_items),
    )

    # 立刻按 jersey.py 的方式验一遍，别等部署才发现格式不对
    info = verify_checkpoint(out_path)
    LOGGER.info("checkpoint 契约校验通过")
    print("\n" + info.report())

    return TrainResult(checkpoint=out_path, classes=classes, best_val_acc=best_acc,
                       epochs_run=len(history), history=history,
                       per_class_acc=per_class,
                       elapsed_sec=time.perf_counter() - started)
