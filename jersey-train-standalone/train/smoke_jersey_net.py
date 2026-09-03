"""Forward/label/loss smoke test for jersey_net + augment.

    python train/smoke_jersey_net.py

Runs on random tensors and needs no dataset - the point is to catch a shape or
ignore-index mistake in seconds, before a download or a training run pays for
it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from augment import build_transforms
from jersey_net import (
    INPUT_SIZE,
    JerseyLoss,
    JerseyNet,
    collapse_label,
    expand_label,
)


def check_labels() -> None:
    cases = {
        "0": (1, 0, -1),
        "00": (2, 0, 0),
        "7": (1, 7, -1),
        "23": (2, 2, 3),
        "99": (2, 9, 9),
        None: (0, -1, -1),
        -1: (0, -1, -1),
        "": (0, -1, -1),
    }
    for raw, want in cases.items():
        got = expand_label(raw)
        assert got == want, f"expand_label({raw!r}) -> {got}, want {want}"

    # 0 and 00 must not collapse onto each other anywhere in the round trip.
    assert collapse_label(*expand_label("0")) == "0"
    assert collapse_label(*expand_label("00")) == "00"
    assert collapse_label(*expand_label("23")) == "23"
    assert collapse_label(*expand_label(None)) is None

    for bad in ("007", "1a", "-5"):
        try:
            expand_label(bad)
        except ValueError:
            continue
        raise AssertionError(f"expand_label({bad!r}) should have raised")
    print("labels        OK  (0/00 distinct, unclear -> length class 0)")


def check_augment() -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (40, 30, 3), dtype=np.uint8))

    train_tf, val_tf = build_transforms(True), build_transforms(False)
    for name, tf in (("train", train_tf), ("val", val_tf)):
        out = tf(image)
        assert out.shape == (3, INPUT_SIZE, INPUT_SIZE), f"{name} -> {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"

    # Every transform must be run repeatedly without exploding on edge cases
    # (tiny kernels, extreme downscale draws).
    for _ in range(40):
        train_tf(image)

    names = " ".join(type(t).__name__ for t in train_tf.transforms)
    assert "Flip" not in names, f"a flip made it into the train pipeline: {names}"
    print("augment       OK  (96x96, finite, no flips, 40 random draws survived)")


def check_forward_and_loss() -> None:
    torch.manual_seed(0)
    model = JerseyNet(pretrained=False)
    x = torch.randn(6, 3, INPUT_SIZE, INPUT_SIZE)
    logits = model(x)
    assert [tuple(t.shape) for t in logits] == [(6, 3), (6, 10), (6, 10)]

    criterion = JerseyLoss()

    # A batch mixing all three sample shapes: two-digit, one-digit, unclear,
    # plus SVHN-style rows where only digit1 is supervised.
    targets = (
        torch.tensor([2, 1, 0, 2, -1, -1]),
        torch.tensor([2, 7, -1, 0, 4, 8]),
        torch.tensor([3, -1, -1, 0, -1, -1]),
    )
    loss, parts = criterion(logits, targets)
    assert torch.isfinite(loss), f"mixed batch gave {loss}"
    loss.backward()
    assert model.head_digit2.weight.grad is not None
    print(f"forward+loss  OK  mixed batch loss={float(loss):.4f} parts={parts}")

    # The SVHN stage sends whole batches with no length or digit2 target at
    # all; that must be 0, not NaN.
    model.zero_grad()
    svhn_targets = (
        torch.full((6,), -1),
        torch.tensor([1, 2, 3, 4, 5, 6]),
        torch.full((6,), -1),
    )
    loss, parts = criterion(model(x), svhn_targets)
    assert torch.isfinite(loss), f"all-ignored heads gave {loss}"
    assert parts["len"] == 0.0 and parts["digit2"] == 0.0
    loss.backward()
    print(f"svhn-shaped   OK  loss={float(loss):.4f} parts={parts}")


def check_freeze() -> None:
    model = JerseyNet(pretrained=False)
    model.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert model.head_len.weight.requires_grad
    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.backbone.parameters())
    print("freeze/unfreeze OK")


if __name__ == "__main__":
    check_labels()
    check_augment()
    check_forward_and_loss()
    check_freeze()
    print("\nall smoke checks passed")
