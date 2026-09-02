"""Per-digit jersey number recognition: model, label expansion, loss.

    from train.jersey_net import JerseyNet, expand_label, JerseyLoss

WHY PER-DIGIT AND NOT CLOSED-SET CLASSIFICATION
-----------------------------------------------
The number set is UNKNOWN and differs per video. FIBA 3x3 allows 0, 00 and
1-99 - 101 legal numbers - with four players per team and no duplicates within
a team, so any one match exposes eight of them. A closed-set head trained on
one match's eight numbers cannot name a ninth, and a 101-way head would leave
93 classes with zero examples; an untrained logit still wins an argmax
sometimes, which manufactures confident predictions for numbers the model has
never seen.

Recognising the DIGITS instead makes every sample pull its weight: an image
labelled "23" is a training example for "this is a 2" AND for "this is a 3",
so eight numbers still cover most of the ten digit shapes. Whole numbers are
reassembled at inference time, where SlotTracker accumulates per-digit
probabilities across frames, discovers each video's roster, and applies the
no-duplicates-within-a-team constraint. Single-frame accuracy is therefore not
the product - it only has to be good enough to feed that accumulator.

0 AND 00 ARE DIFFERENT NUMBERS. Both are legal and a team may field both, so
they are never normalised to the same label; the length head is what
distinguishes them.

THE LENGTH HEAD ENCODES "UNCLEAR" AS CLASS 0
--------------------------------------------
    class 0 -> number not readable (player turned, occluded, motion smear)
    class 1 -> one digit  ("0", "7")
    class 2 -> two digits ("00", "23")

so the class index is literally the digit count, and 0 - an impossible count -
carries the unreadable case. This matters because a large fraction of real
tracklets never show the number at all (SoccerNet marks these -1); training the
model to SAY so is more useful than hiding those frames, since a confident
"unreadable" contributes nothing to the cross-frame vote while a forced guess
actively poisons it.

The digit heads are supervised only where a digit exists: digit2 is ignored for
one-digit numbers, and both digit heads are ignored for unclear samples. That
is what ``ignore_index=-1`` buys - one loss expression covers all three sample
shapes with no branching.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

# The pipeline upscales every crop to this height before inference, so training
# at the same size means the model never sees a rescale it was not trained on.
INPUT_SIZE = 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Ignored by every head's cross entropy. Also the value SoccerNet uses in its
# ground truth for "number never visible in this tracklet", which is a happy
# coincidence but not the reason it was chosen.
IGNORE = -1

LENGTH_CLASSES = {
    0: "unclear - number not readable in this crop",
    1: "one digit  - 0-9",
    2: "two digits - 00-99",
}


# --------------------------------------------------------------------------
# Label expansion
# --------------------------------------------------------------------------


def expand_label(number: str | int | None) -> tuple[int, int, int]:
    """Turn a raw jersey number into ``(length, digit1, digit2)`` targets.

    THE ONE PLACE THIS CONVERSION HAPPENS. Every dataset - SVHN, synthetic,
    SoccerNet, and later the hand-annotated crops - funnels through here, so
    the three loaders cannot drift apart on how "00" or "unclear" is encoded.

        "0"  -> (1, 0, -1)      one digit, a zero
        "00" -> (2, 0,  0)      two digits, both zero - a DIFFERENT number
        "7"  -> (1, 7, -1)
        "23" -> (2, 2,  3)
        None -> (0, -1, -1)     unclear: only the length head learns anything

    ``None``, ``""`` and ``-1`` all mean unclear; -1 is spelled that way
    because it is what SoccerNet's ground truth files contain.
    """
    if number is None:
        return 0, IGNORE, IGNORE

    text = str(number).strip()
    if text == "" or text == "-1":
        return 0, IGNORE, IGNORE

    if not text.isdigit():
        raise ValueError(f"jersey number must be digits or an unclear marker, got {number!r}")

    # "007" is not a legal jersey number; catching it here stops a malformed
    # row from silently training digit1 on a truncated label.
    if len(text) not in (1, 2):
        raise ValueError(f"jersey number must be 1 or 2 digits, got {number!r}")

    if len(text) == 1:
        return 1, int(text), IGNORE
    return 2, int(text[0]), int(text[1])


def collapse_label(length: int, digit1: int, digit2: int) -> str | None:
    """Inverse of :func:`expand_label`; ``None`` for unclear.

    Kept next to its inverse so the round trip stays testable, and so the
    inference side has a canonical implementation to import rather than
    re-deriving the string format from the head semantics.
    """
    if length == 0:
        return None
    if length == 1:
        return str(digit1)
    return f"{digit1}{digit2}"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def _use_project_torch_home() -> None:
    """Cache ImageNet weights inside the repo so later runs are fully offline.

    The server has no outbound access during training; the download happens
    once, into a path that travels with the checkout, instead of into a
    per-user ~/.cache that a fresh node would not have.
    """
    if "TORCH_HOME" not in os.environ:
        cache = Path(__file__).resolve().parent.parent / ".torch_home"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache)


class JerseyNet(nn.Module):
    """ResNet18 trunk, three linear heads off the same 512-d pooled feature.

    One shared trunk rather than three models: the features that tell a 6 from
    a 9 are the features that tell a one-digit crop from a two-digit one, and
    at 96x96 the whole thing costs ~1-2 ms per crop, which is what lets the
    tracker afford to run it on every frame of every player.

    ``pretrained=False`` reconstructs the architecture without touching the
    network - that is the path the inference side and checkpoint loading use.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            _use_project_torch_home()
        import torchvision

        weights = "IMAGENET1K_V1" if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (N, 512, 1, 1)
        self.head_len = nn.Linear(512, 3)
        self.head_digit1 = nn.Linear(512, 10)
        self.head_digit2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.backbone(x).flatten(1)
        return self.head_len(f), self.head_digit1(f), self.head_digit2(f)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the trunk for the warmup epochs of stage 3.

        Fresh heads emit large, meaningless gradients on the first batches;
        letting those reach a trunk that already carries SVHN's digit features
        undoes exactly the thing stage 1 was for.
        """
        for p in self.backbone.parameters():
            p.requires_grad = trainable


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


class JerseyLoss(nn.Module):
    """Weighted sum of three cross entropies, each ignoring -1 targets.

    The masking is the whole trick and it is entirely delegated to
    ``ignore_index``:

      * SVHN pretraining passes length=-1 and digit2=-1, so only digit1 trains
      * a one-digit number passes digit2=-1, so digit2 trains on nothing
      * an unclear sample passes both digits as -1, so only length trains

    No batch ever needs to be split by sample type. The one case to watch is a
    batch where a head sees *no* valid target at all - PyTorch returns NaN
    there rather than 0, so each term is guarded below.
    """

    def __init__(self, w_len: float = 1.0, w_digit1: float = 1.0, w_digit2: float = 1.0) -> None:
        super().__init__()
        self.w = (w_len, w_digit1, w_digit2)
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=outputs[0].device)
        parts: dict[str, float] = {}
        for name, weight, logits, target in zip(
            ("len", "digit1", "digit2"), self.w, outputs, targets
        ):
            if (target != IGNORE).any():
                term = self.ce(logits, target)
            else:
                # Whole batch ignored for this head - contribute nothing, and
                # keep it attached to the graph so DDP sees every parameter.
                term = logits.sum() * 0.0
            total = total + weight * term
            parts[name] = float(term.detach())
        return total, parts


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def save_checkpoint(
    model: JerseyNet,
    path: str | Path,
    *,
    stage: str,
    data_version: str,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Write ``path`` plus a same-named .json describing how to use it.

    The inference side reads the sidecar for input size, normalization and head
    semantics instead of repeating them in its own source. Two hardcoded copies
    of "96" and the ImageNet constants is exactly how a preprocessing mismatch
    ships unnoticed - the model still returns confident logits, they are just
    wrong.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    sidecar = {
        "format": "jersey-per-digit-v1",
        "stage": stage,
        "data_version": data_version,
        "architecture": "resnet18",
        "input_size": INPUT_SIZE,
        "input_layout": "RGB, HWC->CHW, float in [0,1] then normalized",
        "normalize": {"mean": list(MEAN), "std": list(STD)},
        "heads": {
            "length": {"outputs": 3, "classes": LENGTH_CLASSES},
            "digit1": {"outputs": 10, "classes": "digit 0-9, first/left digit"},
            "digit2": {"outputs": 10, "classes": "digit 0-9, second digit; undefined when length==1"},
        },
        "ignore_index": IGNORE,
        "decode": "length==0 -> unclear; length==1 -> str(digit1); length==2 -> str(digit1)+str(digit2)",
        "legal_numbers": "FIBA 3x3: 0, 00, 1-99 (101 total)",
        "metrics": metrics or {},
    }
    if extra:
        sidecar.update(extra)

    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return json_path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> JerseyNet:
    """Rebuild the architecture offline and load weights into it."""
    model = JerseyNet(pretrained=False)
    state = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)
