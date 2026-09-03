"""Metrics and confusion reporting shared by every training stage.

    from evaluate import evaluate, format_report

WHY MACRO-F1 PER DIGIT AND NOT WHOLE-NUMBER ACCURACY
----------------------------------------------------
Whole-number accuracy flatters a model on this data and does it in a way that
is invisible from the number itself. Jersey numbers are heavily long-tailed -
single digits and the low twenties dominate real rosters - so a model that
learns the common numbers and gives up on the rest still scores well, while
being useless for exactly the videos where a rare number shows up. Worse, the
metric is dominated by whichever numbers happen to be frequent in the
validation split, so it moves when the split changes and not when the model
does.

Per-digit macro-F1 averages over the ten digit classes with equal weight, so a
digit the model has quietly abandoned drags the score down no matter how rare
it was in training. Averaging digit1's and digit2's macro-F1 gives the single
number used for checkpoint selection. Whole-number exact match is still
reported, because it is the quantity a human has intuitions about - it is just
not allowed to pick the checkpoint.

Accuracy is not reported as a headline for the same reason: with unclear
samples in the mix, a model that predicts "unclear" often enough can post a
respectable accuracy while never reading a number.

CONFUSION PAIRS
---------------
The report calls out the pairs that actually cost this project accuracy, and
they are not arbitrary:

    6 <-> 9   the same glyph rotated - the one pair a vertical flip creates,
              and the reason no flip augmentation exists in this codebase
    0 <-> 6 <-> 8   closed loops, indistinguishable once the strokes merge
    1 <-> 7   a single crossbar, first thing lost to blur
    3 <-> 8   left side open or closed
    5 <-> 6   upper-left stroke closes or does not

At 20-40 px these are where the errors concentrate, and a model can post a
decent macro-F1 while being systematically wrong on one of them. Printing them
explicitly means a regression shows up as a number rather than as a surprise
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from jersey_net import IGNORE, LENGTH_CLASSES

# Pairs worth watching, in the order the report prints them.
CONFUSION_PAIRS: tuple[tuple[int, int, str], ...] = (
    (6, 9, "rotation - never flip"),
    (0, 6, "closed loop"),
    (0, 8, "closed loop"),
    (6, 8, "closed loop"),
    (1, 7, "crossbar lost to blur"),
    (3, 8, "left side open/closed"),
    (5, 6, "upper-left stroke"),
)


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------


def confusion_matrix(true: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are ground truth, columns are prediction. Ignored targets dropped."""
    mask = true != IGNORE
    true, pred = true[mask], pred[mask]
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def macro_f1(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Macro-F1 over classes with non-zero support, plus the per-class F1s.

    Classes with no ground-truth examples are excluded rather than scored as
    zero: during SVHN pretraining digit2 has no support at all, and counting
    ten absent classes as failures would report 0.0 for a head that was never
    asked to do anything.
    """
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, correct / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, correct / np.maximum(support, 1), 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0.0)

    present = support > 0
    return (float(f1[present].mean()) if present.any() else 0.0), f1


@dataclass
class EvalResult:
    """Everything one validation pass produced, ready to print or to log."""

    cm_length: np.ndarray
    cm_digit1: np.ndarray
    cm_digit2: np.ndarray
    f1_length: float
    f1_digit1: float
    f1_digit2: float
    per_class_digit1: np.ndarray
    per_class_digit2: np.ndarray
    exact_match: float
    n_samples: int
    n_readable: int
    losses: dict[str, float] = field(default_factory=dict)

    @property
    def selection_metric(self) -> float:
        """Mean of the two per-digit macro-F1s. THE checkpoint selector.

        Length F1 is deliberately excluded: it is the easiest of the three
        heads and including it would let a model coast on length while both
        digit heads degrade.
        """
        return (self.f1_digit1 + self.f1_digit2) / 2.0


# --------------------------------------------------------------------------
# Validation pass
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> EvalResult:
    """Run the model over ``loader`` and collect every metric in one pass.

    Whole-number exact match is scored only over samples whose ground truth is
    readable; counting unclear samples as "correct when predicted unclear"
    would let the metric be inflated by the easiest possible decision.
    """
    model.eval()

    true_len, pred_len = [], []
    true_d1, pred_d1 = [], []
    true_d2, pred_d2 = [], []
    exact_hits = exact_total = 0
    loss_sums: dict[str, float] = {}
    n_batches = 0

    for images, t_len, t_d1, t_d2 in loader:
        images = images.to(device, non_blocking=True)
        t_len, t_d1, t_d2 = t_len.to(device), t_d1.to(device), t_d2.to(device)

        logits = model(images)
        if criterion is not None:
            _, parts = criterion(logits, (t_len, t_d1, t_d2))
            for key, value in parts.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + value
            n_batches += 1

        p_len, p_d1, p_d2 = (logit.argmax(1) for logit in logits)

        true_len.append(t_len.cpu().numpy())
        pred_len.append(p_len.cpu().numpy())
        true_d1.append(t_d1.cpu().numpy())
        pred_d1.append(p_d1.cpu().numpy())
        true_d2.append(t_d2.cpu().numpy())
        pred_d2.append(p_d2.cpu().numpy())

        # Exact match: the length must be right AND every digit that length
        # implies must be right. A two-digit number scored on digit1 alone
        # would count "27" as a hit for "23".
        readable = t_len > 0
        if readable.any():
            hit = (p_len == t_len)
            hit &= (p_d1 == t_d1)
            hit &= torch.where(t_len == 2, p_d2 == t_d2, torch.ones_like(hit))
            exact_hits += int((hit & readable).sum())
            exact_total += int(readable.sum())

    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    cm_length = confusion_matrix(cat(true_len), cat(pred_len), 3)
    cm_digit1 = confusion_matrix(cat(true_d1), cat(pred_d1), 10)
    cm_digit2 = confusion_matrix(cat(true_d2), cat(pred_d2), 10)

    f1_length, _ = macro_f1(cm_length)
    f1_digit1, per_d1 = macro_f1(cm_digit1)
    f1_digit2, per_d2 = macro_f1(cm_digit2)

    return EvalResult(
        cm_length=cm_length,
        cm_digit1=cm_digit1,
        cm_digit2=cm_digit2,
        f1_length=f1_length,
        f1_digit1=f1_digit1,
        f1_digit2=f1_digit2,
        per_class_digit1=per_d1,
        per_class_digit2=per_d2,
        exact_match=(exact_hits / exact_total) if exact_total else 0.0,
        n_samples=int(cat(true_len).size),
        n_readable=exact_total,
        losses={k: v / max(n_batches, 1) for k, v in loss_sums.items()},
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _render_matrix(matrix: np.ndarray, labels: list[str], title: str) -> list[str]:
    """Counts with row-normalised percentages; empty rows collapse to a dash."""
    width = max(5, max(len(l) for l in labels) + 1)
    lines = [title, "  " + " " * width + "pred ->"]
    lines.append("  " + "true".ljust(width) + "".join(l.rjust(7) for l in labels) + "     n")

    for i, label in enumerate(labels):
        support = matrix[i].sum()
        if support == 0:
            lines.append("  " + label.ljust(width) + "  (no validation examples)")
            continue
        cells = []
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            if count == 0:
                cells.append("      .")
            else:
                cells.append(f"{100.0 * count / support:6.1f}%")
        lines.append("  " + label.ljust(width) + "".join(cells) + f"{support:6d}")
    return lines


def _render_pairs(cm_digit1: np.ndarray, cm_digit2: np.ndarray) -> list[str]:
    """Two-way leakage on the pairs this project actually loses accuracy to."""
    lines = ["confusable pairs (both digit heads pooled, % of that digit's support)"]
    pooled = cm_digit1 + cm_digit2
    support = pooled.sum(axis=1)

    for a, b, why in CONFUSION_PAIRS:
        if support[a] == 0 and support[b] == 0:
            continue
        a_to_b = 100.0 * pooled[a, b] / support[a] if support[a] else float("nan")
        b_to_a = 100.0 * pooled[b, a] / support[b] if support[b] else float("nan")
        worst = max(x for x in (a_to_b, b_to_a) if not np.isnan(x))
        flag = "  <-- WATCH" if worst >= 5.0 else ""
        lines.append(
            f"  {a} -> {b}: {a_to_b:5.1f}%   {b} -> {a}: {b_to_a:5.1f}%"
            f"   ({why}){flag}"
        )
    return lines


def format_report(result: EvalResult, header: str = "") -> str:
    """The full block printed at the end of every epoch's validation."""
    digits = [str(d) for d in range(10)]
    lines: list[str] = []

    if header:
        lines += ["=" * 78, header, "=" * 78]

    lines.append(
        f"samples {result.n_samples}  (readable {result.n_readable}, "
        f"unclear {result.n_samples - result.n_readable})"
    )
    if result.losses:
        lines.append("val loss   " + "  ".join(f"{k}={v:.4f}" for k, v in result.losses.items()))
    lines.append("")

    lines.append(
        f"SELECTION METRIC  mean per-digit macro-F1 = {result.selection_metric:.4f}"
        f"   (digit1 {result.f1_digit1:.4f}, digit2 {result.f1_digit2:.4f})"
    )
    lines.append(f"length head macro-F1 = {result.f1_length:.4f}   (not used for selection)")
    lines.append(
        f"whole-number exact match = {result.exact_match:.4f}"
        f"   (REFERENCE ONLY - long tail makes this misleading)"
    )
    lines.append("")

    length_labels = [f"{i}:{LENGTH_CLASSES[i].split(' ')[0]}" for i in range(3)]
    lines += _render_matrix(result.cm_length, length_labels, "length head (3x3)")
    lines.append("")
    lines += _render_matrix(result.cm_digit1, digits, "digit1 head (10x10)")
    lines.append("")
    lines += _render_matrix(result.cm_digit2, digits, "digit2 head (10x10)")
    lines.append("")
    lines += _render_pairs(result.cm_digit1, result.cm_digit2)

    weakest = int(np.argmin(np.where(result.cm_digit1.sum(1) > 0, result.per_class_digit1, np.inf)))
    if result.cm_digit1.sum(1)[weakest] > 0:
        lines.append("")
        lines.append(
            f"weakest digit1 class: {weakest} (F1 {result.per_class_digit1[weakest]:.3f})"
        )

    return "\n".join(lines)
