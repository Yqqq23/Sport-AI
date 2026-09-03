"""The training loop shared by stage 1 (SVHN) and stage 3 (SoccerNet).

One loop, two callers. The stages differ only in which data they mount and
whether they warm up with a frozen backbone, and both of those are arguments -
duplicating the loop would mean a fix to the checkpoint-selection logic landing
in one stage and not the other.

CHECKPOINT SELECTION IS ON MEAN PER-DIGIT MACRO-F1
--------------------------------------------------
Not on validation loss, and not on whole-number accuracy. Loss is dominated by
whichever head has the most supervised targets in the batch, so it moves for
reasons unrelated to whether the model reads digits better. See evaluate.py for
why accuracy is the wrong selector.

The last epoch is not assumed to be the best one. Cosine decay usually makes it
so, but "usually" is not a reason to overwrite a better checkpoint from epoch
23 with a worse one from epoch 30.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluate import evaluate, format_report
from jersey_net import save_checkpoint


def _log(message: str, log_file: Path | None) -> None:
    """Print AND append, with a flush after each line.

    Training runs under nohup where stdout is block-buffered; without the
    flush a `tail -f` shows nothing for minutes and looks like a hang.
    """
    print(message, flush=True)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    out_path: str | Path,
    stage: str,
    data_version: str,
    weight_decay: float = 1e-4,
    freeze_backbone_epochs: int = 0,
    log_file: str | Path | None = None,
    log_every: int = 50,
    extra_sidecar: dict | None = None,
    diagnostic_loaders: dict | None = None,
):
    """Train, validate each epoch, keep the best checkpoint. Returns the best result.

    ``val_loader`` selects the checkpoint. ``diagnostic_loaders`` are scored and
    logged but deliberately cannot influence selection - stage 3 uses that to
    watch synthetic performance without letting a rendered-font score decide
    which weights ship.
    """
    out_path = Path(out_path)
    log_file = Path(log_file) if log_file else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    _log(
        f"stage={stage}  epochs={epochs}  lr={lr}  device={device}  amp={use_amp}\n"
        f"train batches={len(train_loader)}  val batches={len(val_loader)}\n"
        f"freeze backbone for first {freeze_backbone_epochs} epoch(s)",
        log_file,
    )

    best_metric = -1.0
    best_result = None
    frozen = None

    for epoch in range(epochs):
        # Fresh heads emit large meaningless gradients on the first batches;
        # letting those reach a trunk that already carries SVHN's digit
        # features undoes the stage it was pretrained for.
        should_freeze = epoch < freeze_backbone_epochs
        if should_freeze != frozen:
            model.set_backbone_trainable(not should_freeze)
            frozen = should_freeze
            _log(f"[epoch {epoch}] backbone {'FROZEN' if should_freeze else 'unfrozen'}", log_file)

        model.train()
        started = time.time()
        running = 0.0
        seen = 0

        for step, (images, t_len, t_d1, t_d2) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = (t_len.to(device), t_d1.to(device), t_d2.to(device))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, parts = criterion(model(images), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach())
            seen += 1
            if log_every and step % log_every == 0:
                detail = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                _log(
                    f"[epoch {epoch} step {step}/{len(train_loader)}] "
                    f"loss={float(loss.detach()):.4f} {detail} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    log_file,
                )

        scheduler.step()
        result = evaluate(model, val_loader, device, criterion)
        metric = result.selection_metric

        _log(
            f"\n[epoch {epoch}] train_loss={running / max(seen, 1):.4f}  "
            f"val_selection_metric={metric:.4f}  "
            f"exact={result.exact_match:.4f}  "
            f"({time.time() - started:.0f}s)",
            log_file,
        )

        for name, loader in (diagnostic_loaders or {}).items():
            diagnostic = evaluate(model, loader, device, criterion)
            _log(
                f"[epoch {epoch}] {name}: metric={diagnostic.selection_metric:.4f} "
                f"digit1={diagnostic.f1_digit1:.4f} digit2={diagnostic.f1_digit2:.4f} "
                f"length={diagnostic.f1_length:.4f} exact={diagnostic.exact_match:.4f} "
                f"(diagnostic only)",
                log_file,
            )

        if metric > best_metric:
            best_metric, best_result = metric, result
            save_checkpoint(
                model,
                out_path,
                stage=stage,
                data_version=data_version,
                metrics={
                    "selection_metric_mean_per_digit_macro_f1": metric,
                    "macro_f1_digit1": result.f1_digit1,
                    "macro_f1_digit2": result.f1_digit2,
                    "macro_f1_length": result.f1_length,
                    "whole_number_exact_match": result.exact_match,
                    "epoch": epoch,
                    "val_samples": result.n_samples,
                },
                extra=extra_sidecar,
            )
            _log(f"[epoch {epoch}] new best -> {out_path} (and {out_path.with_suffix('.json')})", log_file)

    _log("\n" + format_report(best_result, f"FINAL - {stage} - best checkpoint"), log_file)
    return best_result
