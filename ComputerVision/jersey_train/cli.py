"""命令行入口。

推荐路线（对你这个机位最有效）—— 标注走网页，不走 label 子命令
    python -m jersey_train.cli mine --video game.mp4 --out data/mined \
        --annotation-db <网页>/backend/jersey_active_learning.db --video-id 88
    python <网页>/scripts/import_crops.py --manifest data/mined/crops/manifest.jsonl --video-id 88
    # 在网页里逐帧手工标注，然后导出
    python -m jersey_train.cli train --data <导出目录> --out train/jersey_classifier.pt
    python -m jersey_train.cli verify --ckpt train/jersey_classifier.pt

``label`` 子命令是旧的键盘标注器，读的是 ``slot_XXX/`` 布局。mine 现在输出的是
``crops/<video>/f%06d__box_NN.png`` + manifest.jsonl（给网页导入用），两者不通用。

公开数据集路线（需要 Roboflow API key，域差距大，建议和上面混用）
    export ROBOFLOW_API_KEY=...
    python -m jersey_train.cli download --out data/jersey_ocr
    python -m jersey_train.cli train --data data/jersey_ocr --data data/labelled
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)


def cmd_fetch(args) -> int:
    from .datasets import fetch
    out = fetch(args.source, args.out,
                splits=tuple(args.splits), max_per_tracklet=args.max_per_tracklet,
                include_no_number=not args.drop_no_number,
                skip_download=args.skip_download)
    print(f"\n下一步: python -m jersey_train.cli train --data {out}")
    return 0


def cmd_download(args) -> int:
    from .download import run
    run(args.out, api_key_env=args.api_key_env, skip_download=args.skip_download)
    print(f"\n下一步: python -m jersey_train.cli train --data {args.out}")
    return 0


def cmd_mine(args) -> int:
    from .mine import MineConfigurationError, mine
    try:
        s = mine(video=args.video, out_dir=args.out, repo=args.repo,
                 config_path=args.config, start_frame=args.start_frame,
                 end_frame=args.end_frame, stride=args.stride,
                 annotation_db=args.annotation_db, video_id=args.video_id,
                 max_per_track=args.max_per_track,
                 allow_inexact_seek=args.allow_inexact_seek)
    except MineConfigurationError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("\n" + "=" * 70)
    print(s.report())
    print("=" * 70)
    print("下一步（网页标注）:")
    print(f"    python scripts/import_crops.py --manifest {s.manifest_path} \\")
    print(f"        --video-id {args.video_id if args.video_id is not None else '<网页里的 video_id>'}")
    return 0


def cmd_label(args) -> int:
    from .label import run
    n = run(args.src, args.out, batch_size=args.batch_size, move=args.move)
    if n:
        print(f"\n下一步: python -m jersey_train.cli train --data {args.out}")
    return 0


def cmd_train(args) -> int:
    from .train import train
    result = train(
        data_dirs=args.data, out_path=args.out, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, val_fraction=args.val_fraction,
        device=args.device, num_workers=args.workers, seed=args.seed,
        aug_strength=args.aug_strength, balance_classes=not args.no_balance,
        pretrained=not args.no_pretrained, early_stop_patience=args.patience)
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(result.report())
    print("=" * 70)
    Path(args.out).with_suffix(".history.json").write_text(
        json.dumps({"history": result.history,
                    "per_class_val_acc": result.per_class_acc,
                    "classes": result.classes,
                    "best_val_acc": result.best_val_acc}, indent=2),
        encoding="utf-8")
    print("\n启用方法：把 basketball-v2 的 configs/default.yaml 改成")
    print("    jersey:")
    print("      backend: local          # 或 auto")
    print("      local:")
    print(f"        weights: {Path(args.out).resolve()}")
    return 0


def cmd_verify(args) -> int:
    from .contract import CheckpointContractError, verify_checkpoint
    try:
        info = verify_checkpoint(args.ckpt)
    except CheckpointContractError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("[ok] checkpoint 符合 LocalJerseyClassifier 的加载契约")
    print(info.report())
    if args.numbers:
        want = [n.strip() for n in args.numbers.split(",") if n.strip()]
        missing = sorted(set(want) - set(info.classes), key=lambda s: int(s))
        if missing:
            print(f"\n⚠ 这些号码这个权重**预测不出来**（softmax 里没有对应输出）: {missing}")
            print("  不是「不太可能」，是物理上不可能。需要补这些号的训练数据。")
        else:
            print(f"\n[ok] {want} 全部在类别表里")
    return 0


def cmd_predict(args) -> int:
    """拿真实 crop 试一下 —— 验证集分数好看不代表真视频上能用。"""
    import cv2
    import numpy as np
    import torch

    from .contract import build_model, preprocess_bgr, verify_checkpoint

    info = verify_checkpoint(args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(len(info.classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    root = Path(args.images)
    files = ([root] if root.is_file() else
             sorted(f for f in root.rglob("*")
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}))
    if not files:
        print(f"{root} 里没有图片")
        return 1

    print(f"{'file':<44}{'pred':>6}{'conf':>8}{'margin':>8}")
    print("-" * 70)
    n_ok = 0
    for f in files[: args.limit]:
        img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x = torch.from_numpy(preprocess_bgr(img, info.input_size, info.mean,
                                            info.std)).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0]
        top = torch.topk(p, k=min(2, p.numel()))
        conf = float(top.values[0])
        margin = conf - (float(top.values[1]) if top.values.numel() > 1 else 0.0)
        label = info.classes[int(top.indices[0])]
        # 和 LocalJerseyClassifier 的门槛一致，看清楚有多少会被真的采纳
        accepted = conf >= args.min_confidence and margin >= args.min_margin
        n_ok += accepted
        print(f"{f.name[:44]:<44}{label:>6}{conf:>8.3f}{margin:>8.3f}"
              + ("" if accepted else "   rejected"))
    print("-" * 70)
    print(f"通过 min_confidence={args.min_confidence} / min_margin={args.min_margin} 的: "
          f"{n_ok}/{min(len(files), args.limit)}")
    print("通过率太低说明域差距还在 —— 补自己视频的数据，别调阈值。")
    return 0


def cmd_check(args) -> int:
    """上云之前先跑这个。路径、依赖、检测器权重、已有 checkpoint 一次看清。"""
    from pathlib import Path as P

    here = P(__file__).resolve().parent
    project = here.parent
    ok = True
    print("=" * 74)
    print(f"jersey_train : {here}")
    print(f"项目根       : {project}   （从本文件位置推导，不写死）")
    print(f"BASKETBALL_V2: {os.environ.get('BASKETBALL_V2') or '(未设置，靠自动查找)'}")
    print("-" * 74)

    # --- 同级的 trajectory 包（mine 子命令要用）
    traj = project / "trajectory"
    print(f"[{'ok' if traj.is_dir() else 'warn'}]   trajectory/  {traj}"
          + ("" if traj.is_dir() else "   缺失 -> mine 跑不了，train/fetch/verify 不受影响"))

    # --- basketball-v2 本体
    src = None
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    try:
        from trajectory.bridge import locate_repo
        src = locate_repo(args.repo)
        print(f"[ok]   basketball-v2 src = {src}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] 定位不到 basketball-v2:\n{exc}")

    # --- 配置 + 检测器权重
    if src is not None:
        cfg = P(args.config) if args.config else src.parent / "configs" / "default.yaml"
        print(f"[{'ok' if cfg.is_file() else 'FAIL'}]   config = {cfg}")
        if not cfg.is_file():
            ok = False
        else:
            try:
                import re
                text = cfg.read_text(encoding="utf-8")
                for label, pat in (("detector.weights", r"weights:\s*(\S+best\.pt|\S+\.pt)"),
                                   ("jersey.local.weights", r"weights:\s*(\S*jersey\S*\.pt)")):
                    for m in re.finditer(pat, text):
                        w = m.group(1).strip()
                        p = P(w) if P(w).is_absolute() else (cfg.parent / w)
                        exists = p.exists()
                        tag = "ok" if exists else "warn"
                        print(f"[{tag}]   {label}: {w}")
                        if not exists:
                            print(f"         -> 解析为 {p}  **不存在**")
            except Exception as exc:
                print(f"[warn] 读 config 失败: {exc}")

    # --- 已有的 checkpoint
    ck = P(args.ckpt)
    if ck.is_file():
        try:
            from .contract import verify_checkpoint
            info = verify_checkpoint(ck)
            print(f"[ok]   已有 checkpoint 通过契约校验")
            print(info.report())
        except Exception as exc:
            ok = False
            print(f"[FAIL] {ck} 存在但不符合契约: {exc}")
    else:
        print(f"[note] 还没有 {ck} —— 就是本包要造的东西")

    # --- 依赖
    def probe(mod):
        try:
            m = __import__(mod)
            return True, str(getattr(m, "__version__", ""))
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    for mod in ("torch", "torchvision", "cv2", "numpy"):
        good, info = probe(mod)
        print(f"[{'ok' if good else 'FAIL'}]   {mod} {info}")
        if not good:
            ok = False
    for mod, why in (("SoccerNet", "fetch --source soccernet 需要"),
                     ("requests", "fetch --source roboflow 需要"),
                     ("ultralytics", "mine 需要（要跑检测器）")):
        good, info = probe(mod)
        print(f"[ok]   {mod} {info}" if good else f"[warn] 没装 {mod}（{why}）")

    try:
        import torch
        print(f"[{'ok' if torch.cuda.is_available() else 'warn'}]   CUDA "
              f"{'可用: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用（CPU 训练会很慢）'}")
    except Exception:
        pass

    print("=" * 74)
    print("核心就位" if ok else "有缺失，见上面 FAIL")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jersey_train",
        description="训练球衣号分类器，产出 basketball-v2 能直接加载的 jersey_classifier.pt")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="环境自检（上云前先跑这个）")
    c.add_argument("--repo", default=None, help="basketball-v2 路径（或用 BASKETBALL_V2）")
    c.add_argument("--config", default=None)
    c.add_argument("--ckpt", default="train/jersey_classifier.pt")
    c.set_defaults(func=cmd_check)

    f = sub.add_parser("fetch", help="下载公开数据集（soccernet 无需 key，推荐）")
    f.add_argument("--source", choices=["soccernet", "roboflow"], default="soccernet")
    f.add_argument("--out", default="data/soccernet")
    f.add_argument("--splits", nargs="+", default=["train", "test"],
                   help="SoccerNet 的 split，默认 train test（challenge 没标注，不用下）")
    f.add_argument("--max-per-tracklet", type=int, default=25,
                   help="每条 tracklet 最多取几帧（均匀取样）。不限量的话少数球员会主导整个数据集")
    f.add_argument("--drop-no-number", action="store_true",
                   help="丢掉「号码不可见」的 tracklet。默认保留并映射成 'none' 弃权类")
    f.add_argument("--skip-download", action="store_true", help="只对已下载的数据重跑转换")
    f.set_defaults(func=cmd_fetch)

    d = sub.add_parser("download", help="[旧] 只下 Roboflow，等价于 fetch --source roboflow")
    d.add_argument("--out", default="data/jersey_ocr")
    d.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    d.add_argument("--skip-download", action="store_true", help="只对已解压的目录重跑清洗")
    d.set_defaults(func=cmd_download)

    m = sub.add_parser("mine", help="从你自己的视频里挖球衣 crop（供网页标注）")
    m.add_argument("--video", required=True)
    m.add_argument("--out", default="data/mined")
    m.add_argument("--repo", default=None, help="basketball-v2 路径（或用 BASKETBALL_V2）")
    m.add_argument("--config", default=None)
    m.add_argument("--start-frame", type=int, default=0,
                   help="默认 0，别改。非 0 会走 seek，落点可能不精确，见 --allow-inexact-seek")
    m.add_argument("--end-frame", type=int, default=None)
    # 没有默认值是有意的。stride 必须整除网页的 frame_interval，硬编码一个数
    # 在 frame_interval 改掉的那天会静默错位（挖出来的 crop 对不上任何网页帧）。
    m.add_argument("--stride", type=int, default=None,
                   help="每隔几帧挖一次。不给就从 --annotation-db 里读该视频的 frame_interval")
    m.add_argument("--annotation-db", default=None,
                   help="网页后端的 jersey_active_learning.db，用来读 frame_interval")
    m.add_argument("--video-id", type=int, default=None,
                   help="该视频在网页库里的 id，配合 --annotation-db 用")
    m.add_argument("--max-per-track", type=int, default=80,
                   help="每个 track 最多存多少张，防止一个人刷屏")
    m.add_argument("--allow-inexact-seek", action="store_true",
                   help="允许 --start-frame 非 0。会带来帧号错位风险，看代码里的说明再用")
    m.set_defaults(func=cmd_mine)

    l = sub.add_parser("label", help="给挖出来的 crop 打标签（键盘操作）")
    l.add_argument("--src", default="data/mined")
    l.add_argument("--out", default="data/labelled")
    l.add_argument("--batch-size", type=int, default=40)
    l.add_argument("--move", action="store_true", help="移动而不是复制（省磁盘）")
    l.set_defaults(func=cmd_label)

    t = sub.add_parser("train", help="训练分类器")
    t.add_argument("--data", action="append", required=True,
                   help="数据目录，可给多次（manifest.json 或 <label>/ 目录树）")
    t.add_argument("--out", default="train/jersey_classifier.pt")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val-fraction", type=float, default=0.15)
    t.add_argument("--device", default="cuda")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--aug-strength", type=float, default=1.0,
                   help="0=不增强, 1=默认, >1=更狠")
    t.add_argument("--no-balance", action="store_true", help="关掉类别均衡采样")
    t.add_argument("--no-pretrained", action="store_true", help="不用 ImageNet 预训练")
    t.add_argument("--patience", type=int, default=12, help="早停耐心（轮）")
    t.set_defaults(func=cmd_train)

    vf = sub.add_parser("verify", help="按 jersey.py 的方式校验 checkpoint")
    vf.add_argument("--ckpt", default="train/jersey_classifier.pt")
    vf.add_argument("--numbers", default=None,
                    help="逗号分隔，检查这些号码在不在类别表里，例如 0,6,12")
    vf.set_defaults(func=cmd_verify)

    pr = sub.add_parser("predict", help="拿真实 crop 试一下（含置信度门槛）")
    pr.add_argument("--ckpt", default="train/jersey_classifier.pt")
    pr.add_argument("--images", required=True, help="图片或目录")
    pr.add_argument("--limit", type=int, default=40)
    pr.add_argument("--min-confidence", type=float, default=0.50)
    pr.add_argument("--min-margin", type=float, default=0.15)
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _log(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
