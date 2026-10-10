"""命令行入口。

    # 1. 抽取（慢，跑一次就行）
    python -m trajectory.cli extract --video "game.mp4" --out outputs/run1

    # 2. 看看识别到了哪些球员
    python -m trajectory.cli players --csv outputs/run1/tracks.csv

    # 3. 画某个球衣号的轨迹
    python -m trajectory.cli plot --csv outputs/run1/tracks.csv --jersey 7
    python -m trajectory.cli plot --csv outputs/run1/tracks.csv --jersey 7 --video
    python -m trajectory.cli plot --csv outputs/run1/tracks.csv --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Support `python3 /path/to/trajectory/cli.py ...` from any working directory.
# Module execution (`python3 -m trajectory.cli`) still works unchanged.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = "trajectory"

from .court import CourtProjector


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
def cmd_extract(args) -> int:
    from .extract import extract

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tracks.csv"
    summary = extract(
        video=args.video,
        out_csv=csv_path,
        repo=args.repo,
        config_path=args.config,
        calibration=args.calibration,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        duration_sec=args.duration_sec,
        stride=args.stride,
        log_every=args.log_every,
        annotated_video=args.annotated_video,
        write_annotated=not args.no_annotated_video,
        annotated_fourcc=args.annotated_fourcc,
        annotated_scale=args.annotated_scale,
    )
    print("\n" + "=" * 70)
    print("抽取完成")
    print("=" * 70)
    print(summary.report())
    print("=" * 70)
    (out_dir / "extract_summary.json").write_text(
        json.dumps({
            "video": str(args.video), "csv": str(csv_path),
            "frames_read": summary.frames_read,
            "frames_processed": summary.frames_processed,
            "rows_written": summary.rows_written,
            "ocr_backend": summary.ocr_backend,
            "ocr_backend_available": summary.ocr_backend_available,
            "annotated_video": summary.annotated_video,
            # 下载到本地后拿 annotated_bytes 和本地文件大小比对，
            # 对不上就是传输截断（mp4 尾部的 moov 索引丢了 -> 打不开）
            "annotated_bytes": summary.annotated_bytes,
            "annotated_frames": summary.annotated_frames,
            "annotated_ok": summary.annotated_ok,
            "ocr_calls": summary.ocr_calls, "ocr_hits": summary.ocr_hits,
            "out_of_court": summary.out_of_court, "insane": summary.insane,
            "elapsed_sec": round(summary.elapsed_sec, 2),
            "numbers_seen": summary.numbers_seen,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"下一步 : python -m trajectory.cli players --csv {csv_path}")
    return 0


def cmd_players(args) -> int:
    from .paths import list_players, load_records

    df = load_records(args.csv)
    table = list_players(df)
    if table.empty:
        print("CSV 里没有任何可用样本。可能是 OCR 没跑起来（jersey_number 全空）"
              "且所有 slot 都是 0。看 extract_summary.json。")
        return 1
    print(table.to_string(index=False))
    print(f"\n用法: python -m trajectory.cli plot --csv {args.csv} --jersey <号码>")
    return 0


def cmd_plot(args) -> int:
    from .paths import build_track, list_players, load_records
    from .viz import draw_heatmap, draw_trajectory, render_video

    df = load_records(args.csv)
    out_dir = Path(args.out) if args.out else Path(args.csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: list[dict] = []
    if args.all:
        for _, r in list_players(df).iterrows():
            if r["frames"] < args.min_frames:
                continue
            targets.append({r["kind"]: r["value"]})
    elif args.jersey is not None:
        targets.append({"jersey": args.jersey})
    elif args.slot is not None:
        targets.append({"slot": args.slot})
    elif args.identity is not None:
        targets.append({"identity": args.identity})
    else:
        print("要么给 --jersey / --slot / --identity，要么给 --all")
        return 2

    made: list[str] = []
    for spec in targets:
        try:
            tr = build_track(df, smooth_window=args.smooth, **spec)
        except ValueError as exc:
            print(f"跳过 {spec}: {exc}")
            continue
        print("\n" + tr.summary())
        safe = tr.label.replace("#", "num").replace(":", "_")
        made.append(str(draw_trajectory(
            tr, out_dir / f"trajectory_{safe}.png", ppm=args.ppm, show_raw=args.show_raw)))
        if not args.no_heatmap:
            made.append(str(draw_heatmap(tr, out_dir / f"heatmap_{safe}.png", ppm=args.ppm)))
        if args.video:
            ext = ".avi" if args.fourcc.upper() == "MJPG" else ".mp4"
            made.append(str(render_video(
                tr, out_dir / f"trajectory_{safe}{ext}",
                ppm=args.ppm, fps=args.fps, fourcc=args.fourcc)))
        # 轨迹本身也导一份，方便接别的分析
        tr_csv = out_dir / f"track_{safe}.csv"
        with tr_csv.open("w", encoding="utf-8") as fh:
            fh.write("frame_idx,timestamp_sec,x_m,y_m,x_raw_m,y_raw_m,interpolated\n")
            for i in range(len(tr)):
                fh.write(f"{int(tr.frame_idx[i])},{tr.t[i]:.4f},"
                         f"{tr.xy[i,0]:.4f},{tr.xy[i,1]:.4f},"
                         f"{tr.xy_raw[i,0]:.4f},{tr.xy_raw[i,1]:.4f},"
                         f"{int(tr.interpolated[i])}\n")
        made.append(str(tr_csv))

    if not made:
        print("没有产出任何图。先跑 players 子命令看看有哪些球员。")
        return 1
    print("\n产出:")
    for m in made:
        print("  " + m)
    return 0


def cmd_checkvideo(args) -> int:
    """诊断一个 mp4 到底是完整的、还是被截断的。

    MP4 的 moov 原子（轨道信息、采样表、时长）由 FFmpeg 写在 mdat **之后**。
    传输被砍断时 moov 整个丢失，播放器只会说「格式不支持」，看起来像是
    「视频没生成」。这里直接解析 box 结构，把「声明多长」和「实际多长」
    摆在一起，一眼就能分清是编码问题还是传输问题。
    """
    import struct

    import cv2

    p = Path(args.video)
    if not p.is_file():
        print(f"文件不存在: {p}")
        return 1
    data = p.read_bytes()
    n = len(data)
    print("=" * 74)
    print(f"文件 : {p}")
    print(f"大小 : {n:,} 字节 = {n/1048576:.3f} MiB")
    if n and n % (1024 * 1024) == 0:
        print(f"       ⚠ 正好是 {n//1048576} MiB 的整数倍 —— 编码器不会凑出这种整数，"
              "强烈提示传输被截断")
    print("-" * 74)

    off, declared, has_moov, truncated = 0, 0, False, False
    while off + 8 <= n:
        size = struct.unpack(">I", data[off:off + 8][:4])[0]
        typ = data[off + 4:off + 8].decode("ascii", "replace")
        if typ == "moov":
            has_moov = True
        print(f"  box @{off:<12,} type={typ:<6} declared={size:,}")
        if size == 0:
            print("       (size=0 表示一直到文件末尾)")
            break
        if size == 1 and off + 16 <= n:                      # 64 位 largesize
            size = struct.unpack(">Q", data[off + 8:off + 16])[0]
            print(f"       64-bit largesize={size:,}")
        declared = off + size
        off += size
        if off > n:
            truncated = True
            print(f"  !! 下一个 box 起点 {off:,} 超出文件末尾 {n:,} -> 被截断")
            break

    print("-" * 74)
    print(f"moov 索引原子 : {'有' if has_moov else '没有  ← 没有它任何播放器都解不了'}")
    cap = cv2.VideoCapture(str(p))
    opened = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    ok_read = cap.read()[0] if opened else False
    cap.release()
    print(f"OpenCV 打开   : {opened}   帧数 {frames}   能读首帧 {ok_read}")
    print("=" * 74)

    if truncated or not has_moov:
        pct = 100.0 * n / declared if declared else 0.0
        print(f"结论 : ❌ 文件不完整。只拿到约 {pct:.1f}%（应为 {declared:,} 字节 = "
              f"{declared/1048576:.2f} MiB）")
        print("       这是**传输**问题，不是编码问题 —— 服务器上的原文件很可能是好的。")
        print("       重新下载，用能续传/校验的方式：")
        print("         rsync -avP 用户@主机:远端路径 ./          # -P 支持断点续传")
        print("         scp 用户@主机:远端路径 ./ && 比对字节数")
        print("       下载完对一下 md5：")
        print("         服务器: md5sum 远端路径")
        print("         本地  : certutil -hashfile 本地文件 MD5")
        return 1
    if not (opened and ok_read):
        print("结论 : ❌ 结构完整但解不开 —— 编码器收尾可能失败。")
        print("       重跑时换 --annotated-fourcc MJPG 且后缀用 .avi。")
        return 1
    print(f"结论 : ✅ 文件完整可播（{frames} 帧）")
    return 0


def cmd_check(args) -> int:
    """上云之前先跑这个：确认标定、仓库、依赖都就位。"""
    ok = True
    print("=" * 70)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"[{'ok' if in_venv else 'warn'}] python = {sys.executable}")
    if not in_venv:
        print("       当前没有运行在虚拟环境中；推荐使用 trajectory/trajectory.sh。")
    try:
        proj = CourtProjector(args.calibration)
        print("[ok]   标定")
        print(proj.describe())
    except Exception as exc:
        ok = False
        print(f"[FAIL] 标定: {exc}")

    try:
        from .bridge import Backend
        b = Backend(args.repo)
        print(f"[ok]   {b.describe()}")
        cfg = Path(args.config) if args.config else b.default_config_path
        print(f"       config = {cfg}  {'(存在)' if Path(cfg).is_file() else '(缺失!)'}")
        if not Path(cfg).is_file():
            ok = False
    except Exception as exc:
        ok = False
        print(f"[FAIL] basketball-v2: {exc}")

    # 这里一律 catch Exception，不只是 ImportError：环境坏掉时（典型的是
    # numpy 2.x 配上用 numpy 1.x 编译的二进制包，或者 psutil 版本不匹配）
    # import 抛的是 AttributeError 之类，只 catch ImportError 会让自检自己崩掉，
    # 那就完全失去了自检的意义。
    def probe(mod: str) -> tuple[bool, str]:
        try:
            m = __import__(mod)
            return True, str(getattr(m, "__version__", ""))
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    for mod in ("cv2", "numpy", "pandas"):
        good, info = probe(mod)
        if good:
            print(f"[ok]   {mod} {info}")
        else:
            ok = False
            print(f"[FAIL] {mod} 不可用 -> {info}")

    for mod, why in (("scipy", "只影响平滑质量，会退回滑动平均"),
                     ("ultralytics", "extract 跑不了，但 players / plot 不受影响"),
                     ("torch", "extract 跑不了，但 players / plot 不受影响")):
        good, info = probe(mod)
        print(f"[ok]   {mod} {info}" if good else f"[warn] {mod} 不可用（{why}）\n       {info}")

    npv, _ = probe("numpy")
    try:
        import numpy as _np
        if int(_np.__version__.split(".")[0]) >= 2:
            print("[note] numpy 是 2.x。用 numpy 1.x 编译的二进制包（pyarrow / "
                  "numexpr / bottleneck 等）会报 '_ARRAY_API not found'。\n"
                  "       服务器上建议 pin numpy<2，见 trajectory/requirements.txt。")
    except Exception:
        pass

    print("=" * 70)
    print("核心就位" if ok else "有缺失，见上面 FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trajectory",
        description="按球衣号抽取球员俯视轨迹（接 basketball-v2 的检测 + 本项目的米制标定）")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    common_cal = dict(default=None,
                      help="标定 json（默认 calibration/court_3x3.json）")

    e = sub.add_parser("extract", help="跑视频，产出逐帧球场坐标 CSV")
    e.add_argument("--video", required=True, help="输入视频")
    e.add_argument("--out", required=True, help="输出目录")
    e.add_argument("--repo", default=None,
                   help="basketball-v2 路径（也可用环境变量 BASKETBALL_V2）")
    e.add_argument("--config", default=None, help="basketball-v2 的 configs/default.yaml")
    e.add_argument("--calibration", **common_cal)
    e.add_argument("--start-frame", type=int, default=0)
    e.add_argument("--end-frame", type=int, default=None)
    e.add_argument("--duration-sec", type=float, default=None,
                   help="只处理指定秒数；例如 60 表示从 start-frame 起处理一分钟")
    e.add_argument("--stride", type=int, default=1, help="每隔几帧处理一次（默认每帧）")
    e.add_argument("--log-every", type=int, default=200)
    e.add_argument("--annotated-video", default=None,
                   help="原画面检测框视频路径（默认 <out>/annotated.mp4）")
    e.add_argument("--no-annotated-video", action="store_true",
                   help="不生成原画面检测框视频")
    e.add_argument("--annotated-fourcc", default="mp4v",
                   help="标注视频编码，默认 mp4v；服务器不可用时用 MJPG（后缀改 .avi）")
    e.add_argument("--annotated-scale", type=float, default=1.0,
                   help="标注视频缩放，0.5 约等于体积降到 1/4，便于传回本机。"
                        "只影响给人看的视频，检测/投影/CSV 仍用原始分辨率")
    e.set_defaults(func=cmd_extract)

    l = sub.add_parser("players", help="列出 CSV 里有哪些球员")
    l.add_argument("--csv", required=True)
    l.set_defaults(func=cmd_players)

    q = sub.add_parser("plot", help="画某个球员的轨迹 / 热区 / 动画")
    q.add_argument("--csv", required=True)
    q.add_argument("--out", default=None, help="输出目录（默认和 csv 同目录）")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--jersey", type=int, help="球衣号")
    g.add_argument("--slot", type=int, help="槽位号（没识别出球衣号时用）")
    g.add_argument("--identity", type=str, help="identity_key")
    g.add_argument("--all", action="store_true", help="所有球员各画一套")
    q.add_argument("--min-frames", type=int, default=60,
                   help="--all 时，少于这么多帧的球员跳过")
    q.add_argument("--smooth", type=int, default=15, help="平滑窗口（帧）")
    q.add_argument("--ppm", type=float, default=60.0, help="俯视图每米多少像素")
    q.add_argument("--show-raw", action="store_true", help="同时画未平滑的原始轨迹")
    q.add_argument("--no-heatmap", action="store_true")
    q.add_argument("--video", action="store_true", help="额外输出轨迹动画")
    q.add_argument("--fps", type=float, default=30.0)
    q.add_argument("--fourcc", default="mp4v", help="视频编码；服务器上编码器缺失就用 MJPG")
    q.set_defaults(func=cmd_plot)

    cv_ = sub.add_parser("checkvideo",
                         help="诊断一个 mp4 是完整的还是被截断的（打不开时先跑这个）")
    cv_.add_argument("--video", required=True, help="要检查的视频文件")
    cv_.set_defaults(func=cmd_checkvideo)

    c = sub.add_parser("check", help="环境自检（上云前先跑这个）")
    c.add_argument("--repo", default=None)
    c.add_argument("--config", default=None)
    c.add_argument("--calibration", **common_cal)
    c.set_defaults(func=cmd_check)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
