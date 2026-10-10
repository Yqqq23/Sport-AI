"""接入 basketball-v2 的检测 / 跟踪 / 球衣号识别。

为什么是「接入」而不是「复制」
------------------------------
basketball-v2 里已经有一整套能跑的东西：``Tracker``（检测+MOT）、
``SlotTracker``（按位置分配槽位）、``JerseyNumberStabilizer``（球衣号投票+锁定）、
``PlayerIdentityRegistry``（把槽位粘到球衣号上）。重写一遍只会引入分歧。

本模块只干一件事：把 ``basketball-v2/src`` 挂到 sys.path 上，然后把那边的
符号原样导出。任何算法都不在这里实现。

⚠ 那边的模块是**平铺 import** 的（``from calib import ...``，见
   basketball-v2/src/pipeline.py:27），所以必须把 ``src`` 目录本身
   加进 sys.path，加上仓库根目录是不够的。

仓库位置的查找顺序
------------------
    1. 显式传进来的 repo 参数（CLI ``--repo``）
    2. 环境变量 ``BASKETBALL_V2``      ← 上云跑就用这个
    3. 从本文件位置**逐级向上**走，每一级都看它自己和它下面的
       ``basketball-v2/``（覆盖「服务器上整合成一个目录」和
       「本机上两个仓库并排放」两种布局）
    4. ``D:/basketball-v2``（本机开发用的兜底，放最后）

⚠ 第 3 步是 2026-09 加的。原来只看固定的三个位置（项目根、项目根的
   父目录、项目根下面），仓库在 Windows 的 D: 盘、代码在服务器 home 下
   的时候一个都命中不了，报错还只说「找不到」不说该设什么。

找不到就抛异常并把找过的位置全列出来 —— 不做静默降级，
因为没有检测器的话整条链路根本没有输出。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

#: 仓库里必须存在的模块，用来确认路径没找错
_REQUIRED = ("track.py", "jersey.py", "identity.py", "config.py")

_CACHED_SRC: Path | None = None


#: 向上走几级为止。够覆盖 <home>/work/proj/trajectory 这种深度，
#: 又不至于一路走到 / 然后把整个文件系统都 stat 一遍。
_MAX_ASCEND = 6


def _candidates(explicit: str | Path | None) -> list[Path]:
    here = Path(__file__).resolve()
    project_root = here.parent.parent          # <project>/trajectory/.. == <project>
    out: list[Path] = []
    if explicit:
        out.append(Path(explicit))
    env = os.environ.get("BASKETBALL_V2")
    if env:
        out.append(Path(env))

    # 从项目根逐级向上。每一级既可能**就是**仓库（服务器整合布局），
    # 也可能**装着**仓库（两个仓库并排）。
    node = project_root
    for _ in range(_MAX_ASCEND):
        out.append(node)
        out.append(node / "basketball-v2")
        if node.parent == node:                # 到根了
            break
        node = node.parent

    out.append(Path("D:/basketball-v2"))

    # 去重但保序 —— 上面几个来源很容易产生重复项，重复只会让
    # 报错里的「找过这些位置」列表更长更难读。
    seen: set[str] = set()
    unique: list[Path] = []
    for cand in out:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def locate_repo(explicit: str | Path | None = None) -> Path:
    """返回 basketball-v2 的 ``src`` 目录。"""
    tried: list[str] = []
    for cand in _candidates(explicit):
        cand = cand.expanduser()
        # 既允许指向仓库根，也允许直接指向 src
        for src in (cand / "src", cand):
            tried.append(str(src))
            if all((src / m).is_file() for m in _REQUIRED):
                return src.resolve()
    raise FileNotFoundError(
        "找不到 basketball-v2 的 src 目录。\n"
        "\n"
        "  怎么修（二选一）：\n"
        "    1) 命令行传：  --repo /path/to/basketball-v2\n"
        "    2) 设环境变量：export BASKETBALL_V2=/path/to/basketball-v2\n"
        "       （Windows PowerShell: $env:BASKETBALL_V2='D:\\basketball-v2'）\n"
        "\n"
        f"  指向仓库根或直接指向它的 src/ 都行。判定标准是该目录下同时有：\n"
        f"    {', '.join(_REQUIRED)}\n"
        "\n"
        "  已经找过这些位置，都不满足：\n"
        + "\n".join(f"    {t}" for t in tried)
    )


def ensure_on_path(explicit: str | Path | None = None) -> Path:
    """把 basketball-v2/src 挂到 sys.path 最前面（幂等）。"""
    global _CACHED_SRC
    if _CACHED_SRC is not None and explicit is None:
        return _CACHED_SRC
    src = locate_repo(explicit)
    s = str(src)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    _CACHED_SRC = src
    return src


class Backend:
    """basketball-v2 里那些符号的一个薄封装。

    延迟导入：只有真的要跑检测时才去 import ultralytics / torch 等重依赖，
    这样「只画已有轨迹」这类命令在没装模型环境的机器上也能用。
    """

    def __init__(self, repo: str | Path | None = None):
        self.src = ensure_on_path(repo)
        self._mods: dict[str, ModuleType] = {}

    def _mod(self, name: str) -> ModuleType:
        if name not in self._mods:
            self._mods[name] = __import__(name)
        return self._mods[name]

    # -- 配置 --------------------------------------------------------------
    def load_config(self, path: str | Path):
        return self._mod("config").load_config(path)

    @property
    def default_config_path(self) -> Path:
        return self.src.parent / "configs" / "default.yaml"

    # -- 检测 / 跟踪 -------------------------------------------------------
    def make_tracker(self, weights, tracker_config):
        return self._mod("track").Tracker(weights=weights, config=tracker_config)

    # -- 身份 --------------------------------------------------------------
    def make_slot_tracker(self, max_players: int):
        return self._mod("identity").SlotTracker(max_players=max_players)

    def make_identity_registry(self, max_players: int):
        return self._mod("identity").PlayerIdentityRegistry(max_players=max_players)

    # -- 球衣号 ------------------------------------------------------------
    def make_stabilizer(self, stabilizer_config):
        return self._mod("jersey").JerseyNumberStabilizer(stabilizer_config)

    def make_ocr(self, jersey_config):
        j = self._mod("jersey")
        return j.make_ocr_backend(
            jersey_config.backend,
            use_gpu=jersey_config.use_gpu,
            roboflow=jersey_config.roboflow,
            local=jersey_config.local,
            parseq=jersey_config.parseq,
            # Added alongside src/jersey.py's PerDigitJerseyClassifier. This
            # bridge calls make_ocr_backend() directly instead of going through
            # src/pipeline.py, so it needed this kwarg added separately - that
            # mismatch (this file lives outside basketball-v2's repo) was the
            # actual cause of "perdigit unavailable (None)" even after every
            # basketball-v2 file was correctly synced. getattr() rather than a
            # plain attribute access so an OLDER config.py (no perdigit field
            # yet) degrades to None instead of crashing here.
            perdigit=getattr(jersey_config, "perdigit", None),
        )

    @property
    def jersey_helpers(self):
        """``(crop_jersey, gate_reason, GATE_OK, torso_region_from_player_box)``

        固定 4 元组，已有调用方（trajectory/extract.py:200）按这个 arity 解包。
        需要 ``padded_box`` 之类的新符号就用 ``jersey`` 属性拿整个模块，
        不要往这个元组后面追加。
        """
        j = self._mod("jersey")
        return (j.crop_jersey, j.gate_reason, j.GATE_OK, j.torso_region_from_player_box)

    @property
    def jersey(self):
        """basketball-v2 的 ``jersey`` 模块本体。

        比 ``jersey_helpers`` 那个定长元组灵活：加一个符号不用改所有解包点。
        """
        return self._mod("jersey")

    @property
    def detect(self):
        """``detect`` 模块，主要是为了拿 ``CLASS_JERSEY`` 这些类别常数。"""
        return self._mod("detect")

    @property
    def pipeline(self):
        """``pipeline`` 模块。

        要它只为一件事：``_nearest_player`` 和 ``_TorsoBox``。球衣框归属到
        哪个球员、以及躯干回退框怎么伪装成检测框，推理时都在这两个东西里，
        挖数据时**必须**用同一份，否则挖出来的 crop 归属规则和线上不一样。
        名字带下划线是 basketball-v2 的内部约定，这里是有意伸手进去拿的。

        比 ``jersey`` / ``track`` 重：会连带 import calib / render / shot。
        只在真的要挖数据时才碰它。
        """
        return self._mod("pipeline")

    @property
    def annotated_rendering(self):
        """Classes and renderer used by basketball-v2's annotated original view."""
        r = self._mod("render")
        return (
            r.PlayerRender,
            r.BallRender,
            r.RimRender,
            r.RefereeRender,
            r.FrameRenderData,
            r.render_annotated_frame,
        )

    def describe(self) -> str:
        return f"basketball-v2 src = {self.src}"
