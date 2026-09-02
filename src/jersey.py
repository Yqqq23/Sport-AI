"""Jersey number recognition and cross-frame stabilization.

Salvaged from:
  SRC_A/fixed7_jersey_tracking_ocr7/run_basketball_pose_tracking_fixed7_ocr7_stable.py
    - clean_jersey_number                  (lines 89-97)
    - trim_deque                           (lines 183-185)
    - majority_number_from_recent_history  (lines 188-200)
    - update_slot_number_state             (lines 1583-1625)
    - NumberRecognizer tesseract branch    (lines 798-816)

The voting / locking numeric behaviour is preserved verbatim. Every literal
constant below (0.55 confidence gate, the 0.85 vote-weight split, the
``max(2, min(window, max(3, lock_min_votes)))`` quorum, and the
``min(1.0, 0.55 + 0.1 * count + 0.2 * conf)`` lock-confidence formula) comes
from the source and must not be retuned without an experiment.

These behaviours are NEW (they did not exist in any salvaged variant and were
explicitly requested):
  1. leading-zero normalization, so "01" and "1" collapse to one number
  2. an OCR input gate, so OCR only runs on boxes large/sharp enough to read
  3. padded + upscaled crops, so the recognizer never sees a tight box
  4. selectable OCR backends (paddleocr / easyocr / pytesseract)

On (3): the jersey DETECTOR is strong at finding the number (mAP50 ~0.97) but
weak at pixel-exact boxes (mAP50-95 ~0.64), which on a ~30x40 px box means a
few pixels of edge jitter. A tight crop therefore clips digit strokes. Feeding
the recognizer a deliberately padded crop absorbs that jitter, so the weak
high-IoU localization stops mattering. Do not "fix" this by tightening the crop.

On (4): tesseract is the WEAKEST option here by a wide margin - it was built for
scanned documents, not for 30 px sports digits under motion blur. It is kept as
a last-resort fallback only. Prefer paddleocr or easyocr, both of which run in
recognition-only mode on the pre-cropped box and return a real confidence score
(tesseract cannot, which is why it returns a constant 0.7 and makes the
confidence-weighted voting below meaningless).

``cv2`` is imported lazily inside the OCR backends and the sharpness gate so
that the pure-Python voting logic in this module stays importable and unit
testable in an environment without OpenCV.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


# --------------------------------------------------------------------------
# Number cleaning / normalization
# --------------------------------------------------------------------------


def clean_jersey_number(text: Optional[str]) -> Optional[str]:
    """Reduce raw OCR text to a canonical jersey number, or None.

    Salvaged verbatim from the ``_stable`` variant (lines 89-97): keep digits
    only, reject anything longer than two digits, reject "00".

    NEW (not in any source variant): strip leading zeros so that "01" and "1"
    are treated as the same player. "0" itself stays a legal number. The source
    only ever did ``re.sub(r"\\D", "", ...)`` plus a ``[:2]`` truncation, so
    "01" and "1" voted as two separate numbers and could split a track's votes.
    """
    if not text:
        return None
    digits = re.sub(r"\D", "", str(text))
    if not digits or len(digits) > 2:
        return None
    if digits == "00":
        return None
    # --- NEW: leading-zero normalization ("01" -> "1", "0" stays "0") ---
    normalized = digits.lstrip("0")
    return normalized if normalized else "0"


# --------------------------------------------------------------------------
# OCR backend (pluggable)
# --------------------------------------------------------------------------


class JerseyOCRBackend(Protocol):
    """Any jersey OCR implementation. Returns (text, confidence)."""

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        ...


@dataclass
class RoboflowConfig:
    """Settings for the Roboflow-hosted backend.

    NOTE THE ABSENCE OF AN API KEY FIELD. The key is read from the environment
    variable named by ``api_key_env`` and deliberately has nowhere to live in
    the YAML, because configs/default.yaml is committed to the repository and a
    key placed there is a key that gets pushed.
    """

    #: "project-slug/version", e.g. "digits-x7d7z/10".
    #:
    #: MUST be a DEPLOYED model. A Roboflow project of type "text-image-pairs"
    #: is an OCR *dataset*, not a served model, and every call to it returns
    #: HTTP 500 "Service misconfiguration" - verified against
    #: basketball-jersey-numbers-ocr/7, which is exactly that type.
    #:
    #: The workspace is NOT part of the id. "workspace/project/version" is
    #: rejected by the SDK and returns 405 from the REST endpoint.
    model_id: str = "digits-x7d7z/10"
    #: Hosted endpoint by default. Point this at a local inference server
    #: ("http://localhost:9001") for any run longer than a smoke test - the
    #: hosted one measures 2.6-6.8 s per crop.
    api_url: str = "https://serverless.roboflow.com"
    #: Environment variable holding the API key.
    api_key_env: str = "ROBOFLOW_API_KEY"
    timeout_sec: float = 30.0
    #: Per-digit detections below this confidence are dropped before assembly.
    min_digit_confidence: float = 0.30


class PytesseractJerseyOCR:
    """Tesseract digit OCR.

    Preprocessing chain and the fixed 0.7 confidence are salvaged verbatim from
    the ``_stable`` variant, lines 798-816. Tesseract exposes no usable score
    here, so the source returned a constant 0.7; that is kept because the
    downstream vote weighting is calibrated against it (see
    ``JerseyNumberStabilizer.update``: < 0.85 counts as weight 1).
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pytesseract  # noqa: F401

                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                self._available = False
        return bool(self._available)

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0:
            return None, 0.0
        if not self.is_available():
            return None, 0.0
        try:
            import cv2
            import pytesseract

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                bw, config="--psm 7 -c tessedit_char_whitelist=0123456789"
            )
            digits = re.findall(r"\d+", text)
            if not digits:
                return None, 0.0
            candidate = max(digits, key=len)
            candidate = clean_jersey_number(candidate)
            if not candidate:
                return None, 0.0
            return candidate, 0.7
        except Exception:
            return None, 0.0


class PaddleJerseyOCR:
    """PaddleOCR in recognition-only mode. The recommended backend.

    Recognition-only (no text detection) is the right mode here because the
    jersey box is ALREADY located by the YOLO jersey class - running Paddle's
    detector again on a 30 px crop would only find nothing. This is also several
    times faster than the full det+rec pipeline.

    Unlike tesseract this returns a REAL per-prediction confidence, which is
    what makes the confidence weighting in ``JerseyNumberStabilizer.update``
    (>= 0.85 counts double) mean anything.

    PaddleOCR's call signature changed between 2.x and 3.x; both shapes are
    attempted rather than pinning a version, because the GPU server and the
    editing machine will not have the same one.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = True) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._reader: Any = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._load()
        return bool(self._available)

    def _load(self) -> bool:
        try:
            from paddleocr import PaddleOCR
        except Exception:
            return False
        # 3.x renamed/removed several constructor kwargs. Try richest first and
        # fall back, so a version bump degrades to defaults instead of crashing.
        for kwargs in (
            {"lang": self.lang, "use_angle_cls": False, "show_log": False, "use_gpu": self.use_gpu},
            {"lang": self.lang, "use_angle_cls": False, "show_log": False},
            {"lang": self.lang, "use_textline_orientation": False},
            {"lang": self.lang},
        ):
            try:
                self._reader = PaddleOCR(**kwargs)
                return True
            except Exception:
                continue
        return False

    def _run(self, crop: "np.ndarray") -> list[tuple[str, float]]:
        """Normalize the several shapes PaddleOCR returns into (text, score)."""
        reader = self._reader
        raw = None
        for call in (
            lambda: reader.ocr(crop, det=False, cls=False),
            lambda: reader.ocr(crop, det=False),
            lambda: reader.predict(crop),
            lambda: reader.ocr(crop),
        ):
            try:
                raw = call()
                break
            except Exception:
                continue
        if not raw:
            return []

        out: list[tuple[str, float]] = []

        def walk(node: Any) -> None:
            # rec-only 2.x: [[("7", 0.98)]]  |  det+rec 2.x: [[[box, ("7", 0.98)]]]
            if isinstance(node, (tuple, list)):
                if (
                    len(node) == 2
                    and isinstance(node[0], str)
                    and isinstance(node[1], (int, float))
                ):
                    out.append((node[0], float(node[1])))
                    return
                for child in node:
                    walk(child)
            elif isinstance(node, dict):
                # 3.x predict(): {"rec_texts": [...], "rec_scores": [...]}
                texts = node.get("rec_texts") or node.get("rec_text")
                scores = node.get("rec_scores") or node.get("rec_score")
                if texts is not None:
                    texts = texts if isinstance(texts, (list, tuple)) else [texts]
                    scores = scores if isinstance(scores, (list, tuple)) else [scores or 0.0]
                    for i, text in enumerate(texts):
                        score = scores[i] if i < len(scores) else 0.0
                        out.append((str(text), float(score)))

        walk(raw)
        return out

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0 or not self.is_available():
            return None, 0.0
        try:
            candidates = self._run(crop)
        except Exception:
            return None, 0.0
        return _best_digit_candidate(candidates)


class EasyJerseyOCR:
    """EasyOCR in recognition-only mode, restricted to digits.

    ``allowlist="0123456789"`` is the reason this backend is worth having: it
    removes every non-digit from the decoder's search space, so the classic
    jersey failure modes ("B" for 8, "S" for 5, "O" for 0) cannot be emitted at
    all. Tesseract's char whitelist is a post-filter by comparison and does not
    constrain the decode the same way.
    """

    def __init__(self, gpu: bool = True) -> None:
        self.gpu = gpu
        self._reader: Any = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import easyocr

                self._reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
                self._available = True
            except Exception:
                self._available = False
        return bool(self._available)

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0 or not self.is_available():
            return None, 0.0
        try:
            # recognize() skips detection; we already know where the number is.
            raw = self._reader.recognize(
                crop, allowlist="0123456789", detail=1, paragraph=False
            )
        except Exception:
            return None, 0.0

        candidates: list[tuple[str, float]] = []
        for item in raw or []:
            # (box, text, conf) from recognize(), or (text, conf)
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    candidates.append((str(item[1]), float(item[2])))
                elif len(item) == 2 and isinstance(item[0], str):
                    candidates.append((str(item[0]), float(item[1])))
        return _best_digit_candidate(candidates)


#: Letter -> digit fixes applied to scene-text OCR output before parsing.
#:
#: General scene-text models decode over a full alphanumeric charset, so they
#: can and do emit letters for jersey digits. MEASURED with PARSeq on jersey
#: crops: "0" comes back as the letter "O" often enough to be the single largest
#: error source (2 of 24 in the first probe, and BOTH failures were this).
#: Since a jersey number is digits-only by definition, mapping the homoglyph is
#: strictly better than discarding the reading.
#:
#: Only NEAR-IDENTICAL glyphs are listed. Every entry is also a way to turn a
#: correct rejection into a wrong number, so the speculative tail was removed
#: after testing: "b"->6 alone made the junk string "abc" decode as "6".
#: "A"->4, "T"->7, "g"/"q"->9 were dropped for the same reason - they are
#: font-dependent resemblances, not shape collisions.
_HOMOGLYPHS: dict[str, str] = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6",
    "B": "8",
}


def digits_from_scene_text(text: Optional[str]) -> Optional[str]:
    """Reduce raw scene-text output to a jersey number, repairing homoglyphs.

    Applied before ``clean_jersey_number`` so that "O" becomes "0" rather than
    being stripped as a non-digit and losing the reading entirely.
    """
    if not text:
        return None
    repaired = "".join(_HOMOGLYPHS.get(character, character) for character in str(text))
    return clean_jersey_number(repaired)


@dataclass
class PARSeqConfig:
    """Settings for the PARSeq scene-text recognizer."""

    #: torch.hub source. Override with a local clone path when GitHub is not
    #: reachable from the GPU box (set ``source`` to "local" as well).
    repo: str = "baudm/parseq"
    entrypoint: str = "parseq"
    #: "github" or "local". "local" treats ``repo`` as a directory.
    source: str = "github"
    device: str = "cuda"
    #: PARSeq's trained input geometry. Do not change without retraining.
    image_height: int = 32
    image_width: int = 128
    #: Minimum per-character confidence for the reading to count.
    min_confidence: float = 0.50


class PARSeqJerseyOCR:
    """PARSeq scene-text recognition. THE ZERO-TRAINING OPTION.

    Chosen because Koshkina & Elder's jersey-number pipeline reports PARSeq
    working well on jersey numbers WITHOUT any fine-tuning, so pretrained
    general weights are usable as-is today. Everything else in this module
    either needs training (``local``), a network round trip (``roboflow``), or
    is a document-OCR engine being asked to do something it was not built for
    (``pytesseract``).

    MEASURED on synthetic jersey crops at 30-70 px box height with blur: 22/24
    correct out of the box, ~90-190 ms per crop on CPU (expect roughly 5-15 ms
    on a GPU). Both failures were the letter "O" for the digit "0", which
    ``digits_from_scene_text`` now repairs.

    DEPENDENCIES. PARSeq's repo imports ``pytorch_lightning`` (its model
    subclasses LightningModule) and ``nltk`` (one validation metric). Neither is
    needed to run inference. They are imported normally when present; when they
    are absent or broken, ``_install_stubs`` supplies the few symbols PARSeq
    touches at construction time. That keeps a working install working and
    avoids forcing a heavy, fragile dependency chain (pytorch_lightning ->
    torchmetrics -> transformers -> sklearn) onto a machine that only wants to
    read two digits.
    """

    def __init__(self, config: Optional[PARSeqConfig] = None) -> None:
        self.config = config or PARSeqConfig()
        self._model: Any = None
        self._transform: Any = None
        self._device: Any = None
        self._available: Optional[bool] = None
        self.calls = 0

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._load()
        return bool(self._available)

    @staticmethod
    def _install_stubs() -> None:
        """Provide the minimum of pytorch_lightning / nltk that PARSeq needs.

        Only installed for names that are not already importable, so a real
        installation always wins. PARSeq calls exactly three Lightning things
        while being constructed: the LightningModule base class,
        ``save_hyperparameters`` and ``log``.
        """
        import importlib.machinery
        import importlib.util
        import sys
        import types

        def importable(name: str) -> bool:
            """Whether the module can actually be IMPORTED, not merely found.

            find_spec() is not enough. A package can be installed and locatable
            yet raise on import - pytorch_lightning pulls torchmetrics ->
            transformers -> sklearn, and a numpy ABI mismatch anywhere in that
            chain makes it findable but unusable. Checking with find_spec left
            the stub uninstalled and PARSeq then died on the real import.
            """
            if name in sys.modules:
                return True
            try:
                importlib.import_module(name)
                return True
            except Exception:
                return False

        if not importable("pytorch_lightning"):
            import torch.nn as nn

            class _HParams(dict):
                __getattr__ = dict.get
                __setattr__ = dict.__setitem__

            class _LightningModule(nn.Module):
                def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
                    import inspect

                    frame = inspect.currentframe().f_back
                    captured = {
                        key: value
                        for key, value in frame.f_locals.items()
                        if key not in ("self", "__class__") and not key.startswith("_")
                    }
                    object.__setattr__(self, "_hparams_store", _HParams(captured))

                @property
                def hparams(self) -> Any:
                    if not hasattr(self, "_hparams_store"):
                        object.__setattr__(self, "_hparams_store", _HParams())
                    return self._hparams_store

                def log(self, *args: Any, **kwargs: Any) -> None:
                    pass

            lightning = types.ModuleType("pytorch_lightning")
            lightning.LightningModule = _LightningModule
            lightning.Trainer = object
            lightning.LightningDataModule = object

            class _Callback:
                pass

            lightning.Callback = _Callback
            utilities = types.ModuleType("pytorch_lightning.utilities")
            types_module = types.ModuleType("pytorch_lightning.utilities.types")
            types_module.STEP_OUTPUT = object
            types_module.EPOCH_OUTPUT = object
            utilities.types = types_module
            lightning.utilities = utilities
            callbacks = types.ModuleType("pytorch_lightning.callbacks")
            callbacks.Callback = _Callback
            lightning.callbacks = callbacks

            # torch.hub validates deps with find_spec, which raises on a module
            # whose __spec__ is None. Give every stub a real ModuleSpec.
            for module, name in (
                (lightning, "pytorch_lightning"),
                (utilities, "pytorch_lightning.utilities"),
                (types_module, "pytorch_lightning.utilities.types"),
                (callbacks, "pytorch_lightning.callbacks"),
            ):
                module.__spec__ = importlib.machinery.ModuleSpec(name, None)
                sys.modules[name] = module
            LOGGER.info("pytorch_lightning not importable; using an inference-only stub.")

        if not importable("nltk"):
            nltk = types.ModuleType("nltk")

            def _edit_distance(*args: Any, **kwargs: Any) -> int:
                raise NotImplementedError("nltk.edit_distance is a training-only metric")

            nltk.edit_distance = _edit_distance
            nltk.__spec__ = importlib.machinery.ModuleSpec("nltk", None)
            sys.modules["nltk"] = nltk
            LOGGER.info("nltk not importable; using an inference-only stub.")

    def _load(self) -> bool:
        try:
            import torch
            import torchvision.transforms as transforms
        except Exception as exc:
            LOGGER.error("Jersey backend 'parseq' needs torch + torchvision: %s", exc)
            return False

        try:
            self._install_stubs()
        except Exception as exc:  # never let the shim itself break the run
            LOGGER.warning("Could not install PARSeq compatibility stubs: %s", exc)

        try:
            # trust_repo=True is REQUIRED. Without it torch.hub prompts on stdin
            # for confirmation, and a pipeline run would hang forever.
            model = torch.hub.load(
                self.config.repo,
                self.config.entrypoint,
                source=self.config.source,
                pretrained=True,
                trust_repo=True,
            )
        except Exception as exc:
            LOGGER.error(
                "Could not load PARSeq from %s (%s): %s\n"
                "If GitHub is unreachable from this machine, clone it once and set\n"
                "  jersey.parseq.repo: /path/to/parseq\n"
                "  jersey.parseq.source: local",
                self.config.repo,
                self.config.source,
                str(exc)[:300],
            )
            return False

        device = torch.device(
            self.config.device
            if (self.config.device != "cuda" or torch.cuda.is_available())
            else "cpu"
        )
        if self.config.device == "cuda" and device.type == "cpu":
            LOGGER.warning("CUDA requested for PARSeq but unavailable; using CPU (slow).")

        self._model = model.eval().to(device)
        self._device = device
        # Preprocessing must match PARSeq's training exactly: bicubic resize to
        # 32x128 then normalize to [-1, 1]. Taken from the official demo.
        self._transform = transforms.Compose([
            transforms.Resize(
                (self.config.image_height, self.config.image_width),
                transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        LOGGER.info("PARSeq jersey backend ready on %s", device)
        return True

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0 or not self.is_available():
            return None, 0.0
        import torch
        from PIL import Image

        try:
            image = Image.fromarray(crop[:, :, ::-1]).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                probabilities = self._model(tensor).softmax(-1)
            labels, confidences = self._model.tokenizer.decode(probabilities)
            self.calls += 1
            raw = labels[0]
            # Per-character confidences; the number is only as good as its
            # weakest character.
            confidence = float(confidences[0].min()) if len(confidences[0]) else 0.0
        except Exception as exc:
            LOGGER.warning("PARSeq recognition failed: %s", exc)
            return None, 0.0

        if confidence < self.config.min_confidence:
            return None, 0.0
        return digits_from_scene_text(raw), confidence


@dataclass
class LocalClassifierConfig:
    """Settings for the locally-trained jersey classifier."""

    #: Checkpoint from train/train_jersey.py. It is self-describing: the class
    #: list, input size and normalization constants all travel inside it, so
    #: this config never has to restate how the model was trained.
    weights: Optional[Path] = None
    device: str = "cuda"
    #: Predictions below this are discarded before they can vote. Distinct from
    #: the stabilizer's min_observation_confidence, which gates a value that has
    #: already been accepted as a reading.
    min_confidence: float = 0.50
    #: Also require a margin between the top two classes. A softmax can be 0.6
    #: on "8" and 0.39 on "6" - technically above threshold, but a coin flip
    #: between two digits that look identical at 20 px. Set 0 to disable.
    min_margin: float = 0.15


class LocalJerseyClassifier:
    """A locally-loaded closed-set jersey number classifier. THE FAST PATH.

    ~1-2 ms per crop on GPU against 1.3-6.8 SECONDS for a hosted endpoint, with
    no network, no API key, no quota, and no frames leaving the machine. This is
    what makes recognition affordable on every OCR frame of a full game.

    Trained by train/train_jersey.py; see that file for why this is closed-set
    classification rather than sequence OCR, and why the augmentation matters
    more than the architecture.

    The checkpoint carries its own class list. A number outside that list is not
    merely unlikely, it is UNREPRESENTABLE - the softmax has no output for it.
    ``unrepresentable_numbers`` exists so the pipeline can say so out loud
    instead of quietly voting for the nearest trained number forever.
    """

    def __init__(self, config: Optional[LocalClassifierConfig] = None) -> None:
        self.config = config or LocalClassifierConfig()
        self._model: Any = None
        self._classes: list[str] = []
        self._device: Any = None
        self._input_size = 96
        self._mean: Any = None
        self._std: Any = None
        self._available: Optional[bool] = None
        self.calls = 0
        self.rejected_low_confidence = 0

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._load()
        return bool(self._available)

    def _load(self) -> bool:
        from pathlib import Path as _Path

        weights = self.config.weights
        if weights is None:
            LOGGER.error(
                "Jersey backend 'local' needs jersey.local.weights pointing at a "
                "checkpoint from train/train_jersey.py."
            )
            return False
        weights = _Path(weights)
        if not weights.exists():
            LOGGER.error(
                "Jersey classifier weights not found: %s\n"
                "Train them first:\n"
                "  python scripts/download_jersey_ocr_dataset.py --out data/jersey_ocr\n"
                "  python train/train_jersey.py --data data/jersey_ocr",
                weights,
            )
            return False

        try:
            import torch
            from torchvision import models
        except Exception as exc:
            LOGGER.error("Jersey backend 'local' needs torch + torchvision: %s", exc)
            return False

        try:
            checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
            self._classes = list(checkpoint["classes"])
            self._input_size = int(checkpoint.get("input_size", 96))
            mean = checkpoint.get("mean", (0.485, 0.456, 0.406))
            std = checkpoint.get("std", (0.229, 0.224, 0.225))

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self._classes))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            device = torch.device(
                self.config.device
                if (self.config.device != "cuda" or torch.cuda.is_available())
                else "cpu"
            )
            if self.config.device == "cuda" and device.type == "cpu":
                LOGGER.warning("CUDA requested for the jersey classifier but unavailable; using CPU.")
            self._model = model.to(device)
            self._device = device
            self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        except Exception as exc:
            LOGGER.error("Could not load the jersey classifier from %s: %s", weights, exc)
            return False

        LOGGER.info(
            "Local jersey classifier ready: %s on %s, %s classes (val_acc=%s)",
            weights.name,
            self._device,
            len(self._classes),
            checkpoint.get("val_accuracy"),
        )
        return True

    def unrepresentable_numbers(self, numbers: Sequence[str]) -> list[str]:
        """Which of ``numbers`` this checkpoint physically cannot predict."""
        known = set(self._classes)
        return sorted({n for n in numbers if n not in known}, key=lambda s: int(s))

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0 or not self.is_available():
            return None, 0.0
        import cv2
        import torch

        try:
            resized = cv2.resize(
                crop, (self._input_size, self._input_size), interpolation=cv2.INTER_CUBIC
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).to(self._device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            tensor = (tensor - self._mean) / self._std

            with torch.no_grad():
                probabilities = torch.softmax(self._model(tensor), dim=1)[0]
            self.calls += 1

            top = torch.topk(probabilities, k=min(2, probabilities.numel()))
            confidence = float(top.values[0])
            margin = confidence - (float(top.values[1]) if top.values.numel() > 1 else 0.0)
            label = self._classes[int(top.indices[0])]
        except Exception as exc:
            LOGGER.warning("Local jersey classification failed: %s", exc)
            return None, 0.0

        if confidence < self.config.min_confidence or margin < self.config.min_margin:
            self.rejected_low_confidence += 1
            return None, 0.0
        return clean_jersey_number(label), confidence


class RoboflowJerseyOCR:
    """A Roboflow-hosted model, via ``inference_sdk.InferenceHTTPClient``.

    Handles BOTH shapes a Roboflow jersey model comes in, because which one you
    get depends entirely on how the project owner set it up:

    ``object-detection`` (the common case)
        Each DIGIT is a separate detection with its own box. "23" comes back as
        two predictions, class "2" at x=44 and class "3" at x=111, in arbitrary
        order. They must be sorted by x and concatenated - which is why this
        backend cannot be a thin wrapper around ``client.infer``.

    ``classification``
        The whole number is one class. ``predictions`` is a dict keyed by class
        name instead of a list, and the answer is the top-confidence key.

    LATENCY IS THE DESIGN CONSTRAINT. Measured against
    ``https://serverless.roboflow.com``: 2.6-6.8 SECONDS per call. At
    ocr_interval=3 with six players a 26k-frame game issues ~50k calls, which is
    days of wall clock, ignoring rate limits. Two things follow, and neither is
    optional:

    * the pipeline must stop calling once a track's number is locked
      (see ``JerseyNumberStabilizer.is_locked`` and the caller in pipeline.py);
    * for any full-length run, point ``api_url`` at a LOCAL inference server
      (``docker run roboflow/roboflow-inference-server-gpu``, then
      ``http://localhost:9001``). Same model, same API, ~20-50 ms per call
      instead of seconds, no per-call quota, and no frames leaving the machine.

    The API key is read from an ENVIRONMENT VARIABLE, never from the config
    file, so it cannot be committed to git along with the rest of the repo.
    """

    def __init__(
        self,
        model_id: str,
        api_url: str = "https://serverless.roboflow.com",
        api_key_env: str = "ROBOFLOW_API_KEY",
        timeout_sec: float = 30.0,
        min_digit_confidence: float = 0.30,
    ) -> None:
        self.model_id = model_id
        self.api_url = api_url
        self.api_key_env = api_key_env
        self.timeout_sec = float(timeout_sec)
        self.min_digit_confidence = float(min_digit_confidence)
        self._client: Any = None
        self._available: Optional[bool] = None
        #: Consecutive transport failures. A dead endpoint should not cost one
        #: multi-second timeout per crop for the rest of the video.
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self.calls = 0

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._load()
        return bool(self._available)

    def _load(self) -> bool:
        import os

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            LOGGER.error(
                "Jersey backend 'roboflow' needs an API key in $%s. "
                "Export it:  export %s=your_key_here",
                self.api_key_env,
                self.api_key_env,
            )
            return False
        try:
            from inference_sdk import InferenceHTTPClient
        except Exception:
            LOGGER.error(
                "Jersey backend 'roboflow' needs the inference SDK. "
                "Fix:  pip install inference-sdk"
            )
            return False
        try:
            self._client = InferenceHTTPClient(api_url=self.api_url, api_key=api_key)
        except Exception as exc:
            LOGGER.error("Could not build the Roboflow client: %s", exc)
            return False
        LOGGER.info(
            "Roboflow jersey backend ready: model=%s via %s", self.model_id, self.api_url
        )
        if "serverless.roboflow.com" in self.api_url:
            LOGGER.warning(
                "Using the HOSTED Roboflow endpoint (measured 2.6-6.8 s per crop). "
                "This is fine for a short --max-frames check but not for a full "
                "game. For a real run start a local inference server and set "
                "jersey.roboflow.api_url to http://localhost:9001"
            )
        return True

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0 or not self.is_available():
            return None, 0.0
        if self._consecutive_errors >= self._max_consecutive_errors:
            return None, 0.0
        try:
            result = self._client.infer(crop, model_id=self.model_id)
            self._consecutive_errors = 0
            self.calls += 1
        except Exception as exc:
            self._consecutive_errors += 1
            LOGGER.warning(
                "Roboflow inference failed (%s/%s): %s",
                self._consecutive_errors,
                self._max_consecutive_errors,
                str(exc)[:200],
            )
            if self._consecutive_errors >= self._max_consecutive_errors:
                LOGGER.error(
                    "Giving up on the Roboflow backend after %s consecutive "
                    "failures; the rest of this run will produce no numbers. "
                    "Check the model id (%s) is a DEPLOYED model - a Roboflow "
                    "project of type 'text-image-pairs' is a dataset with no "
                    "served model and always returns 500.",
                    self._max_consecutive_errors,
                    self.model_id,
                )
            return None, 0.0
        return self._parse(result)

    def _parse(self, result: Any) -> tuple[Optional[str], float]:
        if not isinstance(result, dict):
            return None, 0.0
        predictions = result.get("predictions")

        # --- classification: {"7": {"confidence": 0.9}, ...} ---------------
        if isinstance(predictions, dict):
            best_text, best_conf = None, 0.0
            for label, payload in predictions.items():
                conf = float((payload or {}).get("confidence", 0.0))
                number = clean_jersey_number(label)
                if number is not None and conf > best_conf:
                    best_text, best_conf = number, conf
            return best_text, best_conf

        if not isinstance(predictions, list) or not predictions:
            return None, 0.0

        # Some classification deployments still return a list; those entries
        # carry no geometry, so treat them as whole-number candidates.
        if all(p.get("x") is None for p in predictions if isinstance(p, dict)):
            return _best_digit_candidate(
                [
                    (str(p.get("class", "")), float(p.get("confidence", 0.0)))
                    for p in predictions
                    if isinstance(p, dict)
                ]
            )

        # --- per-digit detection: sort left to right and concatenate --------
        digits = [
            p
            for p in predictions
            if isinstance(p, dict)
            and str(p.get("class", "")).isdigit()
            and float(p.get("confidence", 0.0)) >= self.min_digit_confidence
        ]
        if not digits:
            return None, 0.0
        digits.sort(key=lambda p: float(p.get("x", 0.0)))
        text = "".join(str(p["class"]) for p in digits)
        number = clean_jersey_number(text)
        if number is None:
            return None, 0.0
        # The number is only as trustworthy as its WEAKEST digit: reading "2"
        # confidently and "3" barely still means "23" is barely read.
        confidence = min(float(p.get("confidence", 0.0)) for p in digits)
        return number, confidence


def _best_digit_candidate(
    candidates: Sequence[tuple[str, float]],
) -> tuple[Optional[str], float]:
    """Pick the highest-confidence candidate that cleans to a legal number."""
    best_number: Optional[str] = None
    best_score = 0.0
    for text, score in candidates:
        number = clean_jersey_number(text)
        if number is not None and float(score) > best_score:
            best_number, best_score = number, float(score)
    return best_number, best_score


#: Backend name -> zero-argument factory. ``auto`` is resolved by
#: ``make_ocr_backend`` in preference order rather than being listed here.
OCR_BACKENDS: dict[str, Any] = {
    "parseq": PARSeqJerseyOCR,
    "local": LocalJerseyClassifier,
    "roboflow": RoboflowJerseyOCR,
    "paddleocr": PaddleJerseyOCR,
    "easyocr": EasyJerseyOCR,
    "pytesseract": PytesseractJerseyOCR,
}

#: Tried in this order for ``backend: auto``.
#:
#: ``local`` first: a classifier trained on jersey numbers beats a general
#: recognizer and costs ~1-2 ms. It is skipped automatically when no checkpoint
#: has been trained yet, which is why ``parseq`` sits immediately behind it -
#: PARSeq needs no training at all and is the strongest zero-setup option.
#:
#: Tesseract is last on purpose: it is the only entry not designed for small,
#: low-contrast, motion-blurred text, and picking it when a better option is
#: available would quietly cap the whole feature's accuracy.
_AUTO_ORDER = ("local", "parseq", "paddleocr", "easyocr", "pytesseract")


def make_ocr_backend(
    name: str = "auto",
    use_gpu: bool = True,
    roboflow: Optional["RoboflowConfig"] = None,
    local: Optional["LocalClassifierConfig"] = None,
    parseq: Optional["PARSeqConfig"] = None,
) -> Any:
    """Build an OCR backend by name, or auto-select the best installed one.

    Raises on an unknown name rather than silently falling back: a typo in the
    config would otherwise send the run to tesseract and look like a tuning
    problem for the rest of the day.

    ``auto`` prefers the LOCAL classifier when its weights exist: it is both the
    most accurate option (trained on jersey numbers specifically) and the
    fastest (~1-2 ms, no network). It silently falls through when no checkpoint
    has been trained yet, so a fresh clone still runs.

    ``roboflow`` is never part of the ``auto`` chain. It is a network service
    with a per-call cost and a key, so it is only ever used when asked for
    explicitly.
    """
    key = (name or "auto").strip().lower()

    if key == "auto":
        for candidate in _AUTO_ORDER:
            if candidate == "local" and (
                local is None or local.weights is None or not Path(local.weights).exists()
            ):
                LOGGER.info(
                    "Local jersey classifier checkpoint is unavailable (%s); "
                    "auto-selection will try PARSeq next.",
                    getattr(local, "weights", None),
                )
                continue
            backend = _construct(candidate, use_gpu, roboflow, local, parseq)
            if backend.is_available():
                LOGGER.info("Jersey OCR backend auto-selected: %s", candidate)
                if candidate == "pytesseract":
                    LOGGER.warning(
                        "Falling back to tesseract, the weakest available jersey "
                        "recognizer. Expect poor accuracy on small numbers. "
                        "Install a better one:  pip install paddleocr  (or easyocr)"
                    )
                return backend
        LOGGER.error(
            "No jersey OCR backend is available (tried %s). Numbers will stay "
            "unknown. Best fix: train the local classifier -\n"
            "  python scripts/download_jersey_ocr_dataset.py --out data/jersey_ocr\n"
            "  python train/train_jersey.py --data data/jersey_ocr",
            ", ".join(_AUTO_ORDER),
        )
        return _construct("pytesseract", use_gpu, roboflow, local, parseq)

    if key not in OCR_BACKENDS:
        raise ValueError(
            f"Unknown jersey OCR backend {name!r}. "
            f"Valid: {sorted(OCR_BACKENDS)} or 'auto'."
        )
    return _construct(key, use_gpu, roboflow, local, parseq)


def _construct(
    key: str,
    use_gpu: bool,
    roboflow: Optional["RoboflowConfig"] = None,
    local: Optional["LocalClassifierConfig"] = None,
    parseq: Optional["PARSeqConfig"] = None,
) -> Any:
    if key == "parseq":
        config = parseq or PARSeqConfig()
        if not use_gpu:
            config = replace(config, device="cpu")
        return PARSeqJerseyOCR(config)
    if key == "local":
        return LocalJerseyClassifier(local or LocalClassifierConfig())
    if key == "roboflow":
        cfg = roboflow or RoboflowConfig()
        return RoboflowJerseyOCR(
            model_id=cfg.model_id,
            api_url=cfg.api_url,
            api_key_env=cfg.api_key_env,
            timeout_sec=cfg.timeout_sec,
            min_digit_confidence=cfg.min_digit_confidence,
        )
    if key == "paddleocr":
        return PaddleJerseyOCR(use_gpu=use_gpu)
    if key == "easyocr":
        return EasyJerseyOCR(gpu=use_gpu)
    return PytesseractJerseyOCR()


# --------------------------------------------------------------------------
# OCR input gate  (NEW - no equivalent in any salvaged variant)
# --------------------------------------------------------------------------


@dataclass
class OCRGateConfig:
    """Thresholds deciding whether a crop is worth running OCR on.

    NEW in this project. The salvaged pipeline ran OCR on every matched box on
    every Nth frame regardless of how small or motion-blurred it was, which is
    the main source of junk votes.

    SIZE DEFAULTS ARE DELIBERATELY LOW. The first calibrated run on 1080p 3x3
    footage gated out 308 of 320 candidate crops (96%) at width 24 / height 32,
    which starved the vote quorum below and meant no number could ever lock.
    Jersey boxes on a wide 3x3 camera are routinely 20x25 px. A small crop is
    not a hopeless crop once it is padded and upscaled (see ``crop_jersey``);
    it is only hopeless below roughly 12 px of digit height, which is what
    these values now encode. Raise them only if you measure junk votes, and
    look at ``--dump-jersey-crops`` output before you do.
    """

    min_box_width: int = 12
    min_box_height: int = 14
    # Set to 0.0 to disable the sharpness test (it costs one Laplacian per crop).
    min_laplacian_variance: float = 0.0

    #: Fraction of box width/height added on EACH side before recognition.
    #: Absorbs the detector's high-IoU localization error - see the module
    #: docstring. 0.0 reproduces the old tight-crop behaviour.
    pad_ratio: float = 0.18
    #: Upscale the padded crop until it is at least this tall before OCR.
    #: Recognizers are trained on text far larger than a 25 px jersey number;
    #: cubic upscaling to a usable size costs microseconds and is the single
    #: cheapest accuracy win available here. 0 disables upscaling.
    upscale_to_height: int = 96


def laplacian_variance(crop: "np.ndarray") -> float:
    """Focus measure: variance of the Laplacian. Higher means sharper."""
    import cv2

    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


#: Reasons a crop was not sent to OCR. Counted separately by the pipeline so a
#: starved run says WHICH threshold starved it - the previous single
#: "skipped by the size/blur gate" counter could not distinguish a too-small box
#: from a too-blurry one, which made the 96% skip rate impossible to diagnose.
GATE_OK = "ok"
GATE_TOO_SMALL = "too_small"
GATE_TOO_BLURRY = "too_blurry"


def gate_reason(
    bbox_xyxy: list[float],
    cfg: OCRGateConfig,
    crop: Optional["np.ndarray"] = None,
) -> str:
    """Why this box was or was not sent to OCR. One of the GATE_* constants."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if (x2 - x1) < float(cfg.min_box_width) or (y2 - y1) < float(cfg.min_box_height):
        return GATE_TOO_SMALL
    if cfg.min_laplacian_variance > 0.0 and crop is not None:
        if laplacian_variance(crop) < float(cfg.min_laplacian_variance):
            return GATE_TOO_BLURRY
    return GATE_OK


def should_run_ocr(
    bbox_xyxy: list[float],
    cfg: OCRGateConfig,
    crop: Optional["np.ndarray"] = None,
) -> bool:
    """True when the box is big enough (and sharp enough) to attempt OCR.

    Thin bool wrapper over ``gate_reason`` so callers that only need a yes/no
    (and the pure-logic tests) do not have to know the reason vocabulary.
    """
    return gate_reason(bbox_xyxy, cfg, crop) == GATE_OK


def torso_region_from_player_box(
    bbox_xyxy: Sequence[float],
    top_fraction: float = 0.15,
    bottom_fraction: float = 0.55,
    width_fraction: float = 0.60,
) -> list[float]:
    """Approximate the jersey-number region from a PLAYER box.

    THE FALLBACK FOR WEIGHTS WITH NO JERSEY CLASS. The pipeline normally takes
    the number region from the detector's ``jersey`` class, which is far more
    accurate. But a checkpoint trained without that class produces no jersey
    boxes at all, and then the OCR loop never runs even once - no crops, no
    calls, no skips, nothing in the summary to explain the silence.

    Geometry is the standard torso approximation used when only a person box is
    available: the number sits in the upper-middle of the torso, below the head
    and above the waist. Defaults take the band from 15% to 55% of box height
    and the middle 60% of its width.

    This is a DEGRADED path and should read as one. It cannot know whether the
    player is facing away, and on a side-on player it crops mostly empty shirt.
    Prefer retraining with a jersey class; use this to get numbers today.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    height = y2 - y1
    width = x2 - x1
    if height <= 0 or width <= 0:
        return [x1, y1, x2, y2]

    center_x = (x1 + x2) / 2.0
    half_width = width * float(width_fraction) / 2.0
    return [
        center_x - half_width,
        y1 + height * float(top_fraction),
        center_x + half_width,
        y1 + height * float(bottom_fraction),
    ]


def padded_box(
    bbox_xyxy: Sequence[float],
    cfg: OCRGateConfig,
    frame_shape: Sequence[int],
) -> Optional[tuple[int, int, int, int]]:
    """The integer box ``crop_jersey`` actually cuts, before any upscaling.

    Split out of ``crop_jersey`` so that anything needing the SAME pixels in
    frame coordinates - dataset mining, a labelling tool, a debug overlay - can
    ask for the box instead of re-deriving ``pad_ratio`` from the docstring.
    Two copies of this arithmetic is exactly how training crops drift away from
    inference crops, and that drift is invisible in every metric you have.

    ``frame_shape`` is anything with ``(height, width)`` first, i.e. a numpy
    ``.shape``. Returns None when the padded box is degenerate or entirely
    outside the frame - the same conditions under which ``crop_jersey`` returns
    None.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)

    pad_x = (x2 - x1) * float(cfg.pad_ratio)
    pad_y = (y2 - y1) * float(cfg.pad_ratio)
    px1 = int(round(max(0.0, x1 - pad_x)))
    py1 = int(round(max(0.0, y1 - pad_y)))
    px2 = int(round(min(float(width), x2 + pad_x)))
    py2 = int(round(min(float(height), y2 + pad_y)))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def crop_jersey(
    frame: "np.ndarray",
    bbox_xyxy: Sequence[float],
    cfg: OCRGateConfig,
) -> Optional["np.ndarray"]:
    """Cut the jersey box out of a frame, padded and upscaled for recognition.

    Three things happen here, in order, and all three matter:

    1. the box is expanded by ``pad_ratio`` on each side, so the detector's
       few-pixel edge error cannot clip a digit stroke;
    2. the expansion is clamped to the frame, so a player at the image border
       still yields a crop instead of an empty array;
    3. the result is cubic-upscaled to ``upscale_to_height``, because every
       available recognizer degrades sharply below ~30 px of text height.

    Returns None only when the box is degenerate or entirely outside the frame.
    """
    import cv2

    box = padded_box(bbox_xyxy, cfg, frame.shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    target = int(cfg.upscale_to_height)
    if target > 0 and crop.shape[0] < target:
        scale = target / float(crop.shape[0])
        crop = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), target),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


# --------------------------------------------------------------------------
# Cross-frame stabilization
# --------------------------------------------------------------------------


@dataclass
class JerseyStabilizerConfig:
    """Voting / locking parameters.

    WARNING - see the Phase A report. ``number_vote_window``,
    ``slot_lock_min_votes`` and ``slot_reassign_patience`` are read as
    ``args.*`` by the source (lines 1593-1624) but are NEVER declared in its
    argparse block (lines 2268-2360) and never set via setattr/set_defaults.
    The source therefore raises AttributeError before it can lock a number, and
    no tuned value for these three survives anywhere in SRC_A. The values below
    are DEFAULTS CHOSEN HERE, not recovered constants, and need calibration.
    """

    number_vote_window: int = 15
    slot_lock_min_votes: int = 3
    slot_reassign_patience: int = 8
    # Salvaged verbatim (line 1590): observations weaker than this never vote.
    min_observation_confidence: float = 0.55
    # Salvaged verbatim (line 1595): >= this confidence counts as 2 votes.
    strong_vote_confidence: float = 0.85


@dataclass
class TrackNumberState:
    """Per-track jersey state.

    The number-related subset of ``FixedPlayerSlot`` (source lines 116-133).
    The bbox/centre/slot-assignment fields are intentionally not carried over:
    they belong to the fixed-slot assignment algorithm, not to number voting,
    and this project keys state on tracker IDs instead of fixed slots.
    """

    track_id: int
    jersey_number: Optional[str] = None
    number_votes: Counter = field(default_factory=Counter)
    is_locked: bool = False
    lock_number: Optional[str] = None
    lock_confidence: float = 0.0
    lock_frame_count: int = 0
    conflict_frame_count: int = 0
    recent_number_history: deque = field(default_factory=deque)
    recent_number_confidences: deque = field(default_factory=deque)
    last_seen_frame: int = -1


def trim_deque(values: deque, max_len: int) -> None:
    """Salvaged verbatim from source lines 183-185."""
    while len(values) > max_len:
        values.popleft()


def majority_number_from_recent_history(
    state: TrackNumberState, window: int
) -> tuple[Optional[str], int, float]:
    """Salvaged verbatim from source lines 188-200 (``slot`` renamed ``state``)."""
    if window <= 0 or not state.recent_number_history:
        return None, 0, 0.0
    recent_numbers = list(state.recent_number_history)[-window:]
    recent_confs = list(state.recent_number_confidences)[-window:]
    if not recent_numbers:
        return None, 0, 0.0

    counts = Counter(recent_numbers)
    best_number, best_count = counts.most_common(1)[0]
    confs = [recent_confs[i] for i, number in enumerate(recent_numbers) if number == best_number]
    mean_conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
    return best_number, int(best_count), mean_conf


class JerseyNumberStabilizer:
    """Aggregates per-frame OCR guesses into one stable number per track.

    The state machine in ``update`` is a direct transcription of
    ``update_slot_number_state`` (source lines 1583-1625). Only the plumbing
    changed: state is keyed by ``track_id`` and looked up here rather than
    being passed in as a pre-resolved slot, and parameters come from a config
    dataclass instead of an argparse namespace.
    """

    def __init__(self, config: Optional[JerseyStabilizerConfig] = None) -> None:
        self.config = config or JerseyStabilizerConfig()
        self.states: dict[int, TrackNumberState] = {}

    def _state(self, track_id: int) -> TrackNumberState:
        state = self.states.get(int(track_id))
        if state is None:
            state = TrackNumberState(track_id=int(track_id))
            self.states[int(track_id)] = state
        return state

    def update(
        self,
        track_id: int,
        raw_text: Optional[str],
        confidence: float,
        frame_idx: int,
    ) -> Optional[str]:
        """Feed one OCR observation for one track. Returns the current best number.

        ``raw_text`` may be None (OCR skipped or failed): the source still
        re-evaluates the lock on every call, so a skipped observation can still
        confirm a pending lock. ``frame_idx`` is recorded for observability and
        deliberately does not affect the voting decision, matching the source.
        """
        cfg = self.config
        state = self._state(track_id)
        state.last_seen_frame = int(frame_idx)

        observed_number = clean_jersey_number(raw_text)
        observed_conf = float(confidence)

        if observed_number is not None and observed_conf >= cfg.min_observation_confidence:
            state.recent_number_history.append(observed_number)
            state.recent_number_confidences.append(observed_conf)
            trim_deque(state.recent_number_history, int(cfg.number_vote_window))
            trim_deque(state.recent_number_confidences, int(cfg.number_vote_window))
            state.number_votes[observed_number] += (
                1 if observed_conf < cfg.strong_vote_confidence else 2
            )

        recent_number, recent_count, recent_conf = majority_number_from_recent_history(
            state, int(cfg.number_vote_window)
        )
        required_recent_votes = max(
            2, min(int(cfg.number_vote_window), max(3, int(cfg.slot_lock_min_votes)))
        )

        if recent_number is None or recent_count < required_recent_votes:
            return self.get_stable_number(track_id)

        if state.is_locked and state.lock_number and recent_number != state.lock_number:
            state.conflict_frame_count += 1
            if state.conflict_frame_count >= int(cfg.slot_reassign_patience):
                state.is_locked = False
                state.lock_number = None
                state.lock_confidence = 0.0
                state.lock_frame_count = 0
                state.conflict_frame_count = 0
            else:
                return self.get_stable_number(track_id)
        else:
            state.conflict_frame_count = 0

        state.jersey_number = recent_number
        if state.lock_number == recent_number or state.lock_number is None:
            state.lock_frame_count += 1
        else:
            state.lock_frame_count = 1
        state.lock_number = recent_number
        state.lock_confidence = min(1.0, 0.55 + 0.1 * recent_count + 0.2 * recent_conf)

        if state.lock_frame_count >= int(cfg.slot_lock_min_votes):
            state.is_locked = True

        return self.get_stable_number(track_id)

    def get_stable_number(self, track_id: int) -> Optional[str]:
        """The number to display for this track, or None if not yet decided.

        Prefers the locked number, mirroring the source's precedence
        (``slot.lock_number or slot.jersey_number``, line 1641).
        """
        state = self.states.get(int(track_id))
        if state is None:
            return None
        if state.is_locked and state.lock_number:
            return state.lock_number
        return state.jersey_number

    def get_confidence(self, track_id: int) -> float:
        state = self.states.get(int(track_id))
        return float(state.lock_confidence) if state is not None else 0.0

    def is_locked(self, track_id: int) -> bool:
        state = self.states.get(int(track_id))
        return bool(state.is_locked) if state is not None else False

    def snapshot(self) -> dict[int, dict[str, Any]]:
        """Debug view of every track's jersey state."""
        return {
            tid: {
                "jersey_number": s.jersey_number,
                "lock_number": s.lock_number,
                "is_locked": s.is_locked,
                "lock_confidence": s.lock_confidence,
                "lock_frame_count": s.lock_frame_count,
                "conflict_frame_count": s.conflict_frame_count,
                "last_seen_frame": s.last_seen_frame,
                "votes": dict(s.number_votes),
            }
            for tid, s in self.states.items()
        }
