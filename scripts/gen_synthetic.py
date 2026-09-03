"""Stage 2 data: render all 101 legal jersey numbers, then destroy them.

    python scripts/gen_synthetic.py                    # 101 x 500 = 50,500 crops
    python scripts/gen_synthetic.py --per-number 50    # quick look first
    python scripts/gen_synthetic.py --list-fonts       # what this box can render

WHY SYNTHETIC DATA IS NOT OPTIONAL HERE
---------------------------------------
FIBA 3x3 allows 0, 00 and 1-99 - 101 numbers - and any one match shows eight of
them. No amount of annotating real footage closes that gap: to see number 87
the pipeline needs a player wearing 87 to have been filmed. Synthetic rendering
is the only source that can cover the full legal range, so it is load-bearing
rather than supplementary. Every number gets the same count, which also means
the long tail is not a long tail here - the rarity of 87 in the real world is
a fact about rosters, not about what a digit looks like.

What it cannot provide is realism, which is what stages 1 and 3 are for. A font
render is not a photograph of fabric stretched over a moving shoulder. Mixing
this 1:1 with SoccerNet in stage 3 is what keeps the model from overfitting to
clean glyph shapes.

THE DEGRADATION ORDER IS PHYSICAL, NOT ARBITRARY
------------------------------------------------
Each step models a stage of the real capture path, applied in the order the
photons and bits actually meet them:

    1. occlusion     an arm, another player, the ball - part of the SCENE
    2. perspective   the player is not square to a fixed wide camera
    3. motion blur   they moved while the shutter was open
    4. downscale     THE CORE STEP - crush to 20-40 px, the real size of a
                     jersey number in this footage, then restore to 96
    5. gaussian      lens softness, worst at the edges of a wide field
    6. JPEG          what the encoder throws away

Step 4 is the one that makes this data useful. Without it the model trains on
crisp glyphs and learns fine stroke detail that does not survive to deployment,
producing a model with excellent validation numbers that collapses on real
input. Applying it AFTER the motion blur and BEFORE the codec matters too:
blurring a small image is not the same as shrinking a blurred one, and a real
encoder compresses what the optics already softened.

OCCLUSION GOES FIRST, NOT LAST
------------------------------
An occluder is in front of the jersey but still behind the lens, so it is
imaged through the same optics as everything else: it gets the same motion
smear, the same 20-40 px crush, the same codec. Pasting it on at the end
instead produces a razor-sharp block on an otherwise soft image - an artifact
no camera can produce, and a free cue the model can learn to key on. The first
contact sheet made this obvious, which is what the contact sheet is for.

The occluders also refuse to bury the number: ``occlude`` tracks which pixels
the digits actually cover and rejects any block that would hide more than
``MAX_DIGIT_COVERAGE`` of them. A crop whose number is entirely behind an arm
is not a hard example, it is a WRONG LABEL - it teaches the model that a blank
patch of fabric reads as 63. Partial occlusion is exactly what we want;
total occlusion is poison.

The train-time augmentation in train/augment.py then applies its own
degradation on top. That is intentional - these images are the *starting*
point for augmentation, not a finished training set - which is exactly why the
contact sheet matters: if a number is unreadable here, augmentation only makes
it worse, and those samples teach the model noise.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augment import RandomMotionBlur
from jersey_net import INPUT_SIZE

REPO = Path(__file__).resolve().parent.parent

# 0, 00, then 1-99. Written out rather than derived so the 0/00 distinction is
# visible in the source instead of hiding in a formatting rule.
LEGAL_NUMBERS: list[str] = ["0", "00"] + [str(n) for n in range(1, 100)]

# Rendered large, degraded, then delivered at INPUT_SIZE. Starting above the
# target means the perspective warp and blur have real pixels to work with -
# degrading at 96 and upscaling would bake in resampling artifacts that no
# camera produces.
CANVAS = 192

# An occluder may hide at most this share of the digit pixels. Past it the
# label stops describing the image - see OCCLUSION GOES FIRST above.
MAX_DIGIT_COVERAGE = 0.45

# Plausible occluders: skin tones, kit colours, shadow, and the ball. Sampling
# real-ish colours instead of uniform RGB matters because a saturated magenta
# block is not a thing that ever crosses in front of a player, and the model
# would learn "bright rectangle" as its occlusion cue rather than "the strokes
# stop here".
OCCLUDER_COLOURS = (
    (196, 158, 128), (150, 112, 84), (98, 70, 52), (232, 198, 172),   # skin
    (28, 28, 32), (58, 58, 66), (12, 12, 14),                          # shadow
    (208, 96, 40), (176, 74, 30),                                      # the ball
    (232, 232, 236), (190, 190, 196),                                  # white kit
    (30, 52, 120), (140, 26, 34), (24, 88, 52),                        # kit colours
)

# Jersey typography is bold, wide, and sans-serif for readability at distance.
# Ranked by how close each is to that; whatever the box has, in this order.
FONT_PREFERENCES = (
    "arialbd", "arial bold", "helveticabold", "impact",
    "dejavusans-bold", "dejavusansbold", "liberationsans-bold",
    "freesansbold", "notosans-bold", "roboto-bold", "opensans-bold",
    "verdanab", "tahomabd", "trebucbd", "calibrib", "segoeuib",
    "arialnb", "franklingothic", "oswald", "bebas",
)

FONT_DIRS = (
    "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts",
    "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts",
)


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------


def discover_fonts(limit: int = 12) -> list[Path]:
    """Find bold sans-serif TTFs, preferring the most jersey-like available.

    Multiple fonts, chosen at random per image, is the point: a single font
    teaches the model that font rather than the digit. Serif and script faces
    are excluded because no basketball kit uses them, and training on shapes
    the deployment domain never contains just spends capacity.
    """
    found: dict[str, Path] = {}
    for directory in FONT_DIRS:
        base = Path(directory).expanduser()
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in {".ttf", ".otf"}:
                continue
            stem = path.stem.lower().replace(" ", "").replace("_", "")
            if any(bad in stem for bad in ("italic", "oblique", "serif", "mono", "script", "emoji")):
                # "sansserif" contains "serif"; keep those.
                if "sansserif" not in stem and "sans-serif" not in stem:
                    continue
            if any(pref.replace(" ", "").replace("-", "") in stem.replace("-", "") for pref in FONT_PREFERENCES):
                found.setdefault(stem, path)

    ranked = sorted(
        found.values(),
        key=lambda p: next(
            (i for i, pref in enumerate(FONT_PREFERENCES)
             if pref.replace(" ", "").replace("-", "") in p.stem.lower().replace(" ", "").replace("_", "").replace("-", "")),
            len(FONT_PREFERENCES),
        ),
    )

    if not ranked:
        # Any bold sans is better than PIL's 11px bitmap default, which cannot
        # be scaled and would make every sample identical.
        for directory in FONT_DIRS:
            base = Path(directory).expanduser()
            if base.is_dir():
                ranked = [p for p in base.rglob("*.ttf")][:limit]
                if ranked:
                    print(f"WARNING: no preferred jersey font found, falling back to {len(ranked)} arbitrary faces")
                    break

    return ranked[:limit]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _random_palette(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """A jersey colour and a number colour with real contrast between them.

    Half dark-on-light, half light-on-dark: both are common in kit design, and
    a model trained on one polarity leans on absolute brightness instead of on
    stroke shape.
    """
    if rng.random() < 0.5:
        background = tuple(rng.randint(0, 90) for _ in range(3))
        foreground = tuple(rng.randint(170, 255) for _ in range(3))
    else:
        background = tuple(rng.randint(165, 255) for _ in range(3))
        foreground = tuple(rng.randint(0, 85) for _ in range(3))
    return background, foreground


def _fabric(size: int, colour: tuple[int, int, int], rng: random.Random) -> Image.Image:
    """Flat colour plus a soft gradient and sensor noise.

    A perfectly flat background is a cue no real crop has; the model would
    learn to expect it and treat real fabric variation as signal.
    """
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[:, :] = colour

    # Directional shading, standing in for a fold and for uneven court lighting.
    ramp = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(ramp, ramp)
    shade = (gx * rng.uniform(-1, 1) + gy * rng.uniform(-1, 1)) * rng.uniform(6, 26)
    base += shade[:, :, None]
    base += np.random.normal(0, rng.uniform(1.5, 6.0), base.shape).astype(np.float32)

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def render_number(
    number: str, font_path: Path | None, rng: random.Random
) -> tuple[Image.Image, Image.Image, tuple[int, int, int]]:
    """Draw ``number`` on fabric at CANVAS resolution.

    Returns the image, a mask of the digit strokes, and the background colour.
    The mask is what lets ``occlude`` tell "covered part of a digit" from
    "covered some empty jersey"; the background colour is what the perspective
    warp fills its corners with, so the tilt does not introduce a colour that
    is nowhere else in the crop.
    """
    background, foreground = _random_palette(rng)
    canvas = _fabric(CANVAS, background, rng)
    draw = ImageDraw.Draw(canvas)

    # Kept clear of 0.72: at that size a perspective tilt pushes strokes past
    # the frame edge, and a crop missing half its second digit is another way
    # to produce a wrong label.
    target_height = int(CANVAS * rng.uniform(0.40, 0.60))
    font = None
    if font_path is not None:
        # Point size and rendered cap height are not the same thing and the
        # ratio differs per face, so measure and correct rather than guess.
        probe = ImageFont.truetype(str(font_path), 100)
        box = probe.getbbox(number)
        rendered = max(box[3] - box[1], 1)
        font = ImageFont.truetype(str(font_path), max(8, int(100 * target_height / rendered)))
    if font is None:
        font = ImageFont.load_default()

    box = draw.textbbox((0, 0), number, font=font)
    x = (CANVAS - (box[2] - box[0])) / 2 - box[0]
    y = (CANVAS - (box[3] - box[1])) / 2 - box[1]
    # Off-centre on purpose: real crops are not perfectly framed, and this
    # compounds with the box jitter in augment.py.
    x += rng.uniform(-0.04, 0.04) * CANVAS
    y += rng.uniform(-0.04, 0.04) * CANVAS

    # Kits very often outline the number in a third colour. Without this the
    # model never sees a digit whose edge is not a straight fabric/ink step.
    stroke_width = rng.choice([0, 0, 2, 3, 4])
    stroke_fill = tuple(rng.randint(0, 255) for _ in range(3)) if stroke_width else None

    draw.text((x, y), number, font=font, fill=foreground,
              stroke_width=stroke_width, stroke_fill=stroke_fill)

    # Same glyph, same position, drawn white on black: a pixel-exact record of
    # where the digits are. Deriving it from a colour threshold on the finished
    # image would misfire whenever the outline or the fabric shading happens to
    # land near the foreground colour.
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).text(
        (x, y), number, font=font, fill=255,
        stroke_width=stroke_width, stroke_fill=255 if stroke_width else None,
    )
    return canvas, mask, background


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


def _perspective_coeffs(source: list[tuple[float, float]], dest: list[tuple[float, float]]) -> list[float]:
    """Solve the 8 coefficients PIL needs to map ``dest`` back to ``source``."""
    matrix = []
    for (x, y), (X, Y) in zip(dest, source):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(source, dtype=np.float64).reshape(8)
    return list(np.linalg.solve(A, B))


def perspective(
    image: Image.Image, rng: random.Random, fill: tuple[int, int, int], strength: float = 0.13
) -> Image.Image:
    """Tilt the plane: a torso is rarely square to a fixed wide camera.

    Corner offsets are independent, so this produces genuine out-of-plane
    rotation rather than the shear an affine would give - the digits get a
    real near/far size difference across the crop.

    ``fill`` is the jersey colour rather than a random one: the corners the
    warp exposes are more jersey, not a colour that appears nowhere else in
    the image and would mark every warped sample as synthetic.
    """
    size = image.size[0]
    jitter = lambda: rng.uniform(-strength, strength) * size
    corners = [(0, 0), (size, 0), (size, size), (0, size)]
    warped = [(x + jitter(), y + jitter()) for x, y in corners]
    coeffs = _perspective_coeffs(corners, warped)
    return image.transform(
        (size, size), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=fill,
    )


def crush_resolution(image: Image.Image, rng: random.Random, min_px: int = 20, max_px: int = 40) -> Image.Image:
    """THE step. Shrink to a real jersey-number size, then restore to 96.

    20-40 px is measured from this project's own wide-camera footage, not
    picked for convenience. AREA on the way down (what a sensor actually does
    when it integrates over a pixel) and BICUBIC on the way up (what the
    pipeline's own upscaler does), so both halves of the trip match reality.
    """
    target = rng.randint(min_px, max_px)
    small = image.resize((target, target), Image.BOX)
    return small.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)


def gaussian(image: Image.Image, rng: random.Random) -> Image.Image:
    from PIL import ImageFilter

    if rng.random() < 0.35:
        return image
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 1.1)))


def jpeg(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(22, 80))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def occlude(
    image: Image.Image, digits: Image.Image, rng: random.Random, max_fraction: float = 0.30
) -> Image.Image:
    """Cover up to 30% of the crop, without burying the number.

    Occluders enter from an edge rather than landing in the middle, because
    that is how an arm or a passing player actually crosses the frame - and a
    block anchored to an edge eats into the number from the side instead of
    punching a hole through its centre.

    Every candidate is checked against the digit mask before it is committed:
    if it would push total digit coverage past MAX_DIGIT_COVERAGE it is
    discarded and another is tried. That is the difference between a hard
    example and a wrong label.
    """
    fraction = rng.uniform(0.0, max_fraction)
    if fraction < 0.02:
        return image

    image = image.copy()
    size = image.size[0]
    digit_pixels = np.asarray(digits, dtype=np.uint8) > 127
    total_digits = max(int(digit_pixels.sum()), 1)
    covered = np.zeros_like(digit_pixels)

    budget = fraction * size * size
    attempts = 0

    while budget > 0 and attempts < 12:
        attempts += 1
        area = budget if rng.random() < 0.5 else budget * rng.uniform(0.3, 0.8)
        w = min(size, max(4, int((area * rng.uniform(0.5, 2.0)) ** 0.5)))
        h = min(size, max(4, int(area / max(w, 1))))

        # Anchor to one of the four edges, protruding inward.
        edge = rng.choice(("left", "right", "top", "bottom"))
        if edge == "left":
            x, y = -rng.randint(0, w // 2), rng.randint(-h // 4, size - h // 2)
        elif edge == "right":
            x, y = size - w + rng.randint(0, w // 2), rng.randint(-h // 4, size - h // 2)
        elif edge == "top":
            x, y = rng.randint(-w // 4, size - w // 2), -rng.randint(0, h // 2)
        else:
            x, y = rng.randint(-w // 4, size - w // 2), size - h + rng.randint(0, h // 2)

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(size, x + w), min(size, y + h)
        if x1 <= x0 or y1 <= y0:
            continue

        candidate = covered.copy()
        candidate[y0:y1, x0:x1] = True
        if (digit_pixels & candidate).sum() / total_digits > MAX_DIGIT_COVERAGE:
            continue

        colour = OCCLUDER_COLOURS[rng.randrange(len(OCCLUDER_COLOURS))]
        # Nudge the sampled tone so repeated draws are not pixel-identical.
        colour = tuple(int(np.clip(c + rng.randint(-18, 18), 0, 255)) for c in colour)
        ImageDraw.Draw(image).rectangle([x0, y0, x1, y1], fill=colour)

        covered = candidate
        budget -= (x1 - x0) * (y1 - y0)

    return image


def degrade(
    image: Image.Image,
    digits: Image.Image,
    background: tuple[int, int, int],
    rng: random.Random,
    motion: RandomMotionBlur,
) -> Image.Image:
    """The full capture path, in physical order. See the module docstring.

    Occlusion first, at scene scale, so the occluder is blurred and crushed
    along with the jersey behind it.
    """
    image = occlude(image, digits, rng)
    image = perspective(image, rng, background)
    image = motion(image)
    image = crush_resolution(image, rng)
    image = gaussian(image, rng)
    return jpeg(image, rng)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate_number(job: tuple[str, int, Path, list[Path], int]) -> list[dict]:
    """Render every image for one jersey number. One multiprocessing unit."""
    number, count, out_root, fonts, seed = job
    rng = random.Random(seed)
    motion = RandomMotionBlur(max_kernel=9, probability=0.5)

    folder = out_root / "images" / number
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(count):
        font = rng.choice(fonts) if fonts else None
        rendered, digits, background = render_number(number, font, rng)
        image = degrade(rendered, digits, background, rng, motion)
        relative = f"images/{number}/{i:05d}.jpg"
        image.save(out_root / relative, format="JPEG", quality=92)
        rows.append({"image": relative, "number": number, "font": font.name if font else "default"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "synthetic")
    parser.add_argument("--sheet", type=Path, default=REPO / "data" / "debug" / "synthetic_sample.png")
    parser.add_argument("--per-number", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0, help="0 = serial")
    parser.add_argument("--seed", type=int, default=1143757)
    parser.add_argument("--list-fonts", action="store_true")
    args = parser.parse_args()

    fonts = discover_fonts()
    print(f"fonts ({len(fonts)}):")
    for font in fonts:
        print(f"  {font}")
    if args.list_fonts:
        return
    if not fonts:
        print("\nERROR: no usable TrueType font. Install e.g. fonts-dejavu and rerun.")
        raise SystemExit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (number, args.per_number, args.out, fonts, args.seed + i)
        for i, number in enumerate(LEGAL_NUMBERS)
    ]

    total = len(LEGAL_NUMBERS) * args.per_number
    print(f"\ngenerating {total} images: {len(LEGAL_NUMBERS)} numbers x {args.per_number}")

    rows: list[dict] = []
    if args.workers > 0:
        from multiprocessing import Pool

        with Pool(args.workers) as pool:
            for i, batch in enumerate(pool.imap_unordered(generate_number, jobs)):
                rows.extend(batch)
                print(f"  {i + 1}/{len(jobs)} numbers done ({len(rows)} images)", flush=True)
    else:
        for i, job in enumerate(jobs):
            rows.extend(generate_number(job))
            print(f"  {i + 1}/{len(jobs)} numbers done ({len(rows)} images)", flush=True)

    manifest = {
        "version": "synthetic-v1",
        "per_number": args.per_number,
        "numbers": LEGAL_NUMBERS,
        "input_size": INPUT_SIZE,
        "seed": args.seed,
        "fonts": [f.name for f in fonts],
        "degradation": (
            "occlusion<=30% (digit coverage capped at "
            f"{MAX_DIGIT_COVERAGE:.0%}) -> perspective -> motion blur -> "
            "downscale 20-40px -> gaussian -> jpeg"
        ),
        "rows": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out / 'manifest.json'} ({len(rows)} rows)")

    # Sample from the finished manifest, not from generation, so the sheet
    # shows what the loader will actually read off disk.
    from jersey_data import save_contact_sheet

    picks = random.Random(0).sample(rows, min(20, len(rows)))
    samples = [(Image.open(args.out / r["image"]).convert("RGB"), f"#{r['number']}") for r in picks]
    sheet = save_contact_sheet(samples, args.sheet)

    print(f"wrote {sheet}")
    print(
        "\nSTOP AND LOOK AT THE CONTACT SHEET.\n"
        "  too sharp  -> the model will learn detail that does not survive to deployment\n"
        "  unreadable -> those samples teach noise; ease crush_resolution/occlude\n"
        "You should be able to read most numbers with effort, and fail on a few."
    )


if __name__ == "__main__":
    main()
