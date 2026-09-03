"""Training-time augmentation for jersey digit crops.

    from train.augment import build_transforms
    train_tf = build_transforms(train=True)
    val_tf   = build_transforms(train=False)

THE AUGMENTATION IS THE DOMAIN ADAPTATION
-----------------------------------------
Every source of training data for this model is cleaner than deployment. SVHN
is a photograph of a sign that someone stood still to capture; the synthetic
numbers start life as a crisp font render; SoccerNet is broadcast footage.
Deployment is a fixed wide 3x3 camera where the number is 20-40 px tall, the
player is running, and the encoder has already thrown away the high
frequencies. A model trained on the source domain as-is validates beautifully
and collapses on real input.

So the transforms here are not regularisation. Each one closes a specific,
named gap between where the pixels came from and where they are going:

    RandomDownscale     resolution   - the big one, see its docstring
    RandomMotionBlur    exposure     - the player moved during the frame
    GaussianBlur        optics       - a wide lens is not sharp at the edges
    RandomJPEG          codec        - what the video decode path leaves behind
    ColorJitter         lighting     - indoor/outdoor, kit colours, white balance
    RandomAffine        localization - see BOX JITTER below
    RandomErasing       occlusion    - an arm, another player, the ball

BOX JITTER IS NOT OPTIONAL
--------------------------
The affine's translate and scale ranges (+/-15%) exist so the model never gets
to assume the digits are perfectly centred. At inference the crop comes from a
detector with real localization error; during annotation it comes from a human
dragging a box by hand. Those two produce differently-framed crops, and a model
trained only on tight centred examples will quietly score them differently. The
jitter makes the model indifferent to which one it is looking at, which is the
only reason hand-drawn boxes and detector boxes can share a training set at all.

NO FLIPS. NOT HORIZONTAL, NOT VERTICAL, NOT VIA A NEGATIVE SCALE
----------------------------------------------------------------
A horizontal flip turns a 2 into a shape that is not a 2 while keeping the
label "2" - it teaches the model that digit identity is not about stroke
direction, which is the one thing it must learn. A vertical flip is worse: it
maps 6 onto 9 and 9 onto 6 with the labels left untouched, so it actively
trains the model to confuse the single pair this project can least afford to
get wrong. Both are in nearly every default recipe. Neither belongs here, and
``scale`` below stays strictly positive so ``RandomAffine`` cannot sneak a
mirror in through the back door.
"""

from __future__ import annotations

import io
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

from jersey_net import INPUT_SIZE, MEAN, STD


class RandomDownscale:
    """Crush to a realistic capture resolution, then restore to INPUT_SIZE.

    THE most important transform in this file. Source images carry stroke
    detail at a spatial frequency that simply does not survive a 20-40 px
    jersey number; a model that leans on that detail has nothing left to look
    at in deployment. Round-tripping through a small size destroys the same
    information the real camera destroys, so the network is forced onto
    features that survive the trip - overall stroke topology, aspect ratio,
    hole count.

    The 0.4 floor puts the low end at ~38 px, which is the upper half of the
    real range; the synthetic generator pushes further down (20-40 px) because
    it starts from a perfectly sharp render and has more to give away.
    """

    def __init__(self, min_scale: float = 0.4, max_scale: float = 1.0, probability: float = 0.9) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        scale = random.uniform(self.min_scale, self.max_scale)
        target = max(8, int(round(INPUT_SIZE * scale)))
        small = image.resize((target, target), Image.BILINEAR)
        return small.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)


class RandomMotionBlur:
    """Directional smear along a random axis: the player moved mid-exposure.

    Implemented as a depthwise conv with a drawn line kernel rather than a
    library call, so this file needs nothing beyond torch. The angle is
    uniform because players run in every direction relative to a fixed camera.
    """

    def __init__(self, max_kernel: int = 9, probability: float = 0.35) -> None:
        self.max_kernel = max_kernel
        self.probability = probability

    def _line_kernel(self, size: int, angle_deg: float) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        radius = (size - 1) / 2.0
        dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        # Walk the line out from the centre in sub-pixel steps and light up
        # whichever cell each step lands in; simpler than rasterising properly
        # and the kernel only has to be plausible, not exact.
        for t in np.linspace(-radius, radius, size * 4):
            r = int(round(radius + t * dy))
            c = int(round(radius + t * dx))
            if 0 <= r < size and 0 <= c < size:
                kernel[r, c] = 1.0
        total = kernel.sum()
        return kernel / total if total > 0 else kernel

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        size = random.choice([3, 5, 7, self.max_kernel])
        kernel = torch.from_numpy(self._line_kernel(size, random.uniform(0, 180)))

        array = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
        weight = kernel.expand(array.shape[0], 1, size, size).contiguous()
        pad = size // 2
        blurred = F.conv2d(
            F.pad(array.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"),
            weight,
            groups=array.shape[0],
        ).squeeze(0)
        out = blurred.clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(out)


class RandomGaussianBlur:
    """Defocus and lens softness, on top of any motion smear."""

    def __init__(self, max_radius: float = 1.4, probability: float = 0.3) -> None:
        self.max_radius = max_radius
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, self.max_radius)))


class RandomJPEG:
    """Recompression artifacts, as produced by the video decode path.

    Applied after the blurs because that is the real order of operations: the
    camera and the motion soften the image, then the encoder quantises what is
    left. Doing it first would let the resize smooth the blocking away.
    """

    def __init__(self, min_quality: int = 25, max_quality: int = 85, probability: float = 0.35) -> None:
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(self.min_quality, self.max_quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_transforms(train: bool):
    """Training pipeline, or the bare resize+normalize used for validation.

    Validation gets NO augmentation - the point of the split is to measure the
    model against a fixed target, and a randomised val set makes epoch-to-epoch
    comparisons noise. It is also what the inference side does, so val numbers
    describe the deployed preprocessing.
    """
    if not train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        # Rotation, box jitter (translate/scale) and shear in one resampling
        # step - chaining separate affines would blur the image three times
        # over for no extra variety. scale stays strictly positive: a negative
        # entry here is a mirror, see the module docstring.
        transforms.RandomAffine(
            degrees=10,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=(-6, 6, -6, 6),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        RandomMotionBlur(),
        RandomDownscale(),
        RandomGaussianBlur(),
        RandomJPEG(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # Last, in tensor space: an eraser patch should not be blurred or
        # rescaled afterwards, because a real occluder is in front of the
        # optics, not behind them.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
