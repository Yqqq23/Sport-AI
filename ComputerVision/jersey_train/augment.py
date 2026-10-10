"""域差距增强 —— 这个包里真正决定成败的部分。

问题在哪
--------
可用的公开数据（Roboflow ``basketball-jersey-numbers-ocr`` v7）是 NBA 转播画面
裁出来的 **224×224 清晰特写**。你的部署目标是固定广角机位，同一个号码只有
**20–40 像素高**，而且经常有运动模糊。

直接拿源域训练，会得到一个验证集准确率很好看、真实视频上直接崩掉的模型。
这不是调参能救的，是训练分布和部署分布根本不是一回事。

所以增强不是「锦上添花」，它就是主线：把每张训练图**压到真实分辨率再还原**，
逼网络去学那些经过这一趟还能活下来的特征。

⚠ 绝对不能水平翻转
------------------
几乎所有默认增强配方里都有 RandomHorizontalFlip，这里是**错的**。
数字不是镜像不变的：2 翻过来不是 2，但标签还写着 "2"。等于往训练集里
掺入系统性错标。

所有增强都在 **BGR uint8** 上做，和 ``crop_jersey`` 交给模型的东西同一个空间，
最后统一走 ``contract.preprocess_bgr``。
"""

from __future__ import annotations

import random

import cv2
import numpy as np


class RandomDownscale:
    """压到真实采集分辨率再还原。**最重要的一个变换。**

    3x3 广角机位下的球衣号高 20–40 px，训练图是 224 px。不做这个，
    网络会去学那些在部署时根本不存在的细笔画。
    """

    def __init__(self, min_px: int = 12, max_px: int = 48, p: float = 0.9,
                 out_size: int = 96):
        self.min_px, self.max_px, self.p, self.out_size = min_px, max_px, p, out_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        t = random.randint(self.min_px, self.max_px)
        small = cv2.resize(img, (t, t), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (self.out_size, self.out_size), interpolation=cv2.INTER_CUBIC)


class RandomMotionBlur:
    """方向性模糊，模拟曝光期间球员在动。"""

    def __init__(self, max_kernel: int = 9, p: float = 0.35):
        self.max_kernel, self.p = max_kernel, p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        k = random.choice([3, 5, 7, self.max_kernel])
        kernel = np.zeros((k, k), np.float32)
        ang = random.random()
        if ang < 0.4:
            kernel[k // 2, :] = 1.0                      # 水平
        elif ang < 0.8:
            kernel[:, k // 2] = 1.0                      # 垂直
        else:
            np.fill_diagonal(kernel, 1.0)                # 斜向
        return cv2.filter2D(img, -1, kernel / kernel.sum())


class RandomJPEG:
    """JPEG 压缩伪影。视频流是压缩过的，训练图往往不是。"""

    def __init__(self, min_q: int = 20, max_q: int = 80, p: float = 0.4):
        self.min_q, self.max_q, self.p = min_q, max_q, p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        q = random.randint(self.min_q, self.max_q)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


class RandomBoxJitter:
    """随机裁掉/扩出一点边，复现检测框的定位误差。

    检测器的 mAP50-95 大约 0.64，也就是框本来就不准。推理时喂进来的 crop
    边界是抖的，训练时每张都严丝合缝，模型会对边界过拟合。
    """

    def __init__(self, max_frac: float = 0.14, p: float = 0.8):
        self.max_frac, self.p = max_frac, p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        h, w = img.shape[:2]
        f = self.max_frac
        dx1, dy1 = int(random.uniform(-f, f) * w), int(random.uniform(-f, f) * h)
        dx2, dy2 = int(random.uniform(-f, f) * w), int(random.uniform(-f, f) * h)
        x1, y1 = max(0, dx1), max(0, dy1)
        x2, y2 = min(w, w + dx2), min(h, h + dy2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return img
        return img[y1:y2, x1:x2]


class RandomPhotometric:
    """亮度 / 对比度 / 色调。球馆灯光和转播打光完全不是一回事。"""

    def __init__(self, p: float = 0.8):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        a = random.uniform(0.65, 1.4)                    # 对比度
        b = random.uniform(-35, 35)                      # 亮度
        img = cv2.convertScaleAbs(img, alpha=a, beta=b)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 1.3), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img


class RandomRotate:
    """小角度旋转。球员身体是斜的，号码也就是斜的。

    幅度刻意压得很小（±12°）：转多了 6 和 9 会开始互相靠近。
    """

    def __init__(self, max_deg: float = 12.0, p: float = 0.6):
        self.max_deg, self.p = max_deg, p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-self.max_deg, self.max_deg), 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)


class RandomNoise:
    """高感光噪声。室内球馆快门快、ISO 高。"""

    def __init__(self, max_sigma: float = 14.0, p: float = 0.4):
        self.max_sigma, self.p = max_sigma, p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        sigma = random.uniform(3.0, self.max_sigma)
        noisy = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        return "Compose([\n" + "".join(
            f"    {type(t).__name__}({vars(t)})\n" for t in self.transforms) + "])"


def train_augment(out_size: int = 96, strength: float = 1.0) -> Compose:
    """训练用增强。``strength`` 0→关掉，1→默认，>1→更狠。

    顺序是有讲究的：先几何（jitter/rotate），再压分辨率，最后才是
    模糊/压缩/噪声 —— 因为真实链路就是这个顺序：
    相机拍到小小的号码 → 运动模糊 → 视频编码。
    """
    s = max(0.0, float(strength))
    return Compose([
        RandomBoxJitter(max_frac=0.14 * s, p=0.8 * min(s, 1.0)),
        RandomRotate(max_deg=12.0 * s, p=0.6 * min(s, 1.0)),
        RandomPhotometric(p=0.8 * min(s, 1.0)),
        RandomDownscale(min_px=max(8, int(12 / max(s, 0.3))), max_px=48,
                        p=0.9 * min(s, 1.0), out_size=out_size),
        RandomMotionBlur(p=0.35 * min(s, 1.0)),
        RandomJPEG(p=0.4 * min(s, 1.0)),
        RandomNoise(p=0.4 * min(s, 1.0)),
    ])


def eval_augment(out_size: int = 96) -> Compose:
    """验证集不做任何随机变换 —— 否则验证指标每次都不一样，没法比。"""
    return Compose([])
