"""训练球衣号分类器，产出 basketball-v2 ``LocalJerseyClassifier`` 能直接加载的权重。

    python -m jersey_train.cli mine  --video game.mp4 --out data/mined
    python -m jersey_train.cli label --src data/mined --out data/labelled
    python -m jersey_train.cli train --data data/labelled --out train/jersey_classifier.pt
    python -m jersey_train.cli verify --ckpt train/jersey_classifier.pt --numbers 0,6,12

``contract.py`` 是这个包的地基：checkpoint 的格式契约（resnet18 + Linear(512,N)、
自描述的 classes/input_size/mean/std）以及照 jersey.py 原样加载一遍的校验函数。
格式对不上，练得再好也是废的。
"""

__all__ = ["contract", "augment", "data", "train", "mine", "label", "download"]
