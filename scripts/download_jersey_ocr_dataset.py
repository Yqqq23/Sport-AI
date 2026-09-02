"""Download the Roboflow jersey-number dataset and prepare it for classification.

    python scripts/download_jersey_ocr_dataset.py --out data/jersey_ocr

Source: https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr
License: CC BY 4.0 (attribution required - see the ATTRIBUTION file this writes).

WHY A PREP STEP AND NOT A PLAIN DOWNLOAD
----------------------------------------
The project is of Roboflow type ``text-image-pairs``, so it exports as JSONL for
VLM fine-tuning, not as a classification folder tree:

    {"image": "....jpg", "prefix": "Read the number.", "suffix": "28"}

Three things in the raw export will silently poison a classifier if copied as-is,
all verified by counting them in v7:

``52 rows have an empty suffix``
    No label at all. Dropped.

``28 rows are labelled "00"``
    A real basketball number, but ``clean_jersey_number`` rejects it (salvaged
    rule, kept so training and inference agree). Since the pipeline can never
    OUTPUT "00", a class for it would only ever produce unreachable predictions.
    Dropped, and reported, so the decision is visible rather than silent.

``40 rows have a corrupted prefix`` (a bare UUID, or "aRead the number.")
    Their suffixes still look sane, so they are KEPT - the prefix is unused by a
    classifier. Counted and reported so the export's quality is on the record.

The label set is taken from the DATA, not assumed to be 0-99. Only 41 distinct
numbers survive cleaning; a 100-way head would leave 59 classes with no training
examples whatsoever, and an untrained logit is perfectly capable of winning an
argmax. See train/train_jersey.py.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jersey import clean_jersey_number  # noqa: E402

WORKSPACE = "roboflow-jvuqo"
PROJECT = "basketball-jersey-numbers-ocr"
VERSION = 7
EXPORT_FORMAT = "jsonl"  # the only classification-usable format this type offers

ATTRIBUTION = f"""\
This directory contains a derivative of:

  basketball-jersey-numbers-ocr (version {VERSION})
  https://universe.roboflow.com/{WORKSPACE}/{PROJECT}
  Provided by a Roboflow user
  License: CC BY 4.0  (https://creativecommons.org/licenses/by/4.0/)

CC BY 4.0 requires attribution. If any model trained on this data is published,
distributed, or demonstrated outside your team, this credit must travel with it.
"""


def download(api_key: str, out_dir: Path) -> Path:
    import requests

    meta_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/{EXPORT_FORMAT}"
    print(f"requesting export link: {meta_url}")
    response = requests.get(meta_url, params={"api_key": api_key}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    link = (payload.get("export") or {}).get("link")
    if not link:
        raise SystemExit(f"No export link in the response: {json.dumps(payload)[:400]}")

    size_mb = (payload.get("export") or {}).get("size")
    print(f"downloading ~{size_mb:.1f} MB ...")
    archive = requests.get(link, timeout=1800)
    archive.raise_for_status()

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        zf.extractall(out_dir)
    print(f"extracted to {out_dir}")
    return out_dir


def build_manifest(root: Path) -> dict:
    """Clean the JSONL splits into one manifest keyed by split.

    The manifest is the single source of truth for training: it carries the
    surviving (relative path, label) pairs plus the class list, so the training
    script never re-derives labels and cannot disagree with this step.
    """
    splits: dict[str, list[dict]] = {}
    dropped = Counter()
    bad_prefix = 0
    label_counts: Counter = Counter()

    for split in ("train", "valid", "test"):
        annotations = root / split / "annotations.jsonl"
        if not annotations.exists():
            print(f"  WARNING: missing {annotations}")
            continue
        rows: list[dict] = []
        for line in annotations.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped["unparseable_json"] += 1
                continue

            raw = str(record.get("suffix", "")).strip()
            prefix = str(record.get("prefix", ""))
            if prefix != "Read the number.":
                # Corrupted prompt text. The suffix is still usable for a
                # classifier, so keep the row but record that the export has
                # damaged entries in it.
                bad_prefix += 1

            if not raw:
                dropped["empty_label"] += 1
                continue
            label = clean_jersey_number(raw)
            if label is None:
                dropped[f"rejected:{raw}"] += 1
                continue

            image = record.get("image")
            if not image or not (root / split / image).exists():
                dropped["missing_image"] += 1
                continue

            rows.append({"image": f"{split}/{image}", "label": label})
            label_counts[label] += 1
        splits[split] = rows

    classes = sorted(label_counts, key=lambda s: int(s))
    manifest = {
        "source": {
            "workspace": WORKSPACE,
            "project": PROJECT,
            "version": VERSION,
            "license": "CC BY 4.0",
            "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}",
        },
        "classes": classes,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "class_counts": {c: label_counts[c] for c in classes},
        "dropped": dict(dropped),
        "rows_with_corrupt_prefix": bad_prefix,
        "splits": splits,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/jersey_ocr"))
    parser.add_argument(
        "--api-key-env",
        default="ROBOFLOW_API_KEY",
        help="Environment variable holding the Roboflow API key",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Re-run the cleaning step on an already-extracted directory",
    )
    args = parser.parse_args()

    out_dir = args.out
    if not args.skip_download:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"No API key in ${args.api_key_env}. "
                f"Export it first:  export {args.api_key_env}=your_key_here"
            )
        download(api_key, out_dir)
    elif not out_dir.exists():
        raise SystemExit(f"--skip-download given but {out_dir} does not exist")

    (out_dir / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")

    print("\ncleaning labels ...")
    manifest = build_manifest(out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = manifest["counts"]
    print(f"\n  kept        {sum(counts.values())} rows  {counts}")
    print(f"  classes     {len(manifest['classes'])}: {' '.join(manifest['classes'])}")
    print(f"  dropped     {sum(manifest['dropped'].values())} {manifest['dropped']}")
    print(f"  bad prefix  {manifest['rows_with_corrupt_prefix']} (kept - prefix unused)")

    rare = {c: n for c, n in manifest["class_counts"].items() if n < 10}
    if rare:
        print(f"\n  THIN CLASSES (<10 examples): {rare}")
        print("  Expect these to be unreliable no matter how long you train.")

    print(f"\n  manifest    {manifest_path}")
    print("\nNext:  python train/train_jersey.py --data", out_dir)


if __name__ == "__main__":
    main()
