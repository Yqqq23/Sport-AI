"""Convert the downloaded Roboflow dataset into this project's 5-class scheme.

    python scripts/prepare_dataset.py [--overwrite]

WHY THIS SCRIPT EXISTS
    train/basketball.yaml points at datasets/basketball, which does not exist on
    a fresh checkout. The only labelled data in the repo is dataset/, a Roboflow
    export using a completely different 10-class scheme in ALPHABETICAL order.
    Nothing else in the project maps between the two, so training cannot start
    until this has run.

    Input  dataset/                    read only, never modified
    Output train/datasets/basketball/  images/{train,val,test}, labels/...

WHY THE MAPPING IS BY NAME, NOT BY INDEX
    Roboflow assigns class indices by sorting names alphabetically. Add or
    rename one class upstream and every index after it shifts. A positional
    mapping would keep working, silently, and relabel the whole dataset - rim
    boxes trained as referees, and no error anywhere. So the mapping below is
    keyed on the NAME read from the source data.yaml, and an unknown source name
    is a hard exit rather than a skip: an unmapped class means the export
    changed and someone needs to look at it.

WHY ball-in-basket IS MERGED INTO ball (and how to reverse this later)
    ball-in-basket looked like a promising second opinion on made shots, since a
    ball visually inside the net carries depth information that src/shot.py's 2D
    geometry cannot see. It is merged into ball anyway, for three reasons:

      1. Too thin. It occurs only in the handful of frames per make where the
         ball is inside the net - a tiny fraction of ball instances. A class that
         rare trains to near-zero recall and then does nothing, silently.
      2. It fights ball for the same pixels. During the frames a shot passes
         through the rim, the same object would carry two competing labels. NMS
         and label assignment degrade both, exactly where precision matters most.
      3. The direction is backwards. README section 8 identifies MISSED BALL
         DETECTIONS DURING THE FAST DESCENT as the expected dominant error
         source - larger than every false-positive mode combined. Trading ball
         recall during the descent for a weak signal about false positives
         damages the primary error source to help a secondary one.

    THIS IS A JUDGEMENT CALL MADE WITHOUT DATA, AND IT IS REVERSIBLE. Once real
    false-positive statistics exist, count how many are the "ball passed behind
    the rim" mode. If that mode dominates, revisit: append ball-in-basket as
    class 5 (never insert it) and record its verdict as an extra column in
    shots.csv, letting src/shot.py stay the decider until the two mechanisms
    have been compared on real footage. Search this file for "ball-in-basket" to
    find the decision.

    The merge is not free either: a ball inside the net is a visually unusual
    ball, and folding it into ball slightly widens that class's appearance
    distribution. That is the intended trade.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from detect import BASKETBALL_CLASS_NAMES  # noqa: E402

# --------------------------------------------------------------------------
# The mapping. Keyed by SOURCE CLASS NAME, never by source index.
# --------------------------------------------------------------------------

#: Roboflow class name -> basketball-v2 class name.
#: A source name missing from this table stops the run.
CLASS_NAME_MAPPING: dict[str, str] = {
    "ball": "ball",
    # See the module docstring for the reasoning and how to reverse this.
    "ball-in-basket": "ball",
    # Roboflow's "number" is the number patch on the shirt: this project's jersey.
    "number": "jersey",
    # Every player state collapses to one class. The pipeline derives possession
    # and shot phase from geometry (src/shot.py), so per-state labels would split
    # scarce data across five categories nothing downstream reads.
    "player": "player",
    "player-in-possession": "player",
    "player-jump-shot": "player",
    "player-layup-dunk": "player",
    "player-shot-block": "player",
    "referee": "referee",
    "rim": "rim",
}

#: Roboflow calls it "valid"; Ultralytics and train/basketball.yaml call it "val".
SPLIT_DIRECTORY_MAPPING = {"train": "train", "valid": "val", "test": "test"}

#: Splits train/basketball.yaml actually references. test is still converted and
#: written - throwing away labelled data would be worse - but stays unreferenced
#: so it cannot be spent as a tuning set by accident.
SPLITS_IN_YAML = ("train", "val")

SOURCE_DIR_NAME = "dataset"
OUTPUT_RELATIVE = Path("train") / "datasets" / "basketball"

#: Class-scheme fields that MUST be identical between the generated data.yaml
#: and train/basketball.yaml. A mismatch here is fatal.
STRICT_YAML_FIELDS = ("nc", "names", "train", "val")


class PrepareError(RuntimeError):
    """Anything that must stop the conversion."""


# --------------------------------------------------------------------------
# Source parsing
# --------------------------------------------------------------------------


def read_source_class_names(data_yaml: Path) -> dict[int, str]:
    """Read the source's index -> name mapping from its own data.yaml.

    The indices in the label files mean whatever THIS file says they mean, so it
    is the only acceptable source for them.
    """
    if not data_yaml.is_file():
        raise PrepareError(
            f"Source data.yaml not found: {data_yaml}\n"
            "The Roboflow export must include it; without it the class indices in "
            "the label files cannot be interpreted at all."
        )

    import yaml

    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    names = data.get("names")
    if isinstance(names, dict):
        parsed = {int(key): str(value) for key, value in names.items()}
    elif isinstance(names, (list, tuple)):
        parsed = {int(index): str(value) for index, value in enumerate(names)}
    else:
        raise PrepareError(f"{data_yaml}: 'names' must be a list or a mapping")

    if not parsed:
        raise PrepareError(f"{data_yaml}: 'names' is empty")

    declared = data.get("nc")
    if declared is not None and int(declared) != len(parsed):
        raise PrepareError(
            f"{data_yaml}: nc={declared} but 'names' lists {len(parsed)} classes"
        )
    return parsed


def build_index_mapping(source_names: dict[int, str]) -> dict[int, int]:
    """Resolve source index -> target index, going through NAMES.

    Every source name must appear in CLASS_NAME_MAPPING. An unmapped name means
    the upstream export changed, and guessing is exactly the failure this whole
    script is arranged to prevent.
    """
    target_ids = {name: index for index, name in BASKETBALL_CLASS_NAMES.items()}

    unmapped = sorted(name for name in source_names.values() if name not in CLASS_NAME_MAPPING)
    if unmapped:
        raise PrepareError(
            "Source classes with no entry in CLASS_NAME_MAPPING: "
            + ", ".join(repr(name) for name in unmapped)
            + "\nThe Roboflow export has changed. Decide explicitly what each new "
            "class becomes and add it to the table at the top of this script. "
            "Skipping them silently would drop real labelled objects, and guessing "
            "would mislabel them."
        )

    mapping: dict[int, int] = {}
    for source_index, source_name in sorted(source_names.items()):
        target_name = CLASS_NAME_MAPPING[source_name]
        if target_name not in target_ids:
            raise PrepareError(
                f"CLASS_NAME_MAPPING sends {source_name!r} to {target_name!r}, which is "
                f"not a basketball-v2 class. Known: {sorted(target_ids)}"
            )
        mapping[source_index] = target_ids[target_name]
    return mapping


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def parse_label_line(line: str, where: str) -> tuple[int, list[float]]:
    parts = line.split()
    if len(parts) != 5:
        raise PrepareError(f"{where}: expected 5 values, got {len(parts)}: {line!r}")
    try:
        class_id = int(parts[0])
        values = [float(item) for item in parts[1:]]
    except ValueError as exc:
        raise PrepareError(f"{where}: non-numeric YOLO values: {line!r}") from exc
    return class_id, values


def convert_split(
    source_split_dir: Path,
    output_root: Path,
    output_split: str,
    index_mapping: dict[int, int],
    source_names: dict[int, str],
    report: dict,
) -> None:
    source_images = source_split_dir / "images"
    source_labels = source_split_dir / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise PrepareError(
            f"Split {source_split_dir.name} is missing images/ or labels/: {source_split_dir}"
        )

    image_dir = output_root / "images" / output_split
    label_dir = output_root / "labels" / output_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in source_images.iterdir() if path.is_file())
    label_paths = {path.stem: path for path in source_labels.iterdir() if path.is_file()}

    source_counts: Counter = Counter()
    target_counts: Counter = Counter()
    images_written = 0
    annotations_written = 0
    images_without_label = []
    empty_label_files = 0

    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        if label_path is None:
            # Not fatal by itself, but reported: in YOLO an image with no label
            # file is trained as pure background.
            images_without_label.append(image_path.name)
            continue

        converted_lines: list[str] = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            where = f"{label_path.name}:{line_number}"
            source_id, values = parse_label_line(line, where)

            if source_id not in index_mapping:
                raise PrepareError(
                    f"{where}: class_id {source_id} is not declared in the source "
                    f"data.yaml (which declares {sorted(source_names)}). The label "
                    "files and data.yaml disagree; the export is inconsistent."
                )
            for value in values:
                if not (0.0 <= value <= 1.0):
                    raise PrepareError(
                        f"{where}: coordinate {value} is outside 0-1. YOLO labels are "
                        "normalised; this file is not in the expected format."
                    )

            target_id = index_mapping[source_id]
            source_counts[source_names[source_id]] += 1
            target_counts[BASKETBALL_CLASS_NAMES[target_id]] += 1
            converted_lines.append(
                f"{target_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
            )

        if not converted_lines:
            empty_label_files += 1

        shutil.copy2(image_path, image_dir / image_path.name)
        (label_dir / f"{image_path.stem}.txt").write_text(
            ("\n".join(converted_lines) + "\n") if converted_lines else "", encoding="utf-8"
        )
        images_written += 1
        annotations_written += len(converted_lines)

    report[output_split] = {
        "images": images_written,
        "annotations": annotations_written,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "images_without_label": images_without_label,
        "empty_label_files": empty_label_files,
    }


def ultralytics_datasets_dir() -> Path | None:
    """Where Ultralytics resolves a RELATIVE dataset `path:` to.

    Read from its settings.json rather than by importing ultralytics, so this
    script keeps working on a machine where the package cannot be imported.
    Returns None when the setting cannot be determined.
    """
    import json
    import os

    candidates = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "Ultralytics" / "settings.json")
        candidates.append(Path(appdata) / "yolo" / "settings.json")
    candidates.append(Path.home() / ".config" / "Ultralytics" / "settings.json")
    candidates.append(Path.home() / ".config" / "yolo" / "settings.json")

    for candidate in candidates:
        try:
            if candidate.is_file():
                value = json.loads(candidate.read_text(encoding="utf-8")).get("datasets_dir")
                if value:
                    return Path(value)
        except Exception:
            continue
    return None


def write_data_yaml(output_root: Path) -> Path:
    """Write the dataset's own data.yaml, with an ABSOLUTE path.

    The path is absolute on purpose, and it is COMPUTED AT RUNTIME rather than
    written into this file, so nothing here is machine specific.

    Ultralytics resolves a relative `path:` against its configured DATASETS_DIR
    (ultralytics/data/utils.py: ``if not path.is_absolute(): path = (DATASETS_DIR
    / path).resolve()``) - NOT against the working directory and NOT against the
    YAML's own location. DATASETS_DIR defaults to ~/datasets and is often pointed
    somewhere else entirely by an unrelated project. A relative path here would
    therefore send training to a directory that has nothing to do with this repo,
    and Ultralytics would report the dataset as missing rather than explain why.
    An absolute path takes the first branch of that check and cannot be
    redirected.
    """
    lines = [
        "# GENERATED by scripts/prepare_dataset.py - do not edit by hand.",
        "# Rerun the script after changing train/basketball.yaml; it verifies the",
        "# two agree on the class scheme and refuses to finish if they do not.",
        "#",
        "# `path` is ABSOLUTE and machine specific by design: Ultralytics resolves a",
        "# relative path against its own DATASETS_DIR setting, not against this file",
        "# or the working directory. Regenerate this file on each machine.",
        f'path: "{str(output_root.resolve()).replace(chr(92), "/")}"',
        "train: images/train",
        "val: images/val",
        # test is converted and present on disk, but deliberately NOT referenced
        # here - see SPLITS_IN_YAML - so it cannot be spent as a tuning set.
        f"nc: {len(BASKETBALL_CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(
        f"  {index}: {BASKETBALL_CLASS_NAMES[index]}"
        for index in range(len(BASKETBALL_CLASS_NAMES))
    )
    path = output_root / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def compare_with_basketball_yaml(generated: Path, canonical: Path) -> tuple[list[str], list[str]]:
    """Field-by-field diff of the generated data.yaml against basketball.yaml.

    Returns (fatal_differences, expected_differences).

    nc, names, train and val must be identical - those are the class scheme and
    the split layout, and a disagreement there means training would learn
    something other than what src/detect.py expects at inference.

    `path` is EXPECTED to differ and is reported separately rather than treated
    as an error: the generated file carries an absolute path so Ultralytics can
    actually find the data (see write_data_yaml), while basketball.yaml keeps a
    relative one so it stays machine independent in version control.
    """
    import yaml

    with generated.open("r", encoding="utf-8") as handle:
        produced = yaml.safe_load(handle) or {}
    with canonical.open("r", encoding="utf-8") as handle:
        expected = yaml.safe_load(handle) or {}

    def normalise(names):
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {int(index): str(value) for index, value in enumerate(names)}
        return {}

    fatal: list[str] = []
    for field in STRICT_YAML_FIELDS:
        if field == "names":
            produced_names = normalise(produced.get("names"))
            expected_names = normalise(expected.get("names"))
            for index in sorted(set(produced_names) | set(expected_names)):
                if produced_names.get(index) != expected_names.get(index):
                    fatal.append(
                        f"names[{index}]: generated={produced_names.get(index)!r} but "
                        f"basketball.yaml={expected_names.get(index)!r}"
                    )
            continue
        if produced.get(field) != expected.get(field):
            fatal.append(
                f"{field}: generated={produced.get(field)!r} but "
                f"basketball.yaml={expected.get(field)!r}"
            )

    informational: list[str] = []
    if produced.get("path") != expected.get("path"):
        informational.append(
            f"path: generated={produced.get('path')!r} (absolute, so Ultralytics "
            f"resolves it directly) vs basketball.yaml={expected.get('path')!r} "
            "(relative, kept machine independent). This difference is intended."
        )
    return fatal, informational


def resolve_training_paths(canonical_yaml: Path, generated_yaml: Path) -> dict:
    """Work out where each YAML would actually send training, and whether it exists.

    This is the check that catches the failure the README warns about but
    mis-explains: a relative `path:` does NOT resolve against the working
    directory, it resolves against Ultralytics' DATASETS_DIR setting.
    """
    import yaml

    datasets_dir = ultralytics_datasets_dir()

    def resolve(yaml_path: Path) -> dict:
        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw = str(data.get("path", ""))
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            how = "absolute path, used as-is"
        elif datasets_dir is not None:
            resolved = (datasets_dir / candidate).resolve()
            how = f"relative, resolved against DATASETS_DIR ({datasets_dir})"
        else:
            resolved = None
            how = "relative, and DATASETS_DIR could not be determined"
        return {
            "yaml": yaml_path,
            "raw_path": raw,
            "resolved": resolved,
            "how": how,
            "train_exists": bool(
                resolved and (resolved / str(data.get("train", "images/train"))).is_dir()
            ),
            "val_exists": bool(
                resolved and (resolved / str(data.get("val", "images/val"))).is_dir()
            ),
        }

    return {
        "datasets_dir": datasets_dir,
        "canonical": resolve(canonical_yaml),
        "generated": resolve(generated_yaml),
    }


def verify_pairing(output_root: Path, split: str) -> list[str]:
    """Every image must have a label file and vice versa."""
    problems: list[str] = []
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_stems = {path.stem for path in image_dir.iterdir() if path.is_file()}
    label_stems = {path.stem for path in label_dir.iterdir() if path.is_file()}
    for stem in sorted(image_stems - label_stems):
        problems.append(f"{split}: image without label: {stem}")
    for stem in sorted(label_stems - image_stems):
        problems.append(f"{split}: label without image: {stem}")
    return problems


def verify_labels(output_root: Path, split: str) -> tuple[list[str], Counter]:
    """Re-read what was written: class ids in range, coordinates in 0-1."""
    problems: list[str] = []
    counts: Counter = Counter()
    label_dir = output_root / "labels" / split
    for label_path in sorted(label_dir.iterdir()):
        if not label_path.is_file():
            continue
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            where = f"{split}/{label_path.name}:{line_number}"
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{where}: expected 5 values, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                values = [float(item) for item in parts[1:]]
            except ValueError:
                problems.append(f"{where}: non-numeric values")
                continue
            if class_id not in BASKETBALL_CLASS_NAMES:
                problems.append(
                    f"{where}: class_id {class_id} outside "
                    f"0-{len(BASKETBALL_CLASS_NAMES) - 1}"
                )
                continue
            for value in values:
                if not (0.0 <= value <= 1.0):
                    problems.append(f"{where}: coordinate {value} outside 0-1")
            counts[class_id] += 1
    return problems, counts


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_report(
    source_names: dict[int, str],
    index_mapping: dict[int, int],
    report: dict,
    written_counts: dict[str, Counter],
    yaml_differences: list[str],
    yaml_informational: list[str],
    path_resolution: dict,
    pairing_problems: list[str],
    label_problems: list[str],
    output_root: Path,
) -> None:
    line = "=" * 74
    print()
    print(line)
    print("DATASET PREPARATION REPORT")
    print(line)

    print()
    print("1. CLASS MAPPING (resolved by name, not by index)")
    print("-" * 74)
    print(f"{'src id':>7}  {'source name':<24}{'->':^6}{'dst id':>7}  {'target name':<12}")
    for source_index in sorted(source_names):
        source_name = source_names[source_index]
        target_index = index_mapping[source_index]
        print(
            f"{source_index:>7}  {source_name:<24}{'->':^6}"
            f"{target_index:>7}  {BASKETBALL_CLASS_NAMES[target_index]:<12}"
        )

    total_source: Counter = Counter()
    total_target: Counter = Counter()
    for split_report in report.values():
        total_source.update(split_report["source_counts"])
        total_target.update(split_report["target_counts"])

    print()
    print("2. BOX COUNTS BEFORE AND AFTER MAPPING")
    print("-" * 74)
    print(f"{'source class':<24}{'boxes':>10}   {'target class':<12}{'boxes':>10}")
    for source_name in sorted(total_source):
        target_name = CLASS_NAME_MAPPING[source_name]
        print(
            f"{source_name:<24}{total_source[source_name]:>10}   "
            f"{target_name:<12}{'':>10}"
        )
    print("-" * 74)
    for target_name in [BASKETBALL_CLASS_NAMES[i] for i in range(len(BASKETBALL_CLASS_NAMES))]:
        print(f"{'':<24}{'':>10}   {target_name:<12}{total_target.get(target_name, 0):>10}")
    print("-" * 74)
    source_total = sum(total_source.values())
    target_total = sum(total_target.values())
    print(f"{'TOTAL':<24}{source_total:>10}   {'TOTAL':<12}{target_total:>10}")
    conserved = source_total == target_total
    print(
        f"  boxes conserved through the mapping: {'YES' if conserved else 'NO - BOXES LOST'}"
    )

    print()
    print("3. PER SPLIT: IMAGE / ANNOTATION PAIRING")
    print("-" * 74)
    print(f"{'split':<8}{'images':>10}{'annotations':>14}{'empty labels':>14}{'unpaired':>12}")
    for split in ("train", "val", "test"):
        if split not in report:
            continue
        entry = report[split]
        print(
            f"{split:<8}{entry['images']:>10}{entry['annotations']:>14}"
            f"{entry['empty_label_files']:>14}{len(entry['images_without_label']):>12}"
        )
    if pairing_problems:
        print("  PAIRING PROBLEMS:")
        for problem in pairing_problems[:20]:
            print(f"    - {problem}")
    else:
        print("  every image has exactly one label file, and vice versa: OK")

    print()
    print("4. LABEL VALIDITY (re-read from what was written)")
    print("-" * 74)
    if label_problems:
        print(f"  {len(label_problems)} PROBLEM(S):")
        for problem in label_problems[:20]:
            print(f"    - {problem}")
    else:
        print(f"  class ids all within 0-{len(BASKETBALL_CLASS_NAMES) - 1}: OK")
        print("  all coordinates within 0-1: OK")

    print()
    print("5. PER-CLASS PRESENCE IN train AND val")
    print("-" * 74)
    print(f"{'class':<12}{'id':>4}{'train':>10}{'val':>10}{'test':>10}   status")
    missing_anywhere = []
    for class_id in range(len(BASKETBALL_CLASS_NAMES)):
        name = BASKETBALL_CLASS_NAMES[class_id]
        train_count = written_counts.get("train", Counter()).get(class_id, 0)
        val_count = written_counts.get("val", Counter()).get(class_id, 0)
        test_count = written_counts.get("test", Counter()).get(class_id, 0)
        if train_count and val_count:
            status = "ok"
        elif not train_count and not val_count:
            status = "ABSENT FROM BOTH - cannot be trained or measured"
            missing_anywhere.append(name)
        elif not val_count:
            status = "no val labels - trainable, but mAP will not be reported"
            missing_anywhere.append(name)
        else:
            status = "no train labels - will not be learned"
            missing_anywhere.append(name)
        print(f"{name:<12}{class_id:>4}{train_count:>10}{val_count:>10}{test_count:>10}   {status}")

    print()
    print("6. GENERATED data.yaml vs train/basketball.yaml")
    print("-" * 74)
    if yaml_differences:
        print("  MISMATCH on a field that must be identical:")
        for difference in yaml_differences:
            print(f"    - {difference}")
    else:
        print(f"  {', '.join(STRICT_YAML_FIELDS)} all identical: OK")
    for note in yaml_informational:
        print(f"  note: {note}")

    print()
    print("7. WHERE TRAINING WILL ACTUALLY LOOK")
    print("-" * 74)
    datasets_dir = path_resolution["datasets_dir"]
    print(f"  Ultralytics DATASETS_DIR: {datasets_dir if datasets_dir else 'UNKNOWN'}")
    print(
        "  A relative `path:` resolves against THAT, not against the working\n"
        "  directory and not against the YAML's own location."
    )
    for label, entry in (
        ("generated data.yaml", path_resolution["generated"]),
        ("train/basketball.yaml", path_resolution["canonical"]),
    ):
        status = "FINDS THE DATA" if (entry["train_exists"] and entry["val_exists"]) else "DOES NOT FIND IT"
        print()
        print(f"  {label}")
        print(f"    path:     {entry['raw_path']}")
        print(f"    {entry['how']}")
        print(f"    resolves: {entry['resolved']}")
        print(f"    -> {status}")

    print()
    print(line)
    print(f"Output: {output_root}")
    if missing_anywhere:
        print()
        print("WARNING: these classes are not usable as they stand:")
        for name in missing_anywhere:
            print(f"  - {name}")
        print(
            "  The public dataset does not cover them the way this project needs.\n"
            "  Fill the gaps with your own footage (docs/annotation_guide.md) before\n"
            "  reading any mAP for them as meaningful."
        )
    print(line)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the Roboflow dataset/ into the 5-class train/datasets/basketball/"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / SOURCE_DIR_NAME,
        help="Roboflow export root (read only). Default: <repo>/dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / OUTPUT_RELATIVE,
        help="Output dataset root. Default: <repo>/train/datasets/basketball",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory first. Required to overwrite.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_root = Path(args.source)
    output_root = Path(args.output)
    canonical_yaml = REPO_ROOT / "train" / "basketball.yaml"

    if not source_root.is_dir():
        print(f"ERROR: source dataset not found: {source_root}", file=sys.stderr)
        return 1
    if not canonical_yaml.is_file():
        print(f"ERROR: not found: {canonical_yaml}", file=sys.stderr)
        return 1

    if output_root.exists():
        if not args.overwrite:
            print(
                f"ERROR: output directory already exists: {output_root}\n"
                "Refusing to write into it. A partial overlay of an old conversion on a\n"
                "new one produces a dataset that is neither, with no error to show for it.\n"
                "Pass --overwrite to delete and rebuild it.",
                file=sys.stderr,
            )
            return 1
        print(f"--overwrite: deleting {output_root}")
        shutil.rmtree(output_root)

    try:
        source_names = read_source_class_names(source_root / "data.yaml")
        index_mapping = build_index_mapping(source_names)

        print(f"Source: {source_root}  ({len(source_names)} classes)")
        print(f"Output: {output_root}  ({len(BASKETBALL_CLASS_NAMES)} classes)")

        report: dict = {}
        for source_split, output_split in SPLIT_DIRECTORY_MAPPING.items():
            source_split_dir = source_root / source_split
            if not source_split_dir.is_dir():
                print(f"  split '{source_split}' not present in the source, skipping")
                continue
            print(f"  converting {source_split} -> {output_split} ...")
            convert_split(
                source_split_dir, output_root, output_split, index_mapping, source_names, report
            )

        if not report:
            raise PrepareError(f"No usable splits found under {source_root}")

        generated_yaml = write_data_yaml(output_root)
        yaml_differences, yaml_informational = compare_with_basketball_yaml(
            generated_yaml, canonical_yaml
        )
        path_resolution = resolve_training_paths(canonical_yaml, generated_yaml)

        pairing_problems: list[str] = []
        label_problems: list[str] = []
        written_counts: dict[str, Counter] = {}
        for split in report:
            pairing_problems.extend(verify_pairing(output_root, split))
            problems, counts = verify_labels(output_root, split)
            label_problems.extend(problems)
            written_counts[split] = counts

    except PrepareError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print_report(
        source_names=source_names,
        index_mapping=index_mapping,
        report=report,
        written_counts=written_counts,
        yaml_differences=yaml_differences,
        yaml_informational=yaml_informational,
        path_resolution=path_resolution,
        pairing_problems=pairing_problems,
        label_problems=label_problems,
        output_root=output_root,
    )

    # Anything that would make training read the wrong thing is fatal. The
    # per-class coverage warning is not: an incomplete public dataset is a real
    # situation, and the report says so loudly enough.
    if yaml_differences:
        print(
            "\nFAILED: the generated data.yaml does not match train/basketball.yaml.\n"
            "Training would read a class scheme that disagrees with src/detect.py.",
            file=sys.stderr,
        )
        return 1
    if pairing_problems or label_problems:
        print("\nFAILED: the written dataset did not pass verification.", file=sys.stderr)
        return 1

    generated_entry = path_resolution["generated"]
    if not (generated_entry["train_exists"] and generated_entry["val_exists"]):
        print(
            "\nFAILED: the generated data.yaml does not resolve to the data just "
            f"written.\n  resolves to: {generated_entry['resolved']}",
            file=sys.stderr,
        )
        return 1

    relative_path = str(output_root.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    print("\nOK. Train with:")
    print(f"    python train/train.py train --data {relative_path}/data.yaml")
    print()
    print("  Use THAT yaml, not train/basketball.yaml. basketball.yaml keeps a")
    print("  relative `path:` so it stays machine independent in version control,")
    print("  and Ultralytics would resolve it against its own DATASETS_DIR rather")
    print("  than against this repository - see section 7 above. basketball.yaml")
    print("  remains the source of truth for the CLASS SCHEME, which section 6")
    print("  just verified the generated file matches exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
