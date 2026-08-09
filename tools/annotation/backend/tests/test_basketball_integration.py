"""Tests for the binding between this tool and the basketball-v2 project.

These cover the failures that are otherwise SILENT: a class scheme that drifts
from train/basketball.yaml, a jersey number normalized differently from the
inference pipeline, and a train/val split that stops protecting against identity
leakage. None of them raise on their own; each produces a dataset that looks
fine and trains the wrong thing.
"""

import json
from pathlib import Path

import pytest

from app.services import basketball_classes as bc
from app.services.dataset_versioning import IDENTITY_CLASS_IDS, UNASSIGNED_TRACK_ID


# --------------------------------------------------------------------------
# Class scheme
# --------------------------------------------------------------------------


def test_class_scheme_matches_basketball_yaml_exactly():
    """The tool's scheme is READ from the project, never restated."""
    assert bc.BASKETBALL_CLASS_NAMES == {
        0: "player",
        1: "ball",
        2: "rim",
        3: "jersey",
        4: "referee",
    }
    assert bc.CLASS_COUNT == 5
    assert bc.CLASS_NAMES_IN_INDEX_ORDER == ["player", "ball", "rim", "jersey", "referee"]


def test_class_order_is_by_index_not_alphabetical():
    """Regression guard for the ordering bug this arrangement exists to prevent.

    Sorting the names would produce ball/jersey/player/referee/rim and silently
    relabel every annotation ever made.
    """
    assert bc.CLASS_NAMES_IN_INDEX_ORDER != sorted(bc.CLASS_NAMES_IN_INDEX_ORDER)
    assert bc.CLASS_PLAYER == 0
    assert bc.CLASS_JERSEY == 3
    assert bc.CLASS_REFEREE == 4


def test_referee_is_appended_last():
    """0-3 must keep their meaning; new classes go on the end."""
    for index, name in enumerate(["player", "ball", "rim", "jersey"]):
        assert bc.BASKETBALL_CLASS_NAMES[index] == name
    assert max(bc.BASKETBALL_CLASS_NAMES) == bc.CLASS_REFEREE


def test_generated_names_block_round_trips(tmp_path):
    generated = tmp_path / "dataset.yaml"
    generated.write_text(
        "\n".join(['path: "somewhere/else"', "train: images/train", "val: images/val"]
                  + bc.render_names_block() + [""]),
        encoding="utf-8",
    )
    # path/train/val differ from basketball.yaml on purpose and are not compared.
    assert bc.compare_with_basketball_yaml(generated) == []
    bc.assert_matches_basketball_yaml(generated)


@pytest.mark.parametrize(
    "names_block, expected_fragment",
    [
        (["nc: 4", "names:", "  0: player", "  1: ball", "  2: rim", "  3: jersey"], "nc"),
        (
            ["nc: 5", "names:", "  0: ball", "  1: jersey", "  2: player",
             "  3: referee", "  4: rim"],
            "names[0]",
        ),
        (
            ["nc: 5", "names:", "  0: player", "  1: ball", "  2: rim",
             "  3: jersey", "  4: umpire"],
            "names[4]",
        ),
    ],
    ids=["dropped-referee", "alphabetical-order", "renamed-class"],
)
def test_mismatched_dataset_yaml_is_rejected(tmp_path, names_block, expected_fragment):
    """A wrong dataset.yaml must raise, never pass quietly."""
    generated = tmp_path / "dataset.yaml"
    generated.write_text("\n".join(['path: "x"'] + names_block + [""]), encoding="utf-8")

    differences = bc.compare_with_basketball_yaml(generated)
    assert differences, "a mismatched dataset.yaml must be reported"
    assert any(expected_fragment in item for item in differences)

    with pytest.raises(bc.BasketballIntegrationError):
        bc.assert_matches_basketball_yaml(generated)


def test_missing_dataset_yaml_is_rejected(tmp_path):
    with pytest.raises(bc.BasketballIntegrationError):
        bc.assert_matches_basketball_yaml(tmp_path / "does_not_exist.yaml")


# --------------------------------------------------------------------------
# Jersey number normalization
# --------------------------------------------------------------------------


def test_clean_jersey_number_is_the_projects_own_function():
    """Not a copy. The identity check is the whole point.

    Two implementations would drift, and every OCR accuracy figure measured
    against this ground truth would be wrong without any error appearing.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("jersey_probe", bc.JERSEY_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["jersey_probe"] = module
    spec.loader.exec_module(module)
    try:
        assert bc.clean_jersey_number.__code__ == module.clean_jersey_number.__code__
    finally:
        sys.modules.pop("jersey_probe", None)


@pytest.mark.parametrize(
    "raw, expected",
    [("01", "1"), ("1", "1"), ("07", "7"), ("0", "0"), ("23", "23"), ("#7", "7")],
)
def test_leading_zeros_collapse(raw, expected):
    assert bc.normalize_jersey_number(raw) == expected


@pytest.mark.parametrize("raw", ["00", "123", "abc"])
def test_illegal_numbers_raise_rather_than_silently_dropping(raw):
    with pytest.raises(ValueError):
        bc.normalize_jersey_number(raw)


def test_blank_number_is_none_not_an_error():
    assert bc.normalize_jersey_number("") is None
    assert bc.normalize_jersey_number(None) is None
    assert bc.normalize_jersey_number("   ") is None


# --------------------------------------------------------------------------
# class_id validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("class_id", [0, 1, 2, 3, 4])
def test_valid_class_ids_accepted(class_id):
    assert bc.validate_class_id(class_id) == class_id


@pytest.mark.parametrize("class_id", [-1, 5, 99])
def test_out_of_range_class_ids_rejected(class_id):
    with pytest.raises(ValueError):
        bc.validate_class_id(class_id)


# --------------------------------------------------------------------------
# Split integrity after the move to 5 classes
# --------------------------------------------------------------------------


def test_only_person_classes_group_the_split():
    """Ball and rim must not join frames together.

    They have no identity, so they are stored with a placeholder track id. If
    they took part in the grouping graph, every frame containing a ball would be
    joined to every other one through that single shared node and the whole
    video would collapse into one component - the 'track' strategy would quietly
    become the 'video' strategy.
    """
    assert IDENTITY_CLASS_IDS == frozenset({bc.CLASS_PLAYER, bc.CLASS_JERSEY})
    assert bc.CLASS_BALL not in IDENTITY_CLASS_IDS
    assert bc.CLASS_RIM not in IDENTITY_CLASS_IDS
    assert bc.CLASS_REFEREE not in IDENTITY_CLASS_IDS
    assert UNASSIGNED_TRACK_ID == 0


def test_metadata_class_mapping_is_index_keyed():
    assert bc.class_mapping_for_metadata() == {
        "0": "player",
        "1": "ball",
        "2": "rim",
        "3": "jersey",
        "4": "referee",
    }


def test_classes_txt_is_newline_separated_in_index_order():
    assert bc.classes_txt_content() == "player\nball\nrim\njersey\nreferee\n"


def test_describe_reports_the_source_of_truth():
    described = bc.describe()
    assert Path(described["source_of_truth"]).name == "basketball.yaml"
    assert described["class_count"] == 5
    assert described["number_bearing_class_name"] == "jersey"
    assert [item["name"] for item in described["classes"]] == bc.CLASS_NAMES_IN_INDEX_ORDER
