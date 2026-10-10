"""Regression checks against the independently refined reference court."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calib import CourtCanvasConfig, CourtProjector, court_canvas_matrix
from config import load_config
from pipeline import build_parser
from render import (
    CourtRenderConfig, FrameRenderData, PlayerRender, TrailHistory,
    court_to_pixel, draw_half_court_template, render_topdown_frame,
)


def test_refined_reference_anchors():
    projector = CourtProjector.from_json(ROOT / "configs/court_calibration_refined.json")
    # Independently clicked, trusted floor landmarks from the reference project.
    # The refined matrix may deviate slightly from clicks, but not by metres.
    anchors = [
        ((422.53, 1054.27), (5.05, 0.0)),
        ((1520.67, 1050.4), (9.95, 0.0)),
        ((666.13, 799.07), (5.05, 5.8)),
        ((967.73, 752.67), (7.5, 7.6)),
        ((971.6, 683.07), (7.5, 12.2)),
    ]
    for image, metres in anchors:
        actual = projector.project_point(*image)
        expected = court_to_pixel(*metres, CourtRenderConfig())
        assert np.linalg.norm(np.subtract(actual, expected)) < 0.04 * 60
    projector.check_frame(1920, 1080)
    with pytest.raises(ValueError, match="expects 1920x1080"):
        projector.check_frame(1280, 720)


@pytest.mark.parametrize("width,height", [(1020, 960), (500, 940), (940, 500)])
def test_equal_scale_and_shared_coordinates(width, height):
    canvas = CourtCanvasConfig(width=width, height=height)
    render = CourtRenderConfig(width=width, height=height)
    matrix = court_canvas_matrix(canvas)
    origin = np.array(court_to_pixel(0, 0, render))
    dx = np.array(court_to_pixel(1, 0, render)) - origin
    dy = np.array(court_to_pixel(0, 1, render)) - origin
    assert np.linalg.norm(dx) == pytest.approx(np.linalg.norm(dy))
    assert dx[0] > 0 and dy[1] < 0
    np.testing.assert_allclose((matrix @ [7.5, 5.8, 1])[:2], court_to_pixel(7.5, 5.8, render))


def test_default_config_and_outside_points():
    config = load_config(ROOT / "configs/default.yaml")
    projector = CourtProjector.from_json(config.calibration_json, config.canvas)
    np.testing.assert_allclose(court_to_pixel(0, 0, config.render.court), (60, 900))
    np.testing.assert_allclose(court_to_pixel(15, 14, config.render.court), (960, 60))
    # Synthesise an image point for an off-canvas court position. It must not
    # become a fake player pinned to the border or contaminate the EMA trail.
    image = np.linalg.inv(projector.homography) @ [-100, 500, 1]
    outside = projector.project_point(*(image[:2] / image[2]))
    assert outside[0] < 0
    court = draw_half_court_template(config.render.court)
    assert court.shape == (960, 1020, 3)
    trails = TrailHistory()
    trails.update(1, (500, 500))
    frame = FrameRenderData(0, 0, players=[PlayerRender(1, [0, 0, 10, 10], court_xy=outside)])
    result = render_topdown_frame(court, frame, trails, config.render.court)
    empty = render_topdown_frame(court, FrameRenderData(0, 0), TrailHistory(), config.render.court)
    np.testing.assert_array_equal(result, empty)
    assert trails.get(1) == []


def test_explicit_perdigit_cli():
    args = build_parser().parse_args(["--input", "test.mp4", "--jersey-backend", "perdigit"])
    assert args.jersey_backend == "perdigit"
