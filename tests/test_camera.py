import numpy as np
import pytest

from overstride.pose.camera import project_points


def make_camera(f=1000.0, cx=960.0, cy=540.0):
    intrinsic = np.array([
        [f, 0.0, cx, 0.0],
        [0.0, f, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    extrinsic = np.eye(4)  # camera coincides with world origin, looking down +Z
    return intrinsic, extrinsic


def test_point_on_optical_axis_projects_to_principal_point():
    intrinsic, extrinsic = make_camera()
    point = np.array([0.0, 0.0, 5000.0])
    pixel = project_points(point, intrinsic, extrinsic)
    np.testing.assert_allclose(pixel, [960.0, 540.0])


def test_offset_point_projects_via_pinhole_formula():
    intrinsic, extrinsic = make_camera(f=1000.0, cx=960.0, cy=540.0)
    point = np.array([500.0, 0.0, 5000.0])
    pixel = project_points(point, intrinsic, extrinsic)
    expected_u = 1000.0 * 500.0 / 5000.0 + 960.0
    np.testing.assert_allclose(pixel, [expected_u, 540.0])


def test_projects_batch_of_points():
    intrinsic, extrinsic = make_camera()
    points = np.array([
        [0.0, 0.0, 5000.0],
        [500.0, 0.0, 5000.0],
        [0.0, 250.0, 2500.0],
    ])
    pixels = project_points(points, intrinsic, extrinsic)
    assert pixels.shape == (3, 2)
    np.testing.assert_allclose(pixels[0], [960.0, 540.0])
    np.testing.assert_allclose(pixels[2], [960.0, 1000.0 * 250.0 / 2500.0 + 540.0])


def test_extrinsic_translation_is_applied():
    intrinsic, extrinsic = make_camera()
    # extrinsic translation t_x=-100 => camera sits at world x=+100, so a point
    # at world x=0 lands at camera-space x=-100 (to the camera's left)
    extrinsic[0, 3] = -100.0
    point = np.array([0.0, 0.0, 5000.0])
    pixel = project_points(point, intrinsic, extrinsic)
    expected_u = 1000.0 * -100.0 / 5000.0 + 960.0
    assert pixel[0] == pytest.approx(expected_u)
