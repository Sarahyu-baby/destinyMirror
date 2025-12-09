import cv2
import numpy as np
import pytest
from face_analyzer import FaceAnalyzer


def test_face_analyzer_init():
    analyzer = FaceAnalyzer()
    assert analyzer.face_mesh is not None
    assert analyzer.landmarks_np is None
    assert analyzer.custom_points is None


def test_process_image_none():
    analyzer = FaceAnalyzer()
    result = analyzer.process_image(None)
    assert result is None


def test_process_image_no_face():
    analyzer = FaceAnalyzer()
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    result = analyzer.process_image(img)

    assert result is None
    assert analyzer.landmarks_np is None
    assert analyzer.custom_points is None


@pytest.mark.parametrize("num_points", [468, 500])   # pytest used here
def test_internal_measure_face_basic(num_points):
    analyzer = FaceAnalyzer()
    fake_lms = np.random.rand(num_points, 2)

    # Only run when correct MediaPipe landmark size
    if num_points == 468:
        measurements, custom_points = analyzer._measure_face(fake_lms)

        assert isinstance(measurements, dict)
        assert isinstance(custom_points, dict)

        expected_keys = [
            "face_width", "face_length", "forehead_height",
            "midface_height", "lower_face_height", "eye_distance",
            "nose_width", "nose_length", "mouth_width", "chin_width",
            "upper_lip_h", "lower_lip_h", "jaw_angle_left", "jaw_angle_right"
        ]

        for key in expected_keys:
            assert key in measurements
            assert isinstance(measurements[key], float)


def test_ratio_calculation():
    analyzer = FaceAnalyzer()

    fake_m = {
        "face_width": 100,
        "face_length": 200,
        "forehead_height": 50,
        "midface_height": 60,
        "lower_face_height": 90,
        "eye_distance": 30,
        "nose_width": 20,
        "nose_length": 40,
        "mouth_width": 50,
        "chin_width": 70,
        "upper_lip_h": 10,
        "lower_lip_h": 12,
        "jaw_angle_left": 120,
        "jaw_angle_right": 118,
    }

    ratios = analyzer._calculate_ratios(fake_m)

    expected_ratio_keys = [
        "face_lw_ratio",
        "forehead_ratio",
        "midface_ratio",
        "lowerface_ratio",
        "eye_distance_ratio",
        "nose_ratio",
        "mouth_chin_ratio",
        "jaw_angle",
        "upper_lip_ratio",
        "lower_lip_ratio",
    ]

    for key in expected_ratio_keys:
        assert key in ratios
        assert isinstance(ratios[key], float)