"""
Tests for measurement pipeline modules.
"""

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.measurement.core import (
    KEYPOINT_INDICES,
    ATCScorer,
    BodyMeasurements,
    GroundEstimator,
    MeasurementPipeline,
    ReferenceMarkerDetector,
)


class TestReferenceMarkerDetector(unittest.TestCase):
    """Test cases for reference marker detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = ReferenceMarkerDetector()

    def test_detect_aruco_marker_no_marker(self):
        """Test ArUco detection when no marker is present."""
        # Create image without ArUco marker
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        scale = self.detector.detect_aruco_marker(image)
        self.assertIsNone(scale)

    def test_detect_aruco_marker_with_marker(self):
        """Test ArUco detection with a marker present."""
        # Create image with ArUco marker
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add a simple ArUco-like pattern
        marker_size = 100
        marker_x, marker_y = 50, 50

        # Create a simple square pattern (simplified ArUco)
        image[marker_y : marker_y + marker_size, marker_x : marker_x + marker_size] = 0
        image[marker_y + 10 : marker_y + 90, marker_x + 10 : marker_x + 90] = 255

        # Mock the ArUco detection to return expected results
        with patch("cv2.aruco.detectMarkers") as mock_detect:
            # Mock corners for a 100px marker
            mock_corners = [[[[50, 50], [150, 50], [150, 150], [50, 150]]]]
            mock_ids = [[0]]
            mock_detect.return_value = (mock_corners, mock_ids, None)

            scale = self.detector.detect_aruco_marker(image)

            # Should return scale factor (50mm / 100px = 0.5 mm/px)
            self.assertIsNotNone(scale)
            self.assertAlmostEqual(scale, 0.5, places=2)

    def test_detect_a4_sheet_no_sheet(self):
        """Test A4 sheet detection when no sheet is present."""
        # Create image without A4 sheet
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        scale = self.detector.detect_a4_sheet(image)
        self.assertIsNone(scale)

    def test_detect_a4_sheet_with_sheet(self):
        """Test A4 sheet detection with a sheet present."""
        # Create image with A4-like rectangle
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add A4-like rectangle (210:297 aspect ratio)
        sheet_width, sheet_height = 100, 141  # Approximate A4 ratio
        sheet_x, sheet_y = 50, 50

        cv2.rectangle(
            image,
            (sheet_x, sheet_y),
            (sheet_x + sheet_width, sheet_y + sheet_height),
            (255, 255, 255),
            -1,
        )

        # Mock contour detection
        with patch("cv2.findContours") as mock_contours:
            # Create mock contour
            contour = np.array(
                [
                    [[sheet_x, sheet_y]],
                    [[sheet_x + sheet_width, sheet_y]],
                    [[sheet_x + sheet_width, sheet_y + sheet_height]],
                    [[sheet_x, sheet_y + sheet_height]],
                ],
                dtype=np.int32,
            )
            mock_contours.return_value = ([contour], None)

            scale = self.detector.detect_a4_sheet(image)

            # Should return scale factor (210mm / 100px = 2.1 mm/px)
            self.assertIsNotNone(scale)
            self.assertAlmostEqual(scale, 2.1, places=1)

    def test_detect_reference_marker_fallback(self):
        """Test reference marker detection with fallback."""
        # Create image without any markers
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        scale = self.detector.detect_reference_marker(image)

        # Should return default scale of 1.0
        self.assertEqual(scale, 1.0)


class TestGroundEstimator(unittest.TestCase):
    """Test cases for ground line estimation."""

    def test_estimate_ground_line_simple_mask(self):
        """Test ground line estimation with a simple mask."""
        # Create a simple mask with a rectangular shape
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        ground_y = GroundEstimator.estimate_ground_line(mask)

        # Ground line should be at y=79 (bottom of the rectangle)
        self.assertAlmostEqual(ground_y, 79.0, places=1)

    def test_estimate_ground_line_empty_mask(self):
        """Test ground line estimation with empty mask."""
        # Create empty mask
        mask = np.zeros((100, 100), dtype=np.uint8)

        ground_y = GroundEstimator.estimate_ground_line(mask)

        # Should return bottom of image
        self.assertEqual(ground_y, 99.0)

    def test_estimate_ground_line_irregular_shape(self):
        """Test ground line estimation with irregular shape."""
        # Create mask with irregular bottom edge
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        # Add some irregularity
        mask[75:80, 30:50] = 0  # Create a gap

        ground_y = GroundEstimator.estimate_ground_line(mask)

        # Should be around the median of the bottom points
        self.assertGreater(ground_y, 70)
        self.assertLess(ground_y, 80)


class TestBodyMeasurements(unittest.TestCase):
    """Test cases for body measurements calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.scale_mm_per_px = 1.0  # 1 mm per pixel
        self.measurements = BodyMeasurements(self.scale_mm_per_px)

        # Create test keypoints (14 keypoints with x, y, visibility)
        self.test_keypoints = [
            [100, 200, 2],  # muzzle_tip
            [100, 180, 2],  # forehead_top
            [100, 150, 2],  # withers
            [100, 120, 2],  # chest_center
            [80, 120, 2],  # left_chest_side
            [120, 120, 2],  # right_chest_side
            [90, 100, 2],  # hip_left
            [110, 100, 2],  # hip_right
            [100, 80, 2],  # tail_base
            [100, 90, 2],  # rump_top
            [80, 140, 2],  # left_fore_hoof
            [120, 140, 2],  # right_fore_hoof
            [90, 120, 2],  # left_rear_hoof
            [110, 120, 2],  # right_rear_hoof
        ]

        # Create test mask
        self.test_mask = np.zeros((200, 200), dtype=np.uint8)
        self.test_mask[80:160, 80:120] = 255  # Simple rectangular mask

    def test_calculate_body_length(self):
        """Test body length calculation."""
        length = self.measurements.calculate_body_length(self.test_keypoints)

        # Distance from muzzle_tip (100, 200) to tail_base (100, 80) = 120 pixels
        expected_length = 120 * self.scale_mm_per_px / 10.0  # Convert to cm
        self.assertAlmostEqual(length, expected_length, places=1)

    def test_calculate_height_at_withers(self):
        """Test height at withers calculation."""
        height = self.measurements.calculate_height_at_withers(
            self.test_keypoints, self.test_mask
        )

        # Withers at (100, 150), ground estimated at y=159, height = 9 pixels
        expected_height = 9 * self.scale_mm_per_px / 10.0  # Convert to cm
        self.assertAlmostEqual(height, expected_height, places=1)

    def test_calculate_chest_width(self):
        """Test chest width calculation."""
        width = self.measurements.calculate_chest_width(
            self.test_keypoints, self.test_mask
        )

        # Distance from left_chest_side (80, 120) to right_chest_side (120, 120) = 40 pixels
        expected_width = 40 * self.scale_mm_per_px / 10.0  # Convert to cm
        self.assertAlmostEqual(width, expected_width, places=1)

    def test_calculate_rump_angle(self):
        """Test rump angle calculation."""
        angle = self.measurements.calculate_rump_angle(self.test_keypoints)

        # Should return a valid angle in degrees
        self.assertGreater(angle, 0)
        self.assertLess(angle, 180)

    def test_calculate_all_measurements(self):
        """Test calculation of all measurements."""
        measurements = self.measurements.calculate_all_measurements(
            self.test_keypoints, self.test_mask
        )

        # Check that all measurements are present
        required_keys = [
            "body_length_cm",
            "height_withers_cm",
            "chest_width_cm",
            "rump_angle_deg",
        ]
        for key in required_keys:
            self.assertIn(key, measurements)
            self.assertIsInstance(measurements[key], (int, float))
            self.assertGreaterEqual(measurements[key], 0)

    def test_missing_keypoints_handling(self):
        """Test handling of missing keypoints."""
        # Create keypoints with some missing (visibility = 0)
        incomplete_keypoints = self.test_keypoints.copy()
        incomplete_keypoints[0][2] = 0  # Make muzzle_tip invisible
        incomplete_keypoints[8][2] = 0  # Make tail_base invisible

        measurements = self.measurements.calculate_all_measurements(
            incomplete_keypoints, self.test_mask
        )

        # Should still return measurements, but some may be 0
        self.assertIn("body_length_cm", measurements)
        self.assertIn("height_withers_cm", measurements)
        self.assertIn("chest_width_cm", measurements)
        self.assertIn("rump_angle_deg", measurements)


class TestATCScorer(unittest.TestCase):
    """Test cases for ATC scoring system."""

    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ATCScorer()

    def test_score_body_length(self):
        """Test body length scoring."""
        # Test different score ranges
        self.assertEqual(self.scorer.score_body_length(200), 4)  # Excellent
        self.assertEqual(self.scorer.score_body_length(170), 3)  # Good
        self.assertEqual(self.scorer.score_body_length(150), 2)  # Fair
        self.assertEqual(self.scorer.score_body_length(120), 1)  # Poor

    def test_score_height(self):
        """Test height scoring."""
        # Test different score ranges
        self.assertEqual(self.scorer.score_height(150), 4)  # Excellent
        self.assertEqual(self.scorer.score_height(130), 3)  # Good
        self.assertEqual(self.scorer.score_height(110), 2)  # Fair
        self.assertEqual(self.scorer.score_height(90), 1)  # Poor

    def test_score_chest(self):
        """Test chest width scoring."""
        # Test different score ranges
        self.assertEqual(self.scorer.score_chest(70), 4)  # Excellent
        self.assertEqual(self.scorer.score_chest(55), 3)  # Good
        self.assertEqual(self.scorer.score_chest(45), 2)  # Fair
        self.assertEqual(self.scorer.score_chest(30), 1)  # Poor

    def test_score_rump(self):
        """Test rump angle scoring."""
        # Test different score ranges
        self.assertEqual(self.scorer.score_rump(30), 4)  # Excellent
        self.assertEqual(self.scorer.score_rump(22), 3)  # Good
        self.assertEqual(self.scorer.score_rump(17), 2)  # Fair
        self.assertEqual(self.scorer.score_rump(10), 1)  # Poor

    def test_calculate_atc_scores(self):
        """Test ATC score calculation."""
        measurements = {
            "body_length_cm": 180,
            "height_withers_cm": 140,
            "chest_width_cm": 60,
            "rump_angle_deg": 25,
        }

        scores = self.scorer.calculate_atc_scores(measurements)

        # Check component scores
        component_scores = scores["atc_component_scores"]
        self.assertIn("body_length_score", component_scores)
        self.assertIn("height_score", component_scores)
        self.assertIn("chest_score", component_scores)
        self.assertIn("rump_score", component_scores)

        # Check total score
        total_score = scores["atc_total_score"]
        expected_total = sum(component_scores.values())
        self.assertEqual(total_score, expected_total)
        self.assertGreaterEqual(total_score, 4)
        self.assertLessEqual(total_score, 16)


class TestMeasurementPipeline(unittest.TestCase):
    """Test cases for the complete measurement pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = MeasurementPipeline()

        # Create test data
        self.test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        self.test_keypoints = [
            [100, 180, 2],
            [100, 160, 2],
            [100, 140, 2],
            [100, 120, 2],
            [80, 120, 2],
            [120, 120, 2],
            [90, 100, 2],
            [110, 100, 2],
            [100, 80, 2],
            [100, 90, 2],
            [80, 140, 2],
            [120, 140, 2],
            [90, 120, 2],
            [110, 120, 2],
        ]
        self.test_mask = np.zeros((200, 200), dtype=np.uint8)
        self.test_mask[80:160, 80:120] = 255

    def test_process_detection(self):
        """Test complete detection processing."""
        result = self.pipeline.process_detection(
            self.test_image, self.test_keypoints, self.test_mask
        )

        # Check result structure
        self.assertIn("measurements_cm", result)
        self.assertIn("atc_component_scores", result)
        self.assertIn("atc_total_score", result)

        # Check measurements
        measurements = result["measurements_cm"]
        required_measurements = [
            "body_length_cm",
            "height_withers_cm",
            "chest_width_cm",
            "rump_angle_deg",
        ]
        for measurement in required_measurements:
            self.assertIn(measurement, measurements)
            self.assertIsInstance(measurements[measurement], (int, float))

        # Check ATC scores
        component_scores = result["atc_component_scores"]
        required_scores = [
            "body_length_score",
            "height_score",
            "chest_score",
            "rump_score",
        ]
        for score in required_scores:
            self.assertIn(score, component_scores)
            self.assertGreaterEqual(component_scores[score], 1)
            self.assertLessEqual(component_scores[score], 4)

        # Check total score
        total_score = result["atc_total_score"]
        self.assertGreaterEqual(total_score, 4)
        self.assertLessEqual(total_score, 16)

    def test_process_detection_with_missing_data(self):
        """Test processing with missing keypoints."""
        # Create keypoints with some missing
        incomplete_keypoints = self.test_keypoints.copy()
        for i in range(0, len(incomplete_keypoints), 3):
            incomplete_keypoints[i + 2] = 0  # Make all keypoints invisible

        result = self.pipeline.process_detection(
            self.test_image, incomplete_keypoints, self.test_mask
        )

        # Should still return a result, but measurements may be 0
        self.assertIn("measurements_cm", result)
        self.assertIn("atc_component_scores", result)
        self.assertIn("atc_total_score", result)


if __name__ == "__main__":
    unittest.main()
