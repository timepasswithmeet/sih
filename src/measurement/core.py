"""
Core measurement pipeline for animal body measurements.

This module handles:
1. Reference marker detection (ArUco or A4 sheet)
2. Scale conversion from pixels to millimeters
3. Body measurements calculation
4. ATC component scoring
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import euclidean

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Keypoint indices (matching the order from data generator)
KEYPOINT_INDICES = {
    "muzzle_tip": 0,
    "forehead_top": 1,
    "withers": 2,
    "chest_center": 3,
    "left_chest_side": 4,
    "right_chest_side": 5,
    "hip_left": 6,
    "hip_right": 7,
    "tail_base": 8,
    "rump_top": 9,
    "left_fore_hoof": 10,
    "right_fore_hoof": 11,
    "left_rear_hoof": 12,
    "right_rear_hoof": 13,
}

# ArUco marker parameters
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
MARKER_SIZE_MM = 50  # 50mm marker
A4_WIDTH_MM = 210  # A4 sheet width in mm
A4_HEIGHT_MM = 297  # A4 sheet height in mm


class ReferenceMarkerDetector:
    """Detects reference markers for scale conversion."""

    def __init__(self):
        self.aruco_params = cv2.aruco.DetectorParameters_create()

    def detect_aruco_marker(self, image: np.ndarray) -> Optional[float]:
        """
        Detect ArUco marker and return scale factor (mm per pixel).

        Args:
            image: Input image

        Returns:
            Scale factor in mm/pixel, or None if no marker found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, ARUCO_DICT, parameters=self.aruco_params
        )

        if ids is not None and len(ids) > 0:
            # Use the first detected marker
            marker_corners = corners[0][0]

            # Calculate marker width in pixels
            marker_width_px = euclidean(marker_corners[0], marker_corners[1])

            # Calculate scale factor
            scale_mm_per_px = MARKER_SIZE_MM / marker_width_px

            logger.info(f"ArUco marker detected. Scale: {scale_mm_per_px:.3f} mm/pixel")
            return scale_mm_per_px

        return None

    def detect_a4_sheet(self, image: np.ndarray) -> Optional[float]:
        """
        Detect A4 sheet as fallback reference marker.

        Args:
            image: Input image

        Returns:
            Scale factor in mm/pixel, or None if no sheet found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                # Calculate area
                area = cv2.contourArea(contour)

                # Check if area is reasonable for A4 sheet
                if area > 10000:  # Minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate aspect ratio
                    aspect_ratio = w / h
                    a4_aspect_ratio = A4_WIDTH_MM / A4_HEIGHT_MM

                    # Check if aspect ratio matches A4 (with tolerance)
                    if abs(aspect_ratio - a4_aspect_ratio) < 0.1:
                        # Calculate scale factor based on width
                        scale_mm_per_px = A4_WIDTH_MM / w

                        logger.info(
                            f"A4 sheet detected. Scale: {scale_mm_per_px:.3f} mm/pixel"
                        )
                        return scale_mm_per_px

        return None

    def detect_reference_marker(self, image: np.ndarray) -> float:
        """
        Detect reference marker and return scale factor.

        Args:
            image: Input image

        Returns:
            Scale factor in mm/pixel (defaults to 1.0 if no marker found)
        """
        # Try ArUco marker first
        scale = self.detect_aruco_marker(image)

        if scale is None:
            # Fallback to A4 sheet detection
            scale = self.detect_a4_sheet(image)

        if scale is None:
            logger.warning(
                "No reference marker detected. Using default scale (1.0 mm/pixel)"
            )
            return 1.0

        return scale


class GroundEstimator:
    """Estimates ground line from segmentation mask."""

    @staticmethod
    def estimate_ground_line(mask: np.ndarray) -> float:
        """
        Estimate ground line from the lowest points of the mask.

        Args:
            mask: Binary segmentation mask

        Returns:
            Y-coordinate of estimated ground line
        """
        # Find the lowest non-zero pixel in each column
        ground_points = []

        for x in range(mask.shape[1]):
            column = mask[:, x]
            nonzero_indices = np.where(column > 0)[0]

            if len(nonzero_indices) > 0:
                # Get the lowest (highest y-value) point
                lowest_y = np.max(nonzero_indices)
                ground_points.append(lowest_y)

        if ground_points:
            # Use the median of ground points for robustness
            ground_y = np.median(ground_points)
            return float(ground_y)

        # Fallback to bottom of image
        return float(mask.shape[0] - 1)


class BodyMeasurements:
    """Calculates body measurements from keypoints and mask."""

    def __init__(self, scale_mm_per_px: float):
        self.scale_mm_per_px = scale_mm_per_px
        self.ground_estimator = GroundEstimator()

    def calculate_body_length(self, keypoints: List[List[float]]) -> float:
        """
        Calculate body length from muzzle tip to tail base.

        Args:
            keypoints: List of [x, y, score] keypoints

        Returns:
            Body length in centimeters
        """
        muzzle_idx = KEYPOINT_INDICES["muzzle_tip"]
        tail_idx = KEYPOINT_INDICES["tail_base"]

        if (
            len(keypoints) > muzzle_idx
            and len(keypoints) > tail_idx
            and keypoints[muzzle_idx][2] > 0
            and keypoints[tail_idx][2] > 0
        ):

            muzzle = keypoints[muzzle_idx]
            tail = keypoints[tail_idx]

            # Calculate Euclidean distance
            distance_px = euclidean([muzzle[0], muzzle[1]], [tail[0], tail[1]])
            distance_mm = distance_px * self.scale_mm_per_px
            distance_cm = distance_mm / 10.0

            return distance_cm

        logger.warning("Missing keypoints for body length calculation")
        return 0.0

    def calculate_height_at_withers(
        self, keypoints: List[List[float]], mask: np.ndarray
    ) -> float:
        """
        Calculate height at withers from ground line.

        Args:
            keypoints: List of [x, y, score] keypoints
            mask: Binary segmentation mask

        Returns:
            Height at withers in centimeters
        """
        withers_idx = KEYPOINT_INDICES["withers"]

        if len(keypoints) > withers_idx and keypoints[withers_idx][2] > 0:
            withers = keypoints[withers_idx]
            ground_y = self.ground_estimator.estimate_ground_line(mask)

            # Calculate vertical distance
            height_px = ground_y - withers[1]
            height_mm = height_px * self.scale_mm_per_px
            height_cm = height_mm / 10.0

            return max(0.0, height_cm)  # Ensure non-negative

        logger.warning("Missing withers keypoint for height calculation")
        return 0.0

    def calculate_chest_width(
        self, keypoints: List[List[float]], mask: np.ndarray
    ) -> float:
        """
        Calculate chest width between left and right chest sides.

        Args:
            keypoints: List of [x, y, score] keypoints
            mask: Binary segmentation mask

        Returns:
            Chest width in centimeters
        """
        left_chest_idx = KEYPOINT_INDICES["left_chest_side"]
        right_chest_idx = KEYPOINT_INDICES["right_chest_side"]

        if (
            len(keypoints) > left_chest_idx
            and len(keypoints) > right_chest_idx
            and keypoints[left_chest_idx][2] > 0
            and keypoints[right_chest_idx][2] > 0
        ):

            left_chest = keypoints[left_chest_idx]
            right_chest = keypoints[right_chest_idx]

            # Calculate horizontal distance
            width_px = abs(right_chest[0] - left_chest[0])
            width_mm = width_px * self.scale_mm_per_px
            width_cm = width_mm / 10.0

            return width_cm

        # Fallback: use mask width at chest center
        chest_center_idx = KEYPOINT_INDICES["chest_center"]
        if len(keypoints) > chest_center_idx and keypoints[chest_center_idx][2] > 0:
            chest_center = keypoints[chest_center_idx]
            chest_y = int(chest_center[1])

            if 0 <= chest_y < mask.shape[0]:
                # Find width at chest level
                row = mask[chest_y, :]
                nonzero_indices = np.where(row > 0)[0]

                if len(nonzero_indices) > 0:
                    width_px = np.max(nonzero_indices) - np.min(nonzero_indices)
                    width_mm = width_px * self.scale_mm_per_px
                    width_cm = width_mm / 10.0

                    return width_cm

        logger.warning("Missing keypoints for chest width calculation")
        return 0.0

    def calculate_rump_angle(self, keypoints: List[List[float]]) -> float:
        """
        Calculate rump angle from hip center to rump top and tail base.

        Args:
            keypoints: List of [x, y, score] keypoints

        Returns:
            Rump angle in degrees
        """
        hip_left_idx = KEYPOINT_INDICES["hip_left"]
        hip_right_idx = KEYPOINT_INDICES["hip_right"]
        rump_top_idx = KEYPOINT_INDICES["rump_top"]
        tail_base_idx = KEYPOINT_INDICES["tail_base"]

        if (
            len(keypoints) > hip_left_idx
            and len(keypoints) > hip_right_idx
            and len(keypoints) > rump_top_idx
            and len(keypoints) > tail_base_idx
            and keypoints[hip_left_idx][2] > 0
            and keypoints[hip_right_idx][2] > 0
            and keypoints[rump_top_idx][2] > 0
            and keypoints[tail_base_idx][2] > 0
        ):

            hip_left = keypoints[hip_left_idx]
            hip_right = keypoints[hip_right_idx]
            rump_top = keypoints[rump_top_idx]
            tail_base = keypoints[tail_base_idx]

            # Calculate hip center
            hip_center = [
                (hip_left[0] + hip_right[0]) / 2,
                (hip_left[1] + hip_right[1]) / 2,
            ]

            # Calculate vectors
            vector1 = [rump_top[0] - hip_center[0], rump_top[1] - hip_center[1]]
            vector2 = [tail_base[0] - hip_center[0], tail_base[1] - hip_center[1]]

            # Calculate angle between vectors
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            if magnitude1 > 0 and magnitude2 > 0:
                cos_angle = dot_product / (magnitude1 * magnitude2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
                angle_rad = math.acos(cos_angle)
                angle_deg = math.degrees(angle_rad)

                return angle_deg

        logger.warning("Missing keypoints for rump angle calculation")
        return 0.0

    def calculate_all_measurements(
        self, keypoints: List[List[float]], mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all body measurements.

        Args:
            keypoints: List of [x, y, score] keypoints
            mask: Binary segmentation mask

        Returns:
            Dictionary of measurements in centimeters/degrees
        """
        measurements = {
            "body_length_cm": self.calculate_body_length(keypoints),
            "height_withers_cm": self.calculate_height_at_withers(keypoints, mask),
            "chest_width_cm": self.calculate_chest_width(keypoints, mask),
            "rump_angle_deg": self.calculate_rump_angle(keypoints),
        }

        return measurements


class ATCScorer:
    """Calculates ATC component scores from measurements."""

    def __init__(self):
        # Rule-based scoring thresholds (can be replaced with trained model)
        self.thresholds = {
            "body_length": {"excellent": 180, "good": 160, "fair": 140},
            "height": {"excellent": 140, "good": 120, "fair": 100},
            "chest": {"excellent": 60, "good": 50, "fair": 40},
            "rump": {"excellent": 25, "good": 20, "fair": 15},  # degrees
        }

    def score_body_length(self, measurement: float) -> int:
        """Score body length measurement."""
        if measurement >= self.thresholds["body_length"]["excellent"]:
            return 4
        elif measurement >= self.thresholds["body_length"]["good"]:
            return 3
        elif measurement >= self.thresholds["body_length"]["fair"]:
            return 2
        else:
            return 1

    def score_height(self, measurement: float) -> int:
        """Score height measurement."""
        if measurement >= self.thresholds["height"]["excellent"]:
            return 4
        elif measurement >= self.thresholds["height"]["good"]:
            return 3
        elif measurement >= self.thresholds["height"]["fair"]:
            return 2
        else:
            return 1

    def score_chest(self, measurement: float) -> int:
        """Score chest width measurement."""
        if measurement >= self.thresholds["chest"]["excellent"]:
            return 4
        elif measurement >= self.thresholds["chest"]["good"]:
            return 3
        elif measurement >= self.thresholds["chest"]["fair"]:
            return 2
        else:
            return 1

    def score_rump(self, measurement: float) -> int:
        """Score rump angle measurement."""
        if measurement >= self.thresholds["rump"]["excellent"]:
            return 4
        elif measurement >= self.thresholds["rump"]["good"]:
            return 3
        elif measurement >= self.thresholds["rump"]["fair"]:
            return 2
        else:
            return 1

    def calculate_atc_scores(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate ATC component scores and total score.

        Args:
            measurements: Dictionary of measurements

        Returns:
            Dictionary containing component scores and total score
        """
        component_scores = {
            "body_length_score": self.score_body_length(measurements["body_length_cm"]),
            "height_score": self.score_height(measurements["height_withers_cm"]),
            "chest_score": self.score_chest(measurements["chest_width_cm"]),
            "rump_score": self.score_rump(measurements["rump_angle_deg"]),
        }

        # Calculate total score (sum of component scores)
        total_score = sum(component_scores.values())

        return {
            "atc_component_scores": component_scores,
            "atc_total_score": total_score,
        }


class MeasurementPipeline:
    """Main measurement pipeline orchestrating all components."""

    def __init__(self):
        self.marker_detector = ReferenceMarkerDetector()
        self.atc_scorer = ATCScorer()

    def process_detection(
        self, image: np.ndarray, keypoints: List[List[float]], mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single detection and return measurements and scores.

        Args:
            image: Original image
            keypoints: List of [x, y, score] keypoints
            mask: Binary segmentation mask

        Returns:
            Dictionary containing measurements and ATC scores
        """
        # Detect reference marker and get scale
        scale_mm_per_px = self.marker_detector.detect_reference_marker(image)

        # Initialize measurement calculator
        measurements_calc = BodyMeasurements(scale_mm_per_px)

        # Calculate all measurements
        measurements = measurements_calc.calculate_all_measurements(keypoints, mask)

        # Calculate ATC scores
        atc_scores = self.atc_scorer.calculate_atc_scores(measurements)

        # Combine results
        result = {"measurements_cm": measurements, **atc_scores}

        return result
