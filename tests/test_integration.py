"""
Integration tests for the complete ATC pipeline.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.generate_synthetic_data import generate_dataset
from src.inference import ATCInference
from src.measurement.core import MeasurementPipeline


class TestEndToEndPipeline(unittest.TestCase):
    """Test cases for end-to-end pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "synthetic_data")
        self.model_path = "artifacts/model.pt"  # Will be created during training

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_synthetic_data_generation(self):
        """Test that synthetic data can be generated successfully."""
        # Generate small dataset
        num_images = 5
        generate_dataset(num_images, self.data_dir)

        # Verify dataset structure
        self.assertTrue(os.path.exists(self.data_dir))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "images")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "annotations.json")))

        # Check images
        images_dir = os.path.join(self.data_dir, "images")
        image_files = list(Path(images_dir).glob("*.jpg"))
        self.assertEqual(len(image_files), num_images)

        # Check annotations
        with open(os.path.join(self.data_dir, "annotations.json"), "r") as f:
            coco_data = json.load(f)

        self.assertEqual(len(coco_data["images"]), num_images)
        self.assertEqual(len(coco_data["annotations"]), num_images)
        self.assertEqual(len(coco_data["categories"]), 1)

    def test_measurement_pipeline_integration(self):
        """Test measurement pipeline with synthetic data."""
        # Generate test data
        generate_dataset(3, self.data_dir)

        # Load an image and its annotation
        images_dir = os.path.join(self.data_dir, "images")
        image_files = list(Path(images_dir).glob("*.jpg"))
        self.assertGreater(len(image_files), 0)

        # Load first image
        image_path = str(image_files[0])
        image = cv2.imread(image_path)
        self.assertIsNotNone(image)

        # Load corresponding annotation
        with open(os.path.join(self.data_dir, "annotations.json"), "r") as f:
            coco_data = json.load(f)

        annotation = coco_data["annotations"][0]
        keypoints = annotation["keypoints"]

        # Create a simple mask (in practice, this would come from model prediction)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        bbox = annotation["bbox"]
        x, y, w, h = bbox
        mask[int(y) : int(y + h), int(x) : int(x + w)] = 255

        # Test measurement pipeline
        pipeline = MeasurementPipeline()
        result = pipeline.process_detection(image, keypoints, mask)

        # Verify result structure
        self.assertIn("measurements_cm", result)
        self.assertIn("atc_component_scores", result)
        self.assertIn("atc_total_score", result)

        # Verify measurements
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
            self.assertGreaterEqual(measurements[measurement], 0)

        # Verify ATC scores
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

        # Verify total score
        total_score = result["atc_total_score"]
        self.assertGreaterEqual(total_score, 4)
        self.assertLessEqual(total_score, 16)

    def test_json_output_format(self):
        """Test that the output JSON follows the required schema."""
        # Generate test data
        generate_dataset(2, self.data_dir)

        # Create mock inference result
        mock_result = {
            "animal_id": None,
            "timestamp": "2024-01-01T12:00:00Z",
            "detections": [
                {
                    "class": "bovine",
                    "score": 0.95,
                    "bbox": [100, 100, 200, 300],
                    "mask_rle": {"counts": "test_counts", "size": [400, 600]},
                    "keypoints": [
                        150,
                        200,
                        2,  # muzzle_tip
                        150,
                        180,
                        2,  # forehead_top
                        150,
                        160,
                        2,  # withers
                        150,
                        140,
                        2,  # chest_center
                        130,
                        140,
                        2,  # left_chest_side
                        170,
                        140,
                        2,  # right_chest_side
                        140,
                        120,
                        2,  # hip_left
                        160,
                        120,
                        2,  # hip_right
                        150,
                        100,
                        2,  # tail_base
                        150,
                        110,
                        2,  # rump_top
                        130,
                        180,
                        2,  # left_fore_hoof
                        170,
                        180,
                        2,  # right_fore_hoof
                        140,
                        160,
                        2,  # left_rear_hoof
                        160,
                        160,
                        2,  # right_rear_hoof
                    ],
                    "measurements_cm": {
                        "body_length_cm": 120.5,
                        "height_withers_cm": 95.2,
                        "chest_width_cm": 45.8,
                        "rump_angle_deg": 22.3,
                    },
                    "atc_component_scores": {
                        "body_length_score": 3,
                        "height_score": 4,
                        "chest_score": 2,
                        "rump_score": 3,
                    },
                    "atc_total_score": 12,
                }
            ],
            "processing_time_ms": 150.5,
            "image_id": "test_image.jpg",
        }

        # Verify JSON structure
        self.assertIn("animal_id", mock_result)
        self.assertIn("timestamp", mock_result)
        self.assertIn("detections", mock_result)
        self.assertIn("processing_time_ms", mock_result)
        self.assertIn("image_id", mock_result)

        # Verify detection structure
        detection = mock_result["detections"][0]
        required_detection_keys = [
            "class",
            "score",
            "bbox",
            "mask_rle",
            "keypoints",
            "measurements_cm",
            "atc_component_scores",
            "atc_total_score",
        ]
        for key in required_detection_keys:
            self.assertIn(key, detection)

        # Verify measurements structure
        measurements = detection["measurements_cm"]
        required_measurements = [
            "body_length_cm",
            "height_withers_cm",
            "chest_width_cm",
            "rump_angle_deg",
        ]
        for measurement in required_measurements:
            self.assertIn(measurement, measurements)

        # Verify ATC scores structure
        atc_scores = detection["atc_component_scores"]
        required_scores = [
            "body_length_score",
            "height_score",
            "chest_score",
            "rump_score",
        ]
        for score in required_scores:
            self.assertIn(score, atc_scores)

        # Verify data types
        self.assertIsInstance(detection["score"], (int, float))
        self.assertIsInstance(detection["bbox"], list)
        self.assertEqual(len(detection["bbox"]), 4)
        self.assertIsInstance(detection["keypoints"], list)
        self.assertEqual(len(detection["keypoints"]), 42)  # 14 keypoints * 3 values
        self.assertIsInstance(detection["atc_total_score"], int)

    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Test with invalid image
        pipeline = MeasurementPipeline()

        # Create invalid inputs
        invalid_image = np.array([])  # Empty image
        invalid_keypoints = []  # Empty keypoints
        invalid_mask = np.array([])  # Empty mask

        # Should handle gracefully
        result = pipeline.process_detection(
            invalid_image, invalid_keypoints, invalid_mask
        )

        # Should still return a result structure
        self.assertIn("measurements_cm", result)
        self.assertIn("atc_component_scores", result)
        self.assertIn("atc_total_score", result)

        # Measurements should be 0 or default values
        measurements = result["measurements_cm"]
        for measurement in measurements.values():
            self.assertGreaterEqual(measurement, 0)

    def test_data_consistency(self):
        """Test data consistency across the pipeline."""
        # Generate test data
        generate_dataset(5, self.data_dir)

        # Load annotations
        with open(os.path.join(self.data_dir, "annotations.json"), "r") as f:
            coco_data = json.load(f)

        # Verify that each image has a corresponding annotation
        self.assertEqual(len(coco_data["images"]), len(coco_data["annotations"]))

        # Verify image-annotation correspondence
        for i, image in enumerate(coco_data["images"]):
            annotation = coco_data["annotations"][i]
            self.assertEqual(image["id"], annotation["image_id"])

            # Verify keypoints count
            self.assertEqual(len(annotation["keypoints"]), 42)  # 14 * 3

            # Verify bbox format
            bbox = annotation["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertTrue(all(isinstance(x, (int, float)) for x in bbox))

            # Verify area calculation
            self.assertGreater(annotation["area"], 0)

    def test_performance_benchmarks(self):
        """Test basic performance benchmarks."""
        import time

        # Generate test data
        generate_dataset(10, self.data_dir)

        # Test data generation performance
        start_time = time.time()
        generate_dataset(5, os.path.join(self.temp_dir, "perf_test"))
        generation_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(generation_time, 30)  # 30 seconds for 5 images

        # Test measurement pipeline performance
        pipeline = MeasurementPipeline()

        # Create test inputs
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        test_keypoints = [[100, 200, 2]] * 14  # 14 keypoints
        test_mask = np.zeros((400, 600), dtype=np.uint8)
        test_mask[100:300, 100:500] = 255

        # Measure processing time
        start_time = time.time()
        for _ in range(10):  # Run 10 iterations
            result = pipeline.process_detection(test_image, test_keypoints, test_mask)
        processing_time = time.time() - start_time

        # Should process quickly (adjust threshold as needed)
        avg_time_per_image = processing_time / 10
        self.assertLess(avg_time_per_image, 1.0)  # Less than 1 second per image


if __name__ == "__main__":
    unittest.main()
