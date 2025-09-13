"""
Tests for data generation and processing modules.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.generate_synthetic_data import (
    KEYPOINTS,
    create_animal_shape,
    create_aruco_marker,
    generate_dataset,
    generate_synthetic_image,
    mask_to_polygon,
    mask_to_rle,
)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test cases for synthetic data generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_size = (640, 480)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_create_aruco_marker(self):
        """Test ArUco marker creation."""
        marker = create_aruco_marker(marker_id=0)

        # Check marker properties
        self.assertIsInstance(marker, np.ndarray)
        self.assertEqual(marker.shape, (100, 100))  # MARKER_SIZE_PX
        self.assertEqual(marker.dtype, np.uint8)

    def test_create_animal_shape(self):
        """Test animal shape creation."""
        width, height = self.test_image_size
        mask, keypoints = create_animal_shape(width, height)

        # Check mask properties
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (height, width))
        self.assertEqual(mask.dtype, np.uint8)

        # Check keypoints
        self.assertIsInstance(keypoints, list)
        self.assertEqual(len(keypoints), 14)  # 14 keypoints

        # Check that all keypoints are within image bounds
        for kp in keypoints:
            self.assertIsInstance(kp, tuple)
            self.assertEqual(len(kp), 2)
            x, y = kp
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, width)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, height)

    def test_mask_to_polygon(self):
        """Test mask to polygon conversion."""
        # Create a simple rectangular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        polygon = mask_to_polygon(mask)

        # Check polygon format
        self.assertIsInstance(polygon, list)
        if polygon:  # If contour was found
            self.assertIsInstance(polygon[0], list)
            self.assertGreater(len(polygon[0]), 0)
            # Check that polygon coordinates are even (x,y pairs)
            self.assertEqual(len(polygon[0]) % 2, 0)

    def test_mask_to_rle(self):
        """Test mask to RLE conversion."""
        # Create a simple rectangular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        rle = mask_to_rle(mask)

        # Check RLE format
        self.assertIsInstance(rle, dict)
        self.assertIn("counts", rle)
        self.assertIn("size", rle)
        self.assertEqual(rle["size"], [100, 100])
        self.assertIsInstance(rle["counts"], list)

    def test_generate_synthetic_image(self):
        """Test synthetic image generation."""
        image, annotation = generate_synthetic_image(1, *self.test_image_size)

        # Check image properties
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(
            image.shape, (self.test_image_size[1], self.test_image_size[0], 3)
        )
        self.assertEqual(image.dtype, np.uint8)

        # Check annotation properties
        self.assertIsInstance(annotation, dict)
        required_keys = [
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "area",
            "bbox",
            "iscrowd",
            "keypoints",
            "num_keypoints",
        ]
        for key in required_keys:
            self.assertIn(key, annotation)

        # Check keypoints format
        keypoints = annotation["keypoints"]
        self.assertIsInstance(keypoints, list)
        self.assertEqual(
            len(keypoints), 14 * 3
        )  # 14 keypoints * 3 values (x, y, visibility)

        # Check bbox format
        bbox = annotation["bbox"]
        self.assertIsInstance(bbox, (list, tuple))
        self.assertEqual(len(bbox), 4)  # [x, y, w, h]

    def test_generate_dataset(self):
        """Test complete dataset generation."""
        output_dir = os.path.join(self.temp_dir, "test_dataset")
        num_images = 5

        generate_dataset(num_images, output_dir)

        # Check output directory structure
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "images")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "annotations.json")))

        # Check images directory
        images_dir = os.path.join(output_dir, "images")
        image_files = list(Path(images_dir).glob("*.jpg"))
        self.assertEqual(len(image_files), num_images)

        # Check annotations file
        annotations_path = os.path.join(output_dir, "annotations.json")
        with open(annotations_path, "r") as f:
            coco_data = json.load(f)

        # Check COCO format structure
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            self.assertIn(key, coco_data)

        # Check data counts
        self.assertEqual(len(coco_data["images"]), num_images)
        self.assertEqual(len(coco_data["annotations"]), num_images)
        self.assertEqual(len(coco_data["categories"]), 1)

        # Check category structure
        category = coco_data["categories"][0]
        self.assertEqual(category["id"], 1)
        self.assertEqual(category["name"], "bovine")
        self.assertEqual(category["keypoints"], KEYPOINTS)

    def test_keypoint_consistency(self):
        """Test that keypoints are consistent across generations."""
        # Generate multiple images and check keypoint consistency
        keypoint_sets = []

        for i in range(3):
            _, annotation = generate_synthetic_image(i + 1, *self.test_image_size)
            keypoints = annotation["keypoints"]
            keypoint_sets.append(keypoints)

        # All should have the same number of keypoints
        for keypoints in keypoint_sets:
            self.assertEqual(len(keypoints), 14 * 3)

        # Check that keypoints are in expected format (x, y, visibility)
        for keypoints in keypoint_sets:
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                self.assertIsInstance(x, (int, float))
                self.assertIsInstance(y, (int, float))
                self.assertIsInstance(visibility, (int, float))
                self.assertGreaterEqual(visibility, 0)
                self.assertLessEqual(visibility, 2)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation."""

    def test_coco_annotation_format(self):
        """Test that generated annotations follow COCO format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate small dataset
            generate_dataset(2, temp_dir)

            # Load annotations
            annotations_path = os.path.join(temp_dir, "annotations.json")
            with open(annotations_path, "r") as f:
                coco_data = json.load(f)

            # Validate image entries
            for image in coco_data["images"]:
                required_keys = ["id", "width", "height", "file_name"]
                for key in required_keys:
                    self.assertIn(key, image)

            # Validate annotation entries
            for annotation in coco_data["annotations"]:
                required_keys = [
                    "id",
                    "image_id",
                    "category_id",
                    "segmentation",
                    "area",
                    "bbox",
                    "iscrowd",
                    "keypoints",
                    "num_keypoints",
                ]
                for key in required_keys:
                    self.assertIn(key, annotation)

                # Check bbox format
                bbox = annotation["bbox"]
                self.assertEqual(len(bbox), 4)
                self.assertTrue(all(isinstance(x, (int, float)) for x in bbox))

                # Check keypoints format
                keypoints = annotation["keypoints"]
                self.assertEqual(len(keypoints), 14 * 3)
                self.assertTrue(all(isinstance(x, (int, float)) for x in keypoints))

    def test_image_file_existence(self):
        """Test that all referenced image files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate dataset
            generate_dataset(3, temp_dir)

            # Load annotations
            annotations_path = os.path.join(temp_dir, "annotations.json")
            with open(annotations_path, "r") as f:
                coco_data = json.load(f)

            # Check that all referenced images exist
            images_dir = os.path.join(temp_dir, "images")
            for image in coco_data["images"]:
                image_path = os.path.join(images_dir, image["file_name"])
                self.assertTrue(os.path.exists(image_path))

                # Check that image can be loaded
                img = cv2.imread(image_path)
                self.assertIsNotNone(img)
                self.assertEqual(img.shape[:2], (image["height"], image["width"]))


if __name__ == "__main__":
    unittest.main()
