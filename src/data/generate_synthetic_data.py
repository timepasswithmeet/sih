#!/usr/bin/env python3
"""
Synthetic data generator for ATC project.

Generates synthetic images with animal-like shapes and COCO format annotations
including segmentation masks and keypoints.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


# Keypoint definitions (14 keypoints in exact order)
KEYPOINTS = [
    "muzzle_tip",
    "forehead_top",
    "withers",
    "chest_center",
    "left_chest_side",
    "right_chest_side",
    "hip_left",
    "hip_right",
    "tail_base",
    "rump_top",
    "left_fore_hoof",
    "right_fore_hoof",
    "left_rear_hoof",
    "right_rear_hoof",
]

# ArUco marker parameters
try:
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
except AttributeError:
    # Fallback for older OpenCV versions
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
MARKER_SIZE_MM = 50  # 50mm marker
MARKER_SIZE_PX = 100  # 100px in image


def create_aruco_marker(marker_id: int = 0) -> np.ndarray:
    """Create an ArUco marker image."""
    try:
        marker = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE_PX)
    except AttributeError:
        # Fallback for older OpenCV versions
        marker = cv2.aruco.drawMarker(ARUCO_DICT, marker_id, MARKER_SIZE_PX)
    return marker


def create_animal_shape(
    width: int, height: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Create a synthetic animal shape using ellipses.

    Returns:
        mask: Binary mask of the animal shape
        keypoints: List of (x, y) keypoint coordinates
    """
    # Create base image
    img = np.zeros((height, width), dtype=np.uint8)

    # Animal body parameters
    body_center_x = width // 2
    body_center_y = height // 2
    body_width = random.randint(80, 120)
    body_height = random.randint(120, 160)

    # Draw main body (ellipse)
    cv2.ellipse(
        img,
        (body_center_x, body_center_y),
        (body_width // 2, body_height // 2),
        0,
        0,
        360,
        255,
        -1,
    )

    # Draw head (smaller ellipse)
    head_x = body_center_x
    head_y = body_center_y - body_height // 3
    head_width = random.randint(40, 60)
    head_height = random.randint(50, 70)
    cv2.ellipse(
        img, (head_x, head_y), (head_width // 2, head_height // 2), 0, 0, 360, 255, -1
    )

    # Draw legs (rectangles)
    leg_width = 15
    leg_height = 60

    # Front legs
    front_leg_x = body_center_x - body_width // 4
    front_leg_y = body_center_y + body_height // 3
    cv2.rectangle(
        img,
        (front_leg_x - leg_width // 2, front_leg_y),
        (front_leg_x + leg_width // 2, front_leg_y + leg_height),
        255,
        -1,
    )

    front_leg_x2 = body_center_x + body_width // 4
    cv2.rectangle(
        img,
        (front_leg_x2 - leg_width // 2, front_leg_y),
        (front_leg_x2 + leg_width // 2, front_leg_y + leg_height),
        255,
        -1,
    )

    # Rear legs
    rear_leg_x = body_center_x - body_width // 4
    rear_leg_y = body_center_y + body_height // 2
    cv2.rectangle(
        img,
        (rear_leg_x - leg_width // 2, rear_leg_y),
        (rear_leg_x + leg_width // 2, rear_leg_y + leg_height),
        255,
        -1,
    )

    rear_leg_x2 = body_center_x + body_width // 4
    cv2.rectangle(
        img,
        (rear_leg_x2 - leg_width // 2, rear_leg_y),
        (rear_leg_x2 + leg_width // 2, rear_leg_y + leg_height),
        255,
        -1,
    )

    # Generate keypoints based on the shape
    keypoints = []

    # Muzzle tip
    keypoints.append((head_x, head_y - head_height // 3))

    # Forehead top
    keypoints.append((head_x, head_y - head_height // 2))

    # Withers (top of body)
    keypoints.append((body_center_x, body_center_y - body_height // 2))

    # Chest center
    keypoints.append((body_center_x, body_center_y))

    # Left and right chest sides
    keypoints.append((body_center_x - body_width // 2, body_center_y))
    keypoints.append((body_center_x + body_width // 2, body_center_y))

    # Hip left and right
    keypoints.append(
        (body_center_x - body_width // 3, body_center_y + body_height // 3)
    )
    keypoints.append(
        (body_center_x + body_width // 3, body_center_y + body_height // 3)
    )

    # Tail base
    keypoints.append((body_center_x, body_center_y + body_height // 2))

    # Rump top
    keypoints.append((body_center_x, body_center_y + body_height // 4))

    # Hoof positions
    keypoints.append((front_leg_x, front_leg_y + leg_height))  # Left fore hoof
    keypoints.append((front_leg_x2, front_leg_y + leg_height))  # Right fore hoof
    keypoints.append((rear_leg_x, rear_leg_y + leg_height))  # Left rear hoof
    keypoints.append((rear_leg_x2, rear_leg_y + leg_height))  # Right rear hoof

    return img, keypoints


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Convert binary mask to polygon coordinates."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert to list format
    polygon = []
    for point in simplified:
        polygon.extend([int(point[0][0]), int(point[0][1])])

    return [polygon]


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask to COCO RLE format."""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"counts": [], "size": [mask.shape[0], mask.shape[1]]}

    # Create a single mask from all contours
    combined_mask = np.zeros_like(mask)
    cv2.fillPoly(combined_mask, contours, 1)

    # Convert to RLE
    rle = {
        "counts": combined_mask.flatten().tolist(),
        "size": [mask.shape[0], mask.shape[1]],
    }

    return rle


def generate_synthetic_image(
    image_id: int, width: int = 640, height: int = 480
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a single synthetic image with annotations."""
    # Create background
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background

    # Add some texture
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Create animal shape
    animal_mask, keypoints = create_animal_shape(width, height)

    # Color the animal
    animal_color = (
        random.randint(100, 200),  # Brownish colors
        random.randint(80, 150),
        random.randint(60, 120),
    )

    # Apply animal color to mask
    colored_animal = np.zeros_like(img)
    colored_animal[animal_mask > 0] = animal_color
    img = cv2.addWeighted(img, 0.7, colored_animal, 0.3, 0)

    # Add ArUco marker
    marker = create_aruco_marker(random.randint(0, 10))
    marker_x = random.randint(50, width - 150)
    marker_y = random.randint(50, height - 150)

    # Resize marker to fit
    marker_resized = cv2.resize(marker, (MARKER_SIZE_PX, MARKER_SIZE_PX))

    # Place marker on image
    img[marker_y : marker_y + MARKER_SIZE_PX, marker_x : marker_x + MARKER_SIZE_PX] = (
        np.stack([marker_resized] * 3, axis=2)
    )

    # Create annotation
    annotation = {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": mask_to_polygon(animal_mask),
        "area": int(np.sum(animal_mask > 0)),
        "bbox": cv2.boundingRect(animal_mask),  # [x, y, w, h]
        "iscrowd": 0,
        "keypoints": [],
        "num_keypoints": len(keypoints),
    }

    # Add keypoints (x, y, visibility)
    for kp in keypoints:
        annotation["keypoints"].extend([kp[0], kp[1], 2])  # 2 = visible

    return img, annotation


def generate_dataset(num_images: int = 30, output_dir: str = "data/synthetic") -> None:
    """Generate the complete synthetic dataset."""
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Initialize COCO dataset structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "bovine",
                "supercategory": "animal",
                "keypoints": KEYPOINTS,
                "skeleton": [
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [4, 6],  # Head to chest
                    [3, 7],
                    [3, 8],
                    [7, 8],  # Withers to hips
                    [7, 9],
                    [8, 9],
                    [9, 10],  # Hips to tail
                    [11, 12],
                    [13, 14],  # Hoof connections
                    [4, 11],
                    [4, 12],
                    [7, 13],
                    [8, 14],  # Body to hooves
                ],
            }
        ],
    }

    print(f"Generating {num_images} synthetic images...")

    for i in range(num_images):
        # Generate image and annotation
        img, annotation = generate_synthetic_image(i + 1)

        # Save image
        img_filename = f"image_{i+1:03d}.jpg"
        img_path = images_path / img_filename
        cv2.imwrite(str(img_path), img)

        # Add image info to COCO data
        image_info = {
            "id": i + 1,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": img_filename,
        }
        coco_data["images"].append(image_info)

        # Add annotation
        coco_data["annotations"].append(annotation)

        if (i + 1) % 5 == 0:
            print(f"Generated {i + 1}/{num_images} images")

    # Save COCO annotations
    annotations_path = output_path / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Dataset generation complete!")
    print(f"Images saved to: {images_path}")
    print(f"Annotations saved to: {annotations_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset for ATC")
    parser.add_argument(
        "--num-images", type=int, default=30, help="Number of images to generate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/synthetic", help="Output directory"
    )

    args = parser.parse_args()

    generate_dataset(args.num_images, args.output_dir)
