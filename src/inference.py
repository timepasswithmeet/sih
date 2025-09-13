#!/usr/bin/env python3
"""
Inference script for ATC (Animal Type Classification) model.

Loads a trained model and runs inference on images, producing
measurements and ATC scores in the specified JSON format.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from pycocotools import mask as mask_util

from src.measurement.core import MeasurementPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATCInference:
    """Main inference class for ATC model."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to config file (optional)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.predictor = None
        self.measurement_pipeline = MeasurementPipeline()
        self.metadata = None

        self._setup_model()

    def _setup_model(self) -> None:
        """Setup the trained model for inference."""
        # Create configuration
        cfg = get_cfg()

        # Load base configuration
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        # Load custom configuration if provided
        if self.config_path and os.path.exists(self.config_path):
            cfg.merge_from_file(self.config_path)

        # Set model device
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Set number of classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "bovine" class

        # Set keypoint configuration
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14

        # Set model weights
        cfg.MODEL.WEIGHTS = self.model_path

        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # Create predictor
        self.predictor = DefaultPredictor(cfg)

        # Set metadata
        self.metadata = MetadataCatalog.get("atc_synthetic_train")
        self.metadata.set(
            thing_classes=["bovine"],
            keypoint_names=[
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
            ],
        )

        logger.info(f"Model loaded from {self.model_path}")
        logger.info(f"Device: {cfg.MODEL.DEVICE}")

    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary containing detection results and measurements
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run prediction
        start_time = time.time()
        outputs = self.predictor(image)
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Process detections
        detections = []

        if "instances" in outputs:
            instances = outputs["instances"]

            for i in range(len(instances)):
                detection = self._process_detection(instances[i], image)
                if detection:
                    detections.append(detection)

        # Create result
        result = {
            "animal_id": None,  # Optional field
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detections": detections,
            "processing_time_ms": inference_time,
            "image_id": os.path.basename(image_path),
        }

        return result

    def _process_detection(
        self, instance: Instances, image: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single detection instance.

        Args:
            instance: Detection instance from model
            image: Original image

        Returns:
            Dictionary containing detection data and measurements
        """
        # Extract basic detection info
        bbox = instance.pred_boxes.tensor[0].cpu().numpy().tolist()
        score = float(instance.scores[0].cpu().numpy())
        class_id = int(instance.pred_classes[0].cpu().numpy())

        # Extract mask
        mask = instance.pred_masks[0].cpu().numpy().astype(np.uint8)

        # Convert mask to RLE format
        mask_rle = mask_util.encode(np.asfortranarray(mask))
        mask_rle["counts"] = mask_rle["counts"].decode("utf-8")

        # Extract keypoints
        keypoints = []
        if hasattr(instance, "pred_keypoints") and instance.pred_keypoints is not None:
            kpts = instance.pred_keypoints[0].cpu().numpy()
            for kpt in kpts:
                keypoints.append([float(kpt[0]), float(kpt[1]), float(kpt[2])])

        # Run measurement pipeline
        try:
            measurements_result = self.measurement_pipeline.process_detection(
                image, keypoints, mask
            )
        except Exception as e:
            logger.warning(f"Measurement pipeline failed: {e}")
            measurements_result = {
                "measurements_cm": {
                    "body_length_cm": 0.0,
                    "height_withers_cm": 0.0,
                    "chest_width_cm": 0.0,
                    "rump_angle_deg": 0.0,
                },
                "atc_component_scores": {
                    "body_length_score": 0,
                    "height_score": 0,
                    "chest_score": 0,
                    "rump_score": 0,
                },
                "atc_total_score": 0,
            }

        # Create detection result
        detection = {
            "class": "bovine",
            "score": score,
            "bbox": bbox,
            "mask_rle": mask_rle,
            "keypoints": keypoints,
            **measurements_result,
        }

        return detection

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of detection results
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Add error result
                error_result = {
                    "animal_id": None,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "detections": [],
                    "processing_time_ms": 0,
                    "image_id": os.path.basename(image_path),
                    "error": str(e),
                }
                results.append(error_result)

        return results

    def visualize_detection(self, image_path: str, output_path: str) -> None:
        """
        Visualize detection results on image.

        Args:
            image_path: Path to input image
            output_path: Path to save visualized image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run prediction
        outputs = self.predictor(image)

        # Create visualizer
        visualizer = Visualizer(
            image[:, :, ::-1],  # Convert BGR to RGB
            metadata=self.metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW,
        )

        # Draw predictions
        if "instances" in outputs:
            vis_output = visualizer.draw_instance_predictions(
                outputs["instances"].to("cpu")
            )

            # Save visualized image
            cv2.imwrite(
                output_path, vis_output.get_image()[:, :, ::-1]
            )  # Convert RGB to BGR
            logger.info(f"Visualization saved to {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run ATC inference")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mask_rcnn_atc.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--images", type=str, required=True, help="Path to image or directory of images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_outputs.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization images"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization images",
    )

    args = parser.parse_args()

    # Initialize inference pipeline
    inference = ATCInference(args.model, args.config)

    # Collect image paths
    image_paths = []
    if os.path.isfile(args.images):
        image_paths = [args.images]
    elif os.path.isdir(args.images):
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(Path(args.images).glob(ext))
        image_paths = [str(p) for p in image_paths]
    else:
        raise ValueError(f"Invalid image path: {args.images}")

    logger.info(f"Found {len(image_paths)} images to process")

    # Run inference
    results = inference.predict_batch(image_paths)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")

    # Generate visualizations if requested
    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)

        for i, image_path in enumerate(image_paths):
            try:
                output_path = os.path.join(args.viz_dir, f"viz_{i:03d}.jpg")
                inference.visualize_detection(image_path, output_path)
            except Exception as e:
                logger.error(f"Failed to visualize {image_path}: {e}")

    # Print summary
    total_detections = sum(len(result.get("detections", [])) for result in results)
    avg_processing_time = np.mean(
        [result.get("processing_time_ms", 0) for result in results]
    )

    logger.info(f"Processing complete:")
    logger.info(f"  Images processed: {len(image_paths)}")
    logger.info(f"  Total detections: {total_detections}")
    logger.info(f"  Average processing time: {avg_processing_time:.1f} ms")


if __name__ == "__main__":
    main()
