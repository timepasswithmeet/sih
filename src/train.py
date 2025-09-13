#!/usr/bin/env python3
"""
Training script for ATC (Animal Type Classification) model.

Uses Detectron2 to train a Mask R-CNN model for instance segmentation
and keypoint detection on synthetic cattle/buffalo data.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def setup_logging(output_dir: str) -> None:
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, "train_log.txt")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info("Training started")


def register_datasets(data_dir: str) -> None:
    """Register COCO datasets for training and validation."""
    # Register training dataset
    register_coco_instances(
        "atc_synthetic_train",
        {},
        os.path.join(data_dir, "annotations.json"),
        os.path.join(data_dir, "images"),
    )

    # For validation, we'll use the same dataset (in practice, you'd split it)
    register_coco_instances(
        "atc_synthetic_val",
        {},
        os.path.join(data_dir, "annotations.json"),
        os.path.join(data_dir, "images"),
    )

    # Set metadata
    MetadataCatalog.get("atc_synthetic_train").set(
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
        keypoint_flip_map=[
            ("left_chest_side", "right_chest_side"),
            ("hip_left", "hip_right"),
            ("left_fore_hoof", "right_fore_hoof"),
            ("left_rear_hoof", "right_rear_hoof"),
        ],
    )

    MetadataCatalog.get("atc_synthetic_val").set(
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
        keypoint_flip_map=[
            ("left_chest_side", "right_chest_side"),
            ("hip_left", "hip_right"),
            ("left_fore_hoof", "right_fore_hoof"),
            ("left_rear_hoof", "right_rear_hoof"),
        ],
    )


class ATCTrainer(DefaultTrainer):
    """Custom trainer for ATC model."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for validation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def run_step(self):
        """Run one training step with logging."""
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        data = next(self._trainer._data_loader_iter)
        data_time = (
            time.perf_counter() - self._trainer._data_loader_iter._last_data_time
        )

        # Forward pass
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        # Backward pass
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        end.record()
        torch.cuda.synchronize()

        # Log metrics
        if self._trainer._last_write_time is None:
            self._trainer._last_write_time = time.perf_counter()

        storage = EventStorage.get_current()
        storage.put_scalars(data_time=data_time, **loss_dict)
        storage.put_scalar("lr", self.optimizer.param_groups[0]["lr"])
        storage.put_scalar("time", time.perf_counter() - self._trainer._last_write_time)

        self._trainer._last_write_time = time.perf_counter()
        self._trainer._iter += 1


def setup_config(config_file: str, output_dir: str, epochs: int = 10) -> Any:
    """Setup Detectron2 configuration."""
    cfg = get_cfg()

    # Load base configuration
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # Load custom configuration if provided
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)

    # Override with command line arguments
    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.MAX_ITER = epochs * 100  # Approximate iterations per epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1, epochs // 5)  # Save checkpoints

    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Freeze backbone for faster training on small dataset
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    return cfg


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ATC model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mask_rcnn_atc.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)

    # Register datasets
    register_datasets(args.data_dir)
    logger.info(f"Registered datasets from {args.data_dir}")

    # Setup configuration
    cfg = setup_config(args.config, args.output_dir, args.epochs)
    logger.info(f"Configuration loaded from {args.config}")

    # Create trainer
    trainer = ATCTrainer(cfg)

    if args.eval_only:
        # Load model and run evaluation
        trainer.resume_or_load(resume=True)
        trainer.test()
    else:
        # Train model
        trainer.resume_or_load(resume=args.resume)
        trainer.train()

    logger.info("Training completed successfully")


if __name__ == "__main__":
    import time

    main()
