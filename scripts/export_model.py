#!/usr/bin/env python3
"""
Model export script for ATC model.

Exports trained PyTorch model to TorchScript and ONNX formats
for deployment and inference optimization.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.export import Caffe2Tracer, add_export_config
from detectron2.export.api import export_caffe2_model
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports trained models to different formats."""

    def __init__(self, model_path: str, config_path: str):
        """
        Initialize model exporter.

        Args:
            model_path: Path to trained PyTorch model
            config_path: Path to model configuration
        """
        self.model_path = model_path
        self.config_path = config_path
        self.cfg = None
        self.model = None

        self._setup_model()

    def _setup_model(self) -> None:
        """Setup the model for export."""
        # Load configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        if os.path.exists(self.config_path):
            self.cfg.merge_from_file(self.config_path)

        # Set model parameters
        self.cfg.MODEL.DEVICE = "cpu"  # Export on CPU
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
        self.cfg.MODEL.WEIGHTS = self.model_path

        # Build model
        self.model = build_model(self.cfg)
        self.model.eval()

        # Load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.model_path)

        logger.info(f"Model loaded from {self.model_path}")

    def export_torchscript(
        self, output_path: str, input_size: tuple = (800, 600)
    ) -> None:
        """
        Export model to TorchScript format.

        Args:
            output_path: Path to save TorchScript model
            input_size: Input image size (width, height)
        """
        logger.info("Exporting to TorchScript format...")

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])

        try:
            # Trace the model
            traced_model = torch.jit.trace(self.model, dummy_input)

            # Save traced model
            traced_model.save(output_path)
            logger.info(f"TorchScript model saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export TorchScript model: {e}")
            # Fallback: save the model state dict
            torch.save(
                self.model.state_dict(), output_path.replace(".pt", "_state_dict.pt")
            )
            logger.info(
                f"Model state dict saved to {output_path.replace('.pt', '_state_dict.pt')}"
            )

    def export_onnx(self, output_path: str, input_size: tuple = (800, 600)) -> None:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_size: Input image size (width, height)
        """
        logger.info("Exporting to ONNX format...")

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])

        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            logger.info(f"ONNX model saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            logger.info("ONNX export failed, but continuing with other exports")

    def export_all(
        self, output_dir: str, input_size: tuple = (800, 600)
    ) -> Dict[str, str]:
        """
        Export model to all supported formats.

        Args:
            output_dir: Directory to save exported models
            input_size: Input image size (width, height)

        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        exported_files = {}

        # Export TorchScript
        ts_path = os.path.join(output_dir, "model_ts.pt")
        self.export_torchscript(ts_path, input_size)
        exported_files["torchscript"] = ts_path

        # Export ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        self.export_onnx(onnx_path, input_size)
        exported_files["onnx"] = onnx_path

        # Save model info
        info_path = os.path.join(output_dir, "export_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Model Export Information\n")
            f.write(f"=======================\n\n")
            f.write(f"Original model: {self.model_path}\n")
            f.write(f"Config file: {self.config_path}\n")
            f.write(f"Input size: {input_size}\n")
            f.write(f"Export date: {torch.utils.data.get_worker_info()}\n\n")
            f.write(f"Exported files:\n")
            for format_name, file_path in exported_files.items():
                f.write(f"  {format_name}: {file_path}\n")

        logger.info(f"Export information saved to {info_path}")

        return exported_files


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export ATC model to different formats"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained PyTorch model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mask_rcnn_atc.yaml",
        help="Path to model configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default="800,600",
        help="Input image size as 'width,height'",
    )

    args = parser.parse_args()

    # Parse input size
    try:
        width, height = map(int, args.input_size.split(","))
        input_size = (width, height)
    except ValueError:
        logger.error("Invalid input size format. Use 'width,height' (e.g., '800,600')")
        return

    # Initialize exporter
    exporter = ModelExporter(args.model, args.config)

    # Export all formats
    exported_files = exporter.export_all(args.output_dir, input_size)

    # Print summary
    logger.info("Export completed successfully!")
    logger.info("Exported files:")
    for format_name, file_path in exported_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"  {format_name}: {file_path} ({file_size:.1f} MB)")
        else:
            logger.warning(f"  {format_name}: {file_path} (export failed)")


if __name__ == "__main__":
    main()
