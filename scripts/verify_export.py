#!/usr/bin/env python3
"""
Model export verification script.

Verifies that exported models (TorchScript, ONNX) produce
similar outputs to the original PyTorch model.
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVerifier:
    """Verifies exported models against original PyTorch model."""

    def __init__(self, pytorch_model_path: str, config_path: str):
        """
        Initialize model verifier.

        Args:
            pytorch_model_path: Path to original PyTorch model
            config_path: Path to model configuration
        """
        self.pytorch_model_path = pytorch_model_path
        self.config_path = config_path
        self.pytorch_model = None
        self.torchscript_model = None
        self.onnx_session = None

        self._setup_models()

    def _setup_models(self) -> None:
        """Setup all models for verification."""
        # Setup PyTorch model
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        if os.path.exists(self.config_path):
            cfg.merge_from_file(self.config_path)

        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
        cfg.MODEL.WEIGHTS = self.pytorch_model_path

        self.pytorch_model = build_model(cfg)
        self.pytorch_model.eval()

        checkpointer = DetectionCheckpointer(self.pytorch_model)
        checkpointer.load(self.pytorch_model_path)

        logger.info("PyTorch model loaded")

    def load_torchscript_model(self, torchscript_path: str) -> None:
        """Load TorchScript model."""
        if os.path.exists(torchscript_path):
            self.torchscript_model = torch.jit.load(torchscript_path)
            self.torchscript_model.eval()
            logger.info(f"TorchScript model loaded from {torchscript_path}")
        else:
            logger.warning(f"TorchScript model not found: {torchscript_path}")

    def load_onnx_model(self, onnx_path: str) -> None:
        """Load ONNX model."""
        if os.path.exists(onnx_path):
            self.onnx_session = ort.InferenceSession(onnx_path)
            logger.info(f"ONNX model loaded from {onnx_path}")
        else:
            logger.warning(f"ONNX model not found: {onnx_path}")

    def create_dummy_input(
        self, batch_size: int = 1, height: int = 600, width: int = 800
    ) -> torch.Tensor:
        """Create dummy input tensor for testing."""
        return torch.randn(batch_size, 3, height, width)

    def compare_outputs(
        self, output1: torch.Tensor, output2: torch.Tensor, tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Compare two model outputs.

        Args:
            output1: First output tensor
            output2: Second output tensor
            tolerance: Tolerance for comparison

        Returns:
            Dictionary with comparison results
        """
        # Calculate differences
        abs_diff = torch.abs(output1 - output2)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()

        # Calculate relative differences
        rel_diff = abs_diff / (torch.abs(output1) + 1e-8)
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()

        # Check if outputs are close
        is_close = torch.allclose(output1, output2, atol=tolerance, rtol=tolerance)

        return {
            "is_close": is_close,
            "max_absolute_diff": max_diff,
            "mean_absolute_diff": mean_diff,
            "max_relative_diff": max_rel_diff,
            "mean_relative_diff": mean_rel_diff,
            "tolerance": tolerance,
        }

    def verify_torchscript(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Verify TorchScript model output."""
        if self.torchscript_model is None:
            return {"error": "TorchScript model not loaded"}

        try:
            with torch.no_grad():
                # PyTorch model output
                pytorch_output = self.pytorch_model(input_tensor)

                # TorchScript model output
                torchscript_output = self.torchscript_model(input_tensor)

                # Compare outputs
                comparison = self.compare_outputs(pytorch_output, torchscript_output)

                return {"status": "success", "comparison": comparison}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def verify_onnx(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Verify ONNX model output."""
        if self.onnx_session is None:
            return {"error": "ONNX model not loaded"}

        try:
            with torch.no_grad():
                # PyTorch model output
                pytorch_output = self.pytorch_model(input_tensor)

                # ONNX model output
                input_numpy = input_tensor.numpy()
                onnx_output = self.onnx_session.run(None, {"input": input_numpy})[0]
                onnx_output = torch.from_numpy(onnx_output)

                # Compare outputs
                comparison = self.compare_outputs(pytorch_output, onnx_output)

                return {"status": "success", "comparison": comparison}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_verification(
        self, torchscript_path: str, onnx_path: str, num_tests: int = 5
    ) -> Dict[str, Any]:
        """
        Run comprehensive verification tests.

        Args:
            torchscript_path: Path to TorchScript model
            onnx_path: Path to ONNX model
            num_tests: Number of test cases to run

        Returns:
            Dictionary with verification results
        """
        # Load models
        self.load_torchscript_model(torchscript_path)
        self.load_onnx_model(onnx_path)

        results = {
            "pytorch_model": self.pytorch_model_path,
            "torchscript_path": torchscript_path,
            "onnx_path": onnx_path,
            "num_tests": num_tests,
            "torchscript_results": [],
            "onnx_results": [],
            "summary": {},
        }

        logger.info(f"Running {num_tests} verification tests...")

        for i in range(num_tests):
            # Create random input
            input_tensor = self.create_dummy_input()

            logger.info(f"Test {i+1}/{num_tests}")

            # Test TorchScript
            ts_result = self.verify_torchscript(input_tensor)
            results["torchscript_results"].append(ts_result)

            # Test ONNX
            onnx_result = self.verify_onnx(input_tensor)
            results["onnx_results"].append(onnx_result)

        # Calculate summary statistics
        self._calculate_summary(results)

        return results

    def _calculate_summary(self, results: Dict[str, Any]) -> None:
        """Calculate summary statistics from verification results."""
        # TorchScript summary
        ts_successes = sum(
            1 for r in results["torchscript_results"] if r.get("status") == "success"
        )
        ts_comparisons = [
            r["comparison"]
            for r in results["torchscript_results"]
            if r.get("status") == "success"
        ]

        if ts_comparisons:
            ts_max_diffs = [c["max_absolute_diff"] for c in ts_comparisons]
            ts_mean_diffs = [c["mean_absolute_diff"] for c in ts_comparisons]

            results["summary"]["torchscript"] = {
                "success_rate": ts_successes / len(results["torchscript_results"]),
                "max_absolute_diff": max(ts_max_diffs),
                "mean_absolute_diff": np.mean(ts_mean_diffs),
                "all_tests_passed": all(c["is_close"] for c in ts_comparisons),
            }
        else:
            results["summary"]["torchscript"] = {
                "success_rate": 0.0,
                "error": "No successful tests",
            }

        # ONNX summary
        onnx_successes = sum(
            1 for r in results["onnx_results"] if r.get("status") == "success"
        )
        onnx_comparisons = [
            r["comparison"]
            for r in results["onnx_results"]
            if r.get("status") == "success"
        ]

        if onnx_comparisons:
            onnx_max_diffs = [c["max_absolute_diff"] for c in onnx_comparisons]
            onnx_mean_diffs = [c["mean_absolute_diff"] for c in onnx_comparisons]

            results["summary"]["onnx"] = {
                "success_rate": onnx_successes / len(results["onnx_results"]),
                "max_absolute_diff": max(onnx_max_diffs),
                "mean_absolute_diff": np.mean(onnx_mean_diffs),
                "all_tests_passed": all(c["is_close"] for c in onnx_comparisons),
            }
        else:
            results["summary"]["onnx"] = {
                "success_rate": 0.0,
                "error": "No successful tests",
            }


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify exported ATC models")
    parser.add_argument(
        "--pytorch-model",
        type=str,
        required=True,
        help="Path to original PyTorch model",
    )
    parser.add_argument(
        "--torchscript-model", type=str, required=True, help="Path to TorchScript model"
    )
    parser.add_argument(
        "--onnx-model", type=str, required=True, help="Path to ONNX model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mask_rcnn_atc.yaml",
        help="Path to model configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/export_verification.txt",
        help="Output file for verification results",
    )
    parser.add_argument(
        "--num-tests", type=int, default=5, help="Number of verification tests to run"
    )

    args = parser.parse_args()

    # Initialize verifier
    verifier = ModelVerifier(args.pytorch_model, args.config)

    # Run verification
    results = verifier.run_verification(
        args.torchscript_model, args.onnx_model, args.num_tests
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        f.write("Model Export Verification Report\n")
        f.write("================================\n\n")

        f.write(f"PyTorch Model: {results['pytorch_model']}\n")
        f.write(f"TorchScript Model: {results['torchscript_path']}\n")
        f.write(f"ONNX Model: {results['onnx_path']}\n")
        f.write(f"Number of Tests: {results['num_tests']}\n\n")

        # TorchScript summary
        f.write("TorchScript Verification Summary:\n")
        f.write("--------------------------------\n")
        ts_summary = results["summary"]["torchscript"]
        f.write(f"Success Rate: {ts_summary['success_rate']:.2%}\n")
        if "error" not in ts_summary:
            f.write(f"Max Absolute Difference: {ts_summary['max_absolute_diff']:.6f}\n")
            f.write(
                f"Mean Absolute Difference: {ts_summary['mean_absolute_diff']:.6f}\n"
            )
            f.write(f"All Tests Passed: {ts_summary['all_tests_passed']}\n")
        else:
            f.write(f"Error: {ts_summary['error']}\n")
        f.write("\n")

        # ONNX summary
        f.write("ONNX Verification Summary:\n")
        f.write("-------------------------\n")
        onnx_summary = results["summary"]["onnx"]
        f.write(f"Success Rate: {onnx_summary['success_rate']:.2%}\n")
        if "error" not in onnx_summary:
            f.write(
                f"Max Absolute Difference: {onnx_summary['max_absolute_diff']:.6f}\n"
            )
            f.write(
                f"Mean Absolute Difference: {onnx_summary['mean_absolute_diff']:.6f}\n"
            )
            f.write(f"All Tests Passed: {onnx_summary['all_tests_passed']}\n")
        else:
            f.write(f"Error: {onnx_summary['error']}\n")
        f.write("\n")

        # Detailed results
        f.write("Detailed Test Results:\n")
        f.write("---------------------\n")
        for i, (ts_result, onnx_result) in enumerate(
            zip(results["torchscript_results"], results["onnx_results"])
        ):
            f.write(f"\nTest {i+1}:\n")
            f.write(f"  TorchScript: {ts_result.get('status', 'unknown')}\n")
            if ts_result.get("status") == "success":
                comp = ts_result["comparison"]
                f.write(f"    Max Diff: {comp['max_absolute_diff']:.6f}\n")
                f.write(f"    Is Close: {comp['is_close']}\n")

            f.write(f"  ONNX: {onnx_result.get('status', 'unknown')}\n")
            if onnx_result.get("status") == "success":
                comp = onnx_result["comparison"]
                f.write(f"    Max Diff: {comp['max_absolute_diff']:.6f}\n")
                f.write(f"    Is Close: {comp['is_close']}\n")

    logger.info(f"Verification results saved to {args.output}")

    # Print summary to console
    print("\nVerification Summary:")
    print("===================")
    print(
        f"TorchScript Success Rate: {results['summary']['torchscript']['success_rate']:.2%}"
    )
    print(f"ONNX Success Rate: {results['summary']['onnx']['success_rate']:.2%}")


if __name__ == "__main__":
    main()
