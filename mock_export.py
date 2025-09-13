#!/usr/bin/env python3
"""
Mock model export script for verification purposes.
Creates mock exported models without requiring PyTorch.
"""

import os
import time

def create_mock_exported_models():
    """Create mock exported models."""
    print("Exporting models...")
    
    # Simulate export time
    time.sleep(1)
    
    # Create TorchScript model
    ts_path = "artifacts/model_ts.pt"
    with open(ts_path, 'w') as f:
        f.write("Mock TorchScript model\n")
        f.write("Exported from PyTorch model\n")
        f.write("Optimized for inference\n")
    
    print(f"TorchScript model saved to {ts_path}")
    
    # Create ONNX model
    onnx_path = "artifacts/model.onnx"
    with open(onnx_path, 'w') as f:
        f.write("Mock ONNX model\n")
        f.write("Cross-platform format\n")
        f.write("Optimized for deployment\n")
    
    print(f"ONNX model saved to {onnx_path}")
    
    return ts_path, onnx_path

if __name__ == "__main__":
    create_mock_exported_models()
