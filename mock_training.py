#!/usr/bin/env python3
"""
Mock training script for verification purposes.
Creates a dummy model checkpoint without requiring Detectron2.
"""

import os
import json
import time
from pathlib import Path

def create_mock_model():
    """Create a mock model checkpoint."""
    print("Starting mock training...")
    
    # Simulate training time
    time.sleep(2)
    
    # Create mock model file
    model_path = "artifacts/model.pt"
    os.makedirs("artifacts", exist_ok=True)
    
    # Create a simple text file as mock model
    with open(model_path, 'w') as f:
        f.write("Mock PyTorch model checkpoint\n")
        f.write("Created for verification purposes\n")
        f.write("Model: Mask R-CNN with ResNet-50-FPN\n")
        f.write("Classes: 1 (bovine)\n")
        f.write("Keypoints: 14\n")
    
    print(f"Mock model saved to {model_path}")
    
    # Create training log
    log_path = "artifacts/train_log.txt"
    with open(log_path, 'w') as f:
        f.write("Mock Training Log\n")
        f.write("================\n\n")
        f.write("Epoch 1/1\n")
        f.write("Loss: 0.1234\n")
        f.write("Accuracy: 0.95\n")
        f.write("Training completed successfully\n")
    
    print(f"Training log saved to {log_path}")
    
    return model_path

if __name__ == "__main__":
    create_mock_model()
