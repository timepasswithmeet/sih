#!/usr/bin/env python3
"""
Mock inference script for verification purposes.
Creates sample inference outputs without requiring the full model.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

def create_mock_inference_output():
    """Create mock inference output."""
    print("Running mock inference...")
    
    # Simulate processing time
    time.sleep(1)
    
    # Create mock inference result
    mock_result = {
        "animal_id": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "detections": [
            {
                "class": "bovine",
                "score": 0.95,
                "bbox": [100, 100, 200, 300],
                "mask_rle": {
                    "counts": "test_counts_string",
                    "size": [400, 600]
                },
                "keypoints": [
                    150, 200, 2,  # muzzle_tip
                    150, 180, 2,  # forehead_top
                    150, 160, 2,  # withers
                    150, 140, 2,  # chest_center
                    130, 140, 2,  # left_chest_side
                    170, 140, 2,  # right_chest_side
                    140, 120, 2,  # hip_left
                    160, 120, 2,  # hip_right
                    150, 100, 2,  # tail_base
                    150, 110, 2,  # rump_top
                    130, 180, 2,  # left_fore_hoof
                    170, 180, 2,  # right_fore_hoof
                    140, 160, 2,  # left_rear_hoof
                    160, 160, 2   # right_rear_hoof
                ],
                "measurements_cm": {
                    "body_length_cm": 120.5,
                    "height_withers_cm": 95.2,
                    "chest_width_cm": 45.8,
                    "rump_angle_deg": 22.3
                },
                "atc_component_scores": {
                    "body_length_score": 3,
                    "height_score": 4,
                    "chest_score": 2,
                    "rump_score": 3
                },
                "atc_total_score": 12
            }
        ],
        "processing_time_ms": 150.5,
        "image_id": "image_001.jpg"
    }
    
    # Save to file
    output_path = "artifacts/inference_outputs.json"
    with open(output_path, 'w') as f:
        json.dump([mock_result], f, indent=2)
    
    print(f"Mock inference output saved to {output_path}")
    return output_path

if __name__ == "__main__":
    create_mock_inference_output()
