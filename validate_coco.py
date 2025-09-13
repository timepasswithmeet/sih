#!/usr/bin/env python3
"""
Simple COCO validation script.
"""

import json
import os

def validate_coco():
    """Validate COCO annotations."""
    annotations_path = "data/synthetic/annotations.json"
    
    if not os.path.exists(annotations_path):
        print("Annotations file not found")
        return False
    
    try:
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Basic validation
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in coco_data:
                print(f"Missing required key: {key}")
                return False
        
        # Count validation
        num_images = len(coco_data["images"])
        num_annotations = len(coco_data["annotations"])
        num_categories = len(coco_data["categories"])
        
        print(f"COCO validation successful")
        print(f"Images: {num_images}")
        print(f"Annotations: {num_annotations}")
        print(f"Categories: {num_categories}")
        
        # Validate structure
        for image in coco_data["images"]:
            required_image_keys = ["id", "width", "height", "file_name"]
            for key in required_image_keys:
                if key not in image:
                    print(f"Image missing key: {key}")
                    return False
        
        for annotation in coco_data["annotations"]:
            required_annotation_keys = ["id", "image_id", "category_id", "keypoints"]
            for key in required_annotation_keys:
                if key not in annotation:
                    print(f"Annotation missing key: {key}")
                    return False
        
        print("All validations passed")
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False

if __name__ == "__main__":
    validate_coco()
