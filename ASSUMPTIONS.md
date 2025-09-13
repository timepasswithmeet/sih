# Assumptions for ATC (Animal Type Classification) Project

## Data and Environment Assumptions
1. **Synthetic Dataset**: We'll generate 30 synthetic images with simple animal-like shapes (ellipses) to simulate cattle/buffaloes
2. **Reference Marker**: Using ArUco markers (6x6_250 dictionary) with 50mm width as the primary reference, with A4 sheet fallback
3. **Keypoint Order**: Following the exact 14-keypoint specification provided
4. **Model Architecture**: Using Mask R-CNN with ResNet-50-FPN backbone from Detectron2
5. **Python Version**: Python 3.10+ with PyTorch 2.0+

## Technical Assumptions
1. **Ground Estimation**: Using the lowest points of segmentation mask to estimate ground line
2. **Missing Keypoints**: Graceful fallback to mask-based heuristics when keypoints are missing
3. **Scale Conversion**: 1 pixel = 1mm when no reference marker is detected (with warning)
4. **ATC Scoring**: Rule-based scoring system with placeholder for trainable regressor
5. **Image Format**: Supporting common formats (JPG, PNG) with RGB color space

## Performance Assumptions
1. **Training**: Single GPU training with batch size 2 for synthetic data
2. **Inference**: CPU inference for deployment, GPU optional
3. **Memory**: 8GB RAM minimum for training, 4GB for inference
4. **Storage**: ~1GB for complete project including models and data

## Deployment Assumptions
1. **Container**: Docker-based deployment with multi-stage builds
2. **Web Interface**: Streamlit for user interaction
3. **API**: JSON-based output format as specified
4. **CI/CD**: GitHub Actions for automated testing and deployment
