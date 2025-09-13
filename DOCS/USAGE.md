# ATC Usage Guide

## Getting Started

This guide provides step-by-step instructions for using the Animal Type Classification (ATC) system for analyzing cattle and buffalo images.

## Prerequisites

Before using the system, ensure you have:

- Python 3.10 or higher
- 4GB+ RAM
- GPU (recommended for training)
- Web browser (for web application)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd atc-cattle-buffalo
```

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v
```

## Quick Start

### Option 1: Web Application (Recommended)

1. **Generate synthetic data:**
```bash
python src/data/generate_synthetic_data.py --num-images 30
```

2. **Train a model:**
```bash
python src/train.py --config configs/mask_rcnn_atc.yaml --epochs 5
```

3. **Launch web application:**
```bash
streamlit run src/app/app.py
```

4. **Open browser and navigate to:** `http://localhost:8501`

### Option 2: Command Line Interface

1. **Generate data and train model** (same as above)

2. **Run inference:**
```bash
python src/inference.py \
    --model artifacts/model.pt \
    --images data/synthetic/images/ \
    --output results.json
```

## Detailed Usage

### Data Generation

The system includes a synthetic data generator for testing and development:

```bash
python src/data/generate_synthetic_data.py \
    --num-images 50 \
    --output-dir data/custom_dataset
```

**Parameters:**
- `--num-images`: Number of images to generate (default: 30)
- `--output-dir`: Output directory (default: data/synthetic)

**Output:**
- `images/`: Directory containing generated images
- `annotations.json`: COCO format annotations

### Model Training

Train the ATC model using Detectron2:

```bash
python src/train.py \
    --config configs/mask_rcnn_atc.yaml \
    --data-dir data/synthetic \
    --output-dir artifacts \
    --epochs 10
```

**Parameters:**
- `--config`: Path to configuration file
- `--data-dir`: Path to dataset directory
- `--output-dir`: Output directory for checkpoints
- `--epochs`: Number of training epochs
- `--resume`: Resume training from last checkpoint

**Output:**
- `model_final.pth`: Final trained model
- `train_log.txt`: Training logs
- `metrics.json`: Training metrics

### Model Inference

Run inference on images:

```bash
python src/inference.py \
    --model artifacts/model.pt \
    --config configs/mask_rcnn_atc.yaml \
    --images path/to/images/ \
    --output results.json \
    --visualize
```

**Parameters:**
- `--model`: Path to trained model
- `--config`: Path to configuration file
- `--images`: Path to image or directory
- `--output`: Output JSON file
- `--visualize`: Generate visualization images
- `--viz-dir`: Directory for visualizations

### Model Export

Export trained models for deployment:

```bash
python scripts/export_model.py \
    --model artifacts/model.pt \
    --config configs/mask_rcnn_atc.yaml \
    --output-dir artifacts/exported
```

**Output:**
- `model_ts.pt`: TorchScript model
- `model.onnx`: ONNX model
- `export_info.txt`: Export information

### Model Verification

Verify exported models:

```bash
python scripts/verify_export.py \
    --pytorch-model artifacts/model.pt \
    --torchscript-model artifacts/exported/model_ts.pt \
    --onnx-model artifacts/exported/model.onnx
```

## Web Application Usage

### Interface Overview

The web application provides three main tabs:

1. **Upload & Analyze**: Upload images and run analysis
2. **Results**: View detailed results and visualizations
3. **Instructions**: Usage guidelines and positioning tips

### Step-by-Step Process

#### 1. Load Model

- Enter model path in sidebar (default: `artifacts/model.pt`)
- Enter config path (default: `configs/mask_rcnn_atc.yaml`)
- Click "Load Model" button
- Wait for "Model loaded successfully!" message

#### 2. Upload Image

- Click "Choose an image file" button
- Select image file (JPG, PNG, BMP supported)
- Image will be displayed in the interface

#### 3. Analyze Image

- Click "🔍 Analyze Image" button
- Wait for analysis to complete
- Results will appear in the "Results" tab

#### 4. View Results

**Basic Information:**
- Processing time
- Number of detections
- Image ID

**For Each Detection:**
- Confidence score
- Animal class
- Body measurements (cm)
- ATC component scores (1-4)
- Total ATC score (4-16)
- Visualization with keypoints and mask

#### 5. Export Results

- Click "📥 Download Results" button
- JSON file will be downloaded
- Contains complete analysis results

### Image Requirements

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

**Image Quality:**
- Minimum resolution: 640×480 pixels
- Good lighting and contrast
- Clear view of the animal
- Reference marker visible

**Positioning Guidelines:**
- Animal should be standing naturally
- Side view preferred
- Full body visible in frame
- Reference marker placed near animal

## Reference Markers

### ArUco Markers

**Specifications:**
- Size: 50mm × 50mm
- Dictionary: 6x6_250
- Print on white paper
- Place near the animal

**Advantages:**
- High accuracy
- Robust detection
- Standardized size

### A4 Sheets

**Specifications:**
- Size: 210mm × 297mm
- White paper
- Place flat near animal

**Advantages:**
- Easy to obtain
- Large size for visibility
- Fallback option

### Marker Placement

**Best Practices:**
- Place on ground level
- Ensure full visibility
- Avoid shadows or reflections
- Keep marker flat and unwrinkled

## Understanding Results

### Body Measurements

**Body Length (cm):**
- Distance from muzzle tip to tail base
- Indicates overall body size
- Used for growth assessment

**Height at Withers (cm):**
- Vertical height from ground to withers
- Indicates body height
- Important for breed standards

**Chest Width (cm):**
- Horizontal width at chest level
- Indicates body depth
- Related to muscle development

**Rump Angle (degrees):**
- Angle between hip, rump, and tail
- Indicates body conformation
- Affects breeding value

### ATC Scoring System

**Component Scores (1-4):**
- **4**: Excellent (above industry standards)
- **3**: Good (meets industry standards)
- **2**: Fair (below average)
- **1**: Poor (well below standards)

**Total Score (4-16):**
- Sum of all component scores
- Higher scores indicate better conformation
- Used for breeding and selection decisions

### Score Interpretation

**Excellent (13-16):**
- Superior conformation
- Ideal for breeding
- High commercial value

**Good (9-12):**
- Above average conformation
- Suitable for breeding
- Good commercial value

**Fair (5-8):**
- Average conformation
- Acceptable for production
- Moderate commercial value

**Poor (4):**
- Below average conformation
- Not recommended for breeding
- Lower commercial value

## Troubleshooting

### Common Issues

**No detections found:**
- Check image quality and lighting
- Ensure animal is clearly visible
- Verify model is loaded correctly

**Inaccurate measurements:**
- Check reference marker placement
- Ensure marker is fully visible
- Verify marker size and type

**Missing keypoints:**
- Animal pose may be unclear
- Try different camera angle
- Ensure full body is visible

**Model loading errors:**
- Check model file path
- Verify model file exists
- Ensure sufficient memory

### Error Messages

**"Failed to load model":**
- Check model path in sidebar
- Verify model file exists
- Check file permissions

**"Analysis failed":**
- Check image format and quality
- Ensure sufficient memory
- Try with different image

**"No reference marker detected":**
- Check marker placement
- Ensure marker is visible
- Try different marker type

### Performance Issues

**Slow inference:**
- Use GPU if available
- Reduce image resolution
- Close other applications

**Memory errors:**
- Reduce batch size
- Use smaller images
- Close other applications

## Advanced Usage

### Batch Processing

Process multiple images:

```bash
python src/inference.py \
    --model artifacts/model.pt \
    --images data/test_images/ \
    --output batch_results.json
```

### Custom Configuration

Modify training parameters:

```yaml
# configs/custom_config.yaml
SOLVER:
  BASE_LR: 0.0001
  MAX_ITER: 5000
  IMS_PER_BATCH: 4
```

### Docker Deployment

Run with Docker:

```bash
# Build image
docker build -t atc-cattle-buffalo .

# Run container
docker run -p 8501:8501 atc-cattle-buffalo
```

### API Integration

Use the inference module in your code:

```python
from src.inference import ATCInference

# Initialize inference
inference = ATCInference("artifacts/model.pt")

# Run inference
result = inference.predict_single_image("image.jpg")

# Process results
for detection in result["detections"]:
    measurements = detection["measurements_cm"]
    scores = detection["atc_component_scores"]
    print(f"Body length: {measurements['body_length_cm']:.1f} cm")
    print(f"Total score: {detection['atc_total_score']}")
```

## Best Practices

### Image Capture

1. **Lighting**: Use natural light or bright artificial lighting
2. **Background**: Plain background preferred
3. **Distance**: Fill frame with animal while keeping marker visible
4. **Angle**: Side view at animal's shoulder height
5. **Stability**: Use tripod or stable surface

### Data Management

1. **Organization**: Keep images organized by date/session
2. **Backup**: Regular backup of important data
3. **Versioning**: Track model versions and results
4. **Documentation**: Record analysis parameters

### Quality Control

1. **Validation**: Check results for reasonableness
2. **Calibration**: Regular marker size verification
3. **Consistency**: Use same setup for comparable results
4. **Review**: Manual review of edge cases

## Support and Resources

### Documentation

- **README.md**: Project overview and setup
- **DOCS/architecture.md**: Technical architecture details
- **API Reference**: Code documentation and examples

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Contributions**: Submit improvements and fixes

### Updates

- **Version History**: Track changes and improvements
- **Release Notes**: New features and bug fixes
- **Migration Guide**: Upgrade instructions

## Conclusion

The ATC system provides a comprehensive solution for animal type classification and measurement. By following this guide, you can effectively use the system for research, breeding, and commercial applications. For additional support or questions, please refer to the documentation or contact the development team.
