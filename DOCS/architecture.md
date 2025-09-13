# ATC Architecture Documentation

## System Overview

The Animal Type Classification (ATC) system is a comprehensive machine learning pipeline designed to analyze images of cattle and buffaloes, extract body measurements, and generate standardized scoring. The system combines computer vision, deep learning, and measurement science to provide accurate and consistent animal evaluation.

## Architecture Components

### 1. Data Generation Layer

**Purpose**: Generate synthetic training data for model development and testing.

**Components**:
- `src/data/generate_synthetic_data.py`: Synthetic dataset generator
- Creates animal-like shapes using computer graphics
- Generates COCO format annotations with segmentation masks
- Includes 14 keypoint annotations per animal
- Adds reference markers (ArUco or A4 sheet simulation)

**Key Features**:
- Reproducible data generation with fixed random seeds
- Configurable number of images and complexity
- Realistic animal proportions and poses
- Reference marker placement for scale calibration

### 2. Model Training Layer

**Purpose**: Train deep learning models for instance segmentation and keypoint detection.

**Components**:
- `src/train.py`: Training script using Detectron2
- `configs/mask_rcnn_atc.yaml`: Model configuration
- Mask R-CNN with ResNet-50-FPN backbone
- Custom keypoint head for 14 anatomical points

**Architecture Details**:
```
Input Image (3, H, W)
    ↓
ResNet-50 Backbone
    ↓
Feature Pyramid Network (FPN)
    ↓
Region Proposal Network (RPN)
    ↓
ROI Heads:
├── Box Head (classification + regression)
├── Mask Head (segmentation)
└── Keypoint Head (14 keypoints)
    ↓
Outputs:
├── Bounding boxes
├── Segmentation masks
└── Keypoint coordinates
```

**Training Process**:
1. Data loading and augmentation
2. Forward pass through network
3. Loss calculation (classification, regression, segmentation, keypoints)
4. Backward pass and optimization
5. Model checkpointing and validation

### 3. Inference Layer

**Purpose**: Run trained models on new images and extract predictions.

**Components**:
- `src/inference.py`: Main inference script
- `ATCInference` class: Model loading and prediction
- Batch processing support
- Visualization capabilities

**Inference Pipeline**:
```
Input Image
    ↓
Preprocessing (resize, normalize)
    ↓
Model Forward Pass
    ↓
Post-processing:
├── NMS (Non-Maximum Suppression)
├── Confidence filtering
└── Output formatting
    ↓
Detection Results
```

### 4. Measurement Pipeline

**Purpose**: Convert model predictions into physical measurements and scores.

**Components**:
- `src/measurement/core.py`: Core measurement logic
- `ReferenceMarkerDetector`: Scale calibration
- `BodyMeasurements`: Physical measurement calculation
- `ATCScorer`: Scoring system implementation

**Measurement Process**:
```
Detection Results
    ↓
Reference Marker Detection:
├── ArUco marker detection
├── A4 sheet detection
└── Scale factor calculation
    ↓
Ground Line Estimation
    ↓
Body Measurements:
├── Body length (muzzle to tail)
├── Height at withers
├── Chest width
└── Rump angle
    ↓
ATC Scoring (1-4 per component)
    ↓
Final Results
```

### 5. Web Application Layer

**Purpose**: Provide user-friendly interface for image analysis.

**Components**:
- `src/app/app.py`: Streamlit web application
- Interactive image upload and analysis
- Real-time visualization of results
- Results export and download

**Application Features**:
- File upload with format validation
- Model loading and caching
- Progress indicators and error handling
- Interactive visualizations with Plotly
- JSON result export

### 6. Model Export Layer

**Purpose**: Export trained models for deployment and optimization.

**Components**:
- `scripts/export_model.py`: Model export script
- `scripts/verify_export.py`: Export verification
- Support for TorchScript and ONNX formats

**Export Process**:
```
PyTorch Model
    ↓
TorchScript Export:
├── Model tracing
├── Optimization
└── Serialization
    ↓
ONNX Export:
├── Graph conversion
├── Operator mapping
└── Serialization
    ↓
Verification:
├── Output comparison
├── Accuracy validation
└── Performance testing
```

## Data Flow

### Training Data Flow
```
Synthetic Data Generator
    ↓
COCO Format Annotations
    ↓
Detectron2 DataLoader
    ↓
Model Training
    ↓
Checkpoint Saving
```

### Inference Data Flow
```
Input Image
    ↓
Model Inference
    ↓
Detection Results
    ↓
Measurement Pipeline
    ↓
ATC Scoring
    ↓
JSON Output
```

### Web Application Data Flow
```
User Upload
    ↓
Image Processing
    ↓
Model Inference
    ↓
Result Visualization
    ↓
User Interface
```

## Key Design Decisions

### 1. Synthetic Data Approach
- **Rationale**: Enables rapid prototyping and testing without real animal data
- **Benefits**: Controlled environment, reproducible results, no privacy concerns
- **Limitations**: May not capture all real-world variations

### 2. Detectron2 Framework
- **Rationale**: State-of-the-art instance segmentation with keypoint support
- **Benefits**: Pre-trained models, active development, comprehensive features
- **Considerations**: Complex configuration, large model size

### 3. 14-Keypoint System
- **Rationale**: Covers essential anatomical landmarks for measurement
- **Benefits**: Sufficient for body measurements, manageable complexity
- **Trade-offs**: May miss some detailed anatomical features

### 4. Reference Marker System
- **Rationale**: Enables accurate scale conversion from pixels to millimeters
- **Benefits**: Flexible marker types, fallback options
- **Considerations**: Requires user cooperation for marker placement

### 5. Rule-based Scoring
- **Rationale**: Provides interpretable and consistent scoring
- **Benefits**: Transparent logic, industry-standard approach
- **Future**: Can be replaced with learned scoring models

## Performance Considerations

### Computational Requirements
- **Training**: GPU recommended (8GB+ VRAM)
- **Inference**: CPU sufficient for single images
- **Memory**: 4GB+ RAM for full pipeline

### Optimization Strategies
- **Model Quantization**: Reduce model size and inference time
- **Batch Processing**: Process multiple images simultaneously
- **Caching**: Cache model loading and preprocessing
- **Export Formats**: Use optimized formats (TorchScript, ONNX)

### Scalability
- **Horizontal Scaling**: Multiple inference instances
- **Batch Processing**: Process image batches
- **Model Serving**: Dedicated model serving infrastructure

## Security and Privacy

### Data Handling
- **No Persistent Storage**: Images processed in memory
- **Temporary Files**: Automatic cleanup of temporary files
- **No Data Collection**: No user data stored or transmitted

### Model Security
- **Local Processing**: All inference runs locally
- **No External Calls**: No data sent to external services
- **Open Source**: Transparent model and code

## Monitoring and Logging

### Logging Strategy
- **Application Logs**: User actions and system events
- **Model Logs**: Training progress and metrics
- **Error Logs**: Exception handling and debugging

### Metrics Tracking
- **Performance Metrics**: Inference time, accuracy
- **Usage Metrics**: Image processing counts
- **Error Metrics**: Failure rates and types

## Future Enhancements

### Model Improvements
- **Real Data Training**: Train on actual animal images
- **Multi-class Support**: Support for different animal types
- **3D Measurements**: Add depth estimation capabilities

### Pipeline Enhancements
- **Automated Scoring**: Machine learning-based scoring
- **Batch Processing**: Web interface for multiple images
- **API Endpoints**: REST API for integration

### User Experience
- **Mobile Support**: Responsive design for mobile devices
- **Offline Mode**: Local processing without internet
- **Advanced Visualization**: 3D visualization of measurements

## Dependencies and Technologies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Detectron2**: Computer vision library
- **OpenCV**: Image processing
- **Streamlit**: Web application framework

### Supporting Libraries
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image handling
- **PyYAML**: Configuration management
- **Plotly**: Interactive visualizations

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Type checking
- **Docker**: Containerization

## Conclusion

The ATC system architecture provides a robust, scalable foundation for animal type classification and measurement. The modular design enables easy extension and modification, while the comprehensive testing and documentation ensure reliability and maintainability. The system successfully combines cutting-edge computer vision techniques with practical measurement science to deliver accurate and consistent results.
