#!/usr/bin/env python3
"""
Streamlit web application for ATC (Animal Type Classification).

Provides a user-friendly interface for uploading images, running inference,
and viewing results including measurements and ATC scores.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.inference import ATCInference

# Page configuration
st.set_page_config(
    page_title="ATC - Animal Type Classification",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_inference_model(model_path: str, config_path: str) -> ATCInference:
    """Load the inference model with caching."""
    try:
        return ATCInference(model_path, config_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def create_guidance_overlay() -> np.ndarray:
    """Create a guidance overlay image for user instructions."""
    # Create a transparent overlay
    overlay = np.zeros((600, 800, 4), dtype=np.uint8)

    # Draw animal silhouette guide
    cv2.ellipse(overlay, (400, 300), (120, 80), 0, 0, 360, (0, 255, 0, 100), -1)  # Body
    cv2.ellipse(overlay, (400, 200), (60, 50), 0, 0, 360, (0, 255, 0, 100), -1)  # Head

    # Draw reference marker guide
    cv2.rectangle(overlay, (50, 50), (150, 200), (255, 0, 0, 150), 2)  # A4 sheet guide

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        overlay, "Position animal here", (300, 150), font, 1, (0, 255, 0, 255), 2
    )
    cv2.putText(
        overlay,
        "Place reference marker (A4 sheet)",
        (20, 30),
        font,
        0.7,
        (255, 0, 0, 255),
        2,
    )

    return overlay


def display_measurements(measurements: Dict[str, float]) -> None:
    """Display measurements in a formatted way."""
    st.markdown(
        '<div class="sub-header">📏 Body Measurements</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Body Length",
            value=f"{measurements.get('body_length_cm', 0):.1f} cm",
            help="Distance from muzzle tip to tail base",
        )

        st.metric(
            label="Height at Withers",
            value=f"{measurements.get('height_withers_cm', 0):.1f} cm",
            help="Vertical height from ground to withers",
        )

    with col2:
        st.metric(
            label="Chest Width",
            value=f"{measurements.get('chest_width_cm', 0):.1f} cm",
            help="Horizontal width at chest level",
        )

        st.metric(
            label="Rump Angle",
            value=f"{measurements.get('rump_angle_deg', 0):.1f}°",
            help="Angle of rump from hip to tail",
        )


def display_atc_scores(atc_scores: Dict[str, Any]) -> None:
    """Display ATC scores with visual indicators."""
    st.markdown(
        '<div class="sub-header">🏆 ATC Component Scores</div>', unsafe_allow_html=True
    )

    component_scores = atc_scores.get("atc_component_scores", {})
    total_score = atc_scores.get("atc_total_score", 0)

    # Create score visualization
    fig = go.Figure()

    categories = ["Body Length", "Height", "Chest", "Rump"]
    scores = [
        component_scores.get("body_length_score", 0),
        component_scores.get("height_score", 0),
        component_scores.get("chest_score", 0),
        component_scores.get("rump_score", 0),
    ]

    # Color mapping for scores
    colors = ["#ff4444", "#ff8800", "#ffaa00", "#44aa44"]
    score_colors = [colors[min(score - 1, 3)] for score in scores]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=scores,
            marker_color=score_colors,
            text=scores,
            textposition="auto",
        )
    )

    fig.update_layout(
        title="ATC Component Scores",
        xaxis_title="Components",
        yaxis_title="Score (1-4)",
        yaxis=dict(range=[0, 5]),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display total score
    st.markdown(
        f"""
    <div class="metric-container">
        <h3>Total ATC Score: {total_score}/16</h3>
        <p>Score Range: 4-16 (4 = Poor, 16 = Excellent)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_detection_visualization(
    image: np.ndarray, detection: Dict[str, Any]
) -> None:
    """Display detection visualization with keypoints and mask."""
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding box
    bbox = detection.get("bbox", [0, 0, 0, 0])
    x, y, w, h = bbox
    cv2.rectangle(image_rgb, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    # Draw keypoints
    keypoints = detection.get("keypoints", [])
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x_kp, y_kp, score = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if score > 0:
                cv2.circle(image_rgb, (int(x_kp), int(y_kp)), 5, (0, 255, 0), -1)
                cv2.putText(
                    image_rgb,
                    str(i // 3),
                    (int(x_kp) + 5, int(y_kp) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

    # Display image
    st.image(image_rgb, caption="Detection Visualization", use_column_width=True)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown(
        '<h1 class="main-header">🐄 ATC - Animal Type Classification</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="info-box">
        <strong>Welcome to ATC!</strong> This application analyzes images of cattle and buffaloes to measure 
        body dimensions and calculate Animal Type Classification (ATC) scores. Upload an image with a 
        reference marker (A4 sheet or ArUco marker) for accurate measurements.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="artifacts/model.pt",
        help="Path to the trained model checkpoint",
    )

    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/mask_rcnn_atc.yaml",
        help="Path to the model configuration file",
    )

    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            inference_model = load_inference_model(model_path, config_path)
            if inference_model:
                st.sidebar.success("Model loaded successfully!")
                st.session_state.inference_model = inference_model
            else:
                st.sidebar.error("Failed to load model!")

    # Check if model is loaded
    if "inference_model" not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return

    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Analyze", "📊 Results", "ℹ️ Instructions"])

    with tab1:
        st.markdown(
            '<div class="sub-header">Upload Image for Analysis</div>',
            unsafe_allow_html=True,
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image of a cattle or buffalo with a reference marker",
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".jpg"
                        ) as tmp_file:
                            image.save(tmp_file.name)

                            # Run inference
                            result = (
                                st.session_state.inference_model.predict_single_image(
                                    tmp_file.name
                                )
                            )

                            # Store result in session state
                            st.session_state.analysis_result = result

                            # Clean up temp file
                            os.unlink(tmp_file.name)

                        st.success("Analysis completed successfully!")

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    with tab2:
        st.markdown(
            '<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True
        )

        if "analysis_result" in st.session_state:
            result = st.session_state.analysis_result

            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Processing Time", f"{result.get('processing_time_ms', 0):.1f} ms"
                )
            with col2:
                st.metric("Detections", len(result.get("detections", [])))
            with col3:
                st.metric("Image ID", result.get("image_id", "Unknown"))

            # Display detections
            detections = result.get("detections", [])
            if detections:
                for i, detection in enumerate(detections):
                    st.markdown(f"### Detection {i+1}")

                    # Display score and class
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Confidence Score", f"{detection.get('score', 0):.3f}"
                        )
                    with col2:
                        st.metric("Class", detection.get("class", "Unknown"))

                    # Display measurements
                    measurements = detection.get("measurements_cm", {})
                    if measurements:
                        display_measurements(measurements)

                    # Display ATC scores
                    atc_scores = {
                        "atc_component_scores": detection.get(
                            "atc_component_scores", {}
                        ),
                        "atc_total_score": detection.get("atc_total_score", 0),
                    }
                    if atc_scores["atc_component_scores"]:
                        display_atc_scores(atc_scores)

                    # Display visualization
                    if uploaded_file is not None:
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        if len(image_np.shape) == 3:
                            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                            display_detection_visualization(image_bgr, detection)

            # Display raw JSON
            with st.expander("📄 Raw JSON Output"):
                st.json(result)

            # Download results
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Results",
                data=json_str,
                file_name=f"atc_results_{result.get('image_id', 'unknown')}.json",
                mime="application/json",
            )

        else:
            st.info(
                "No analysis results available. Please upload and analyze an image first."
            )

    with tab3:
        st.markdown(
            '<div class="sub-header">📋 Instructions</div>', unsafe_allow_html=True
        )

        st.markdown(
            """
        ### How to Use ATC
        
        1. **Prepare Your Image:**
           - Take a clear photo of a cattle or buffalo from the side
           - Ensure the animal is fully visible in the frame
           - Place a reference marker (A4 sheet or ArUco marker) near the animal
        
        2. **Reference Marker Options:**
           - **A4 Sheet**: Place a standard A4 paper (210mm × 297mm) in the frame
           - **ArUco Marker**: Use a 50mm × 50mm ArUco marker (6x6_250 dictionary)
        
        3. **Image Requirements:**
           - Supported formats: JPG, JPEG, PNG, BMP
           - Minimum resolution: 640×480 pixels
           - Good lighting and contrast
           - Animal should be standing naturally
        
        4. **Analysis Process:**
           - Upload your image using the file uploader
           - Click "Analyze Image" to run the detection and measurement pipeline
           - View results including body measurements and ATC scores
        
        ### Understanding Results
        
        **Body Measurements:**
        - **Body Length**: Distance from muzzle tip to tail base
        - **Height at Withers**: Vertical height from ground to withers
        - **Chest Width**: Horizontal width at chest level
        - **Rump Angle**: Angle of rump from hip to tail
        
        **ATC Scores:**
        - Each component scored 1-4 (1=Poor, 4=Excellent)
        - Total score: 4-16
        - Based on industry standards for cattle evaluation
        
        ### Troubleshooting
        
        - **No detections**: Ensure animal is clearly visible and well-lit
        - **Inaccurate measurements**: Check reference marker placement and size
        - **Missing keypoints**: Animal pose may be unclear, try a different angle
        """
        )

        # Display guidance overlay
        st.markdown("### 📐 Positioning Guide")
        overlay = create_guidance_overlay()
        st.image(
            overlay,
            caption="Recommended positioning for best results",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
