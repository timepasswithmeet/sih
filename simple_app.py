#!/usr/bin/env python3
"""
Simple Streamlit app for ATC demonstration.
"""

import streamlit as st
import json
import os
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="ATC - Animal Type Classification",
    page_icon="🐄",
    layout="wide",
)

def main():
    """Main Streamlit application."""
    # Header
    st.title("🐄 ATC - Animal Type Classification")
    
    st.info("""
    **Welcome to ATC!** This application analyzes images of cattle and buffaloes to measure 
    body dimensions and calculate Animal Type Classification (ATC) scores. Upload an image with a 
    reference marker (A4 sheet or ArUco marker) for accurate measurements.
    """)

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Check if we have mock results
    mock_results_path = "artifacts/inference_outputs.json"
    if os.path.exists(mock_results_path):
        st.sidebar.success("✅ Mock results available")
    else:
        st.sidebar.warning("⚠️ No mock results found")

    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Analyze", "📊 Results", "ℹ️ Instructions"])

    with tab1:
        st.subheader("Upload Image for Analysis")
        
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
            if st.button("🔍 Analyze Image (Mock)", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Load mock results
                        if os.path.exists(mock_results_path):
                            with open(mock_results_path, 'r') as f:
                                result = json.load(f)
                            
                            # Store result in session state
                            st.session_state.analysis_result = result
                            st.success("Analysis completed successfully! (Using mock data)")
                        else:
                            # Create a simple mock result
                            mock_result = {
                                "image_id": "demo_001",
                                "processing_time_ms": 150.5,
                                "detections": [
                                    {
                                        "bbox": [100, 100, 200, 300],
                                        "score": 0.95,
                                        "class": "cattle",
                                        "keypoints": [150, 120, 1.0, 200, 150, 1.0, 180, 200, 1.0],
                                        "measurements_cm": {
                                            "body_length_cm": 180.5,
                                            "height_withers_cm": 140.2,
                                            "chest_width_cm": 45.8,
                                            "rump_angle_deg": 12.3
                                        },
                                        "atc_component_scores": {
                                            "body_length_score": 3,
                                            "height_score": 4,
                                            "chest_score": 3,
                                            "rump_score": 2
                                        },
                                        "atc_total_score": 12
                                    }
                                ]
                            }
                            st.session_state.analysis_result = mock_result
                            st.success("Analysis completed successfully! (Using generated mock data)")
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    with tab2:
        st.subheader("Analysis Results")

        if "analysis_result" in st.session_state:
            result = st.session_state.analysis_result

            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.1f} ms")
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
                        st.metric("Confidence Score", f"{detection.get('score', 0):.3f}")
                    with col2:
                        st.metric("Class", detection.get("class", "Unknown"))

                    # Display measurements
                    measurements = detection.get("measurements_cm", {})
                    if measurements:
                        st.markdown("#### 📏 Body Measurements")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Body Length", f"{measurements.get('body_length_cm', 0):.1f} cm")
                            st.metric("Height at Withers", f"{measurements.get('height_withers_cm', 0):.1f} cm")
                        
                        with col2:
                            st.metric("Chest Width", f"{measurements.get('chest_width_cm', 0):.1f} cm")
                            st.metric("Rump Angle", f"{measurements.get('rump_angle_deg', 0):.1f}°")

                    # Display ATC scores
                    atc_scores = detection.get("atc_component_scores", {})
                    total_score = detection.get("atc_total_score", 0)
                    
                    if atc_scores:
                        st.markdown("#### 🏆 ATC Component Scores")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Body Length", atc_scores.get("body_length_score", 0))
                        with col2:
                            st.metric("Height", atc_scores.get("height_score", 0))
                        with col3:
                            st.metric("Chest", atc_scores.get("chest_score", 0))
                        with col4:
                            st.metric("Rump", atc_scores.get("rump_score", 0))
                        
                        st.markdown(f"**Total ATC Score: {total_score}/16**")

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
            st.info("No analysis results available. Please upload and analyze an image first.")

    with tab3:
        st.subheader("📋 Instructions")
        
        st.markdown("""
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
        """)

if __name__ == "__main__":
    main()
