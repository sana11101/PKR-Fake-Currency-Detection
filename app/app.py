# =============================================================
# app/app.py
# WRITTEN BY: Aqsa
# PURPOSE: Streamlit web application for PKR Fake Currency Detection
#          Users upload a note image and get:
#          - Real or Fake prediction
#          - Confidence percentage
#          - Grad-CAM heatmap showing suspicious region
#          - Model comparison chart
# =============================================================

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
from PIL import Image
import io

# Add parent directory to path so we can import from gradcam folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradcam.visualize import generate_gradcam, overlay_heatmap

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="PKR Fake Currency Detector",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Custom CSS Styling
# ------------------------------------------------------------------

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1a5276;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-real {
        background-color: #d5f5e3;
        border: 3px solid #27ae60;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e8449;
    }
    .result-fake {
        background-color: #fadbd8;
        border: 3px solid #e74c3c;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        color: #c0392b;
    }
    .info-box {
        background-color: #eaf4fb;
        border-left: 5px solid #3498db;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load CNN Model (cached so it only loads once)
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Loads the trained CNN model.
    @st.cache_resource means this only runs once — not on every click.
    """
    model_path = "model/saved/cnn_model.h5"

    if not os.path.exists(model_path):
        return None

    model = tf.keras.models.load_model(model_path)
    return model

# ------------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------------

def preprocess_uploaded_image(uploaded_file):
    """
    Converts a Streamlit uploaded file to numpy array for CNN.
    Returns: (preprocessed array for CNN, original RGB image)
    """

    # Read uploaded file as PIL Image
    pil_image = Image.open(uploaded_file).convert('RGB')

    # Convert to numpy array
    original = np.array(pil_image)

    # Resize for CNN
    resized = cv2.resize(original, (224, 224))

    # Normalize
    normalized = resized / 255.0

    # Add batch dimension
    input_array = np.expand_dims(normalized, axis=0).astype(np.float32)

    return input_array, original

# ------------------------------------------------------------------
# Convert matplotlib figure to image for Streamlit display
# ------------------------------------------------------------------

def fig_to_image(fig):
    """Converts a matplotlib figure to a PIL Image for Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

# ------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------

def main():

    # ------ HEADER ------
    st.markdown('<p class="main-title">🇵🇰 PKR Fake Currency Detector</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered detection using CNN + Grad-CAM Explainability</p>',
                unsafe_allow_html=True)

    st.divider()

    # ------ SIDEBAR ------
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/32/Flag_of_Pakistan.svg",
                 width=100)
        st.markdown("## About This Project")
        st.markdown("""
        **University Semester Project**

        Detects fake Pakistani Rupee notes using:
        - 🧠 CNN (Main Model)
        - 📍 Grad-CAM Heatmap
        - 📊 KNN Comparison
        - 📉 Logistic Regression Comparison

        **Team:**
        - Sana — Model & Dataset
        - Aqsa — UI & Grad-CAM
        """)

        st.divider()
        st.markdown("### Supported Notes")
        st.markdown("- 100 PKR\n- 500 PKR\n- 1000 PKR")

        st.divider()
        st.markdown("### How to Use")
        st.markdown("1. Upload a clear photo of a PKR note\n2. Click **Analyze Note**\n3. View result + heatmap")

    # ------ LOAD MODEL ------
    model = load_model()

    if model is None:
        st.error("⚠️ CNN model not found! Please train the model first.")
        st.code("python model/train_cnn.py", language="bash")
        return

    # ------ FILE UPLOAD ------
    st.markdown("### 📤 Upload a PKR Note Image")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose an image of a Pakistani Rupee note",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit photo of the note"
        )

    # ------ INFO BOX ------
    st.markdown("""
    <div class="info-box">
    💡 <b>Tips for best results:</b>
    Place the note on a flat surface with good lighting.
    Make sure the full note is visible in the photo.
    </div>
    """, unsafe_allow_html=True)

    # ------ ANALYZE BUTTON ------
    if uploaded_file is not None:

        st.divider()

        # Show uploaded image preview
        col_prev1, col_prev2, col_prev3 = st.columns([1, 2, 1])
        with col_prev2:
            st.markdown("#### 🖼️ Uploaded Image Preview")
            st.image(uploaded_file, use_column_width=True)

        # Analyze button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analyze_clicked = st.button(
                "🔍 Analyze Note",
                type="primary",
                use_container_width=True
            )

        if analyze_clicked:

            with st.spinner("🧠 Analyzing note with CNN..."):

                # Preprocess image
                image_array, original_image = preprocess_uploaded_image(uploaded_file)

                # Get CNN prediction
                prediction_value = model.predict(image_array, verbose=0)[0][0]

                if prediction_value >= 0.5:
                    label      = "FAKE"
                    confidence = float(prediction_value * 100)
                else:
                    label      = "REAL"
                    confidence = float((1 - prediction_value) * 100)

            st.divider()

            # ------ RESULT DISPLAY ------
            st.markdown("## 📊 Analysis Results")

            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                if label == "FAKE":
                    st.markdown(
                        f'<div class="result-fake">❌ FAKE NOTE DETECTED<br>'
                        f'<span style="font-size:1rem">Confidence: {confidence:.1f}%</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-real">✅ REAL NOTE<br>'
                        f'<span style="font-size:1rem">Confidence: {confidence:.1f}%</span></div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ------ GRAD-CAM HEATMAP ------
            with st.spinner("🔥 Generating Grad-CAM heatmap..."):

                try:
                    heatmap = generate_gradcam(model, image_array, layer_name='conv4')
                    heatmap_overlay = overlay_heatmap(heatmap, original_image)

                    st.markdown("### 🔥 Grad-CAM Explainability")
                    st.markdown(
                        "The heatmap shows **which part of the note** the CNN focused on. "
                        "🔴 Red/Yellow = Most suspicious regions."
                    )

                    col_h1, col_h2 = st.columns(2)

                    with col_h1:
                        st.markdown("**Original Note**")
                        st.image(original_image, use_column_width=True)

                    with col_h2:
                        st.markdown(f"**Grad-CAM Heatmap — {label}**")
                        st.image(heatmap_overlay, use_column_width=True)

                except Exception as e:
                    st.warning(f"Grad-CAM visualization unavailable: {str(e)}")

            # ------ MODEL COMPARISON ------
            st.divider()
            st.markdown("### 📈 Model Comparison")
            st.markdown("Comparison of CNN with traditional ML models on the same dataset:")

            # Load saved comparison chart
            comparison_path = "model/saved/model_comparison.png"
            if os.path.exists(comparison_path):
                st.image(comparison_path, use_column_width=True)
            else:
                st.info("📊 Run `python model/train_comparison.py` to generate comparison chart.")

                # Show placeholder chart
                fig, ax = plt.subplots(figsize=(8, 4))
                models_names = ['Logistic\nRegression', 'KNN\n(K=5)', 'CNN']
                placeholder   = [65, 72, 94]
                colors         = ['#2ecc71', '#3498db', '#e74c3c']
                bars = ax.bar(models_names, placeholder, color=colors,
                              width=0.5, edgecolor='black')
                for bar, acc in zip(bars, placeholder):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.5,
                            f'{acc}%*', ha='center', fontweight='bold')
                ax.set_title('Model Accuracy Comparison (Placeholder)', fontsize=13)
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 110)
                ax.grid(axis='y', alpha=0.3)
                fig.text(0.5, -0.05, '*Placeholder values — train models for actual results',
                         ha='center', color='gray', fontsize=9)
                st.pyplot(fig)
                plt.close()

            # ------ CONCLUSION ------
            st.divider()
            st.markdown("### 🧠 Why CNN Performs Best")
            st.markdown("""
            | Model | Why It Works / Doesn't |
            |---|---|
            | **CNN** ⭐ | Learns spatial features (watermarks, threads, patterns) directly from pixels |
            | **KNN** | Compares pixel-by-pixel similarity — misses spatial relationships |
            | **Logistic Regression** | Linear model — too simple for complex image patterns |

            > CNN's ability to learn **hierarchical visual features** makes it ideal for detecting subtle forgery signs in currency notes.
            """)

# ------------------------------------------------------------------
# Run the app
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
