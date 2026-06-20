# ============================================================================
# SIMPLE STREAMLIT INTERFACE - LOAD & USE SAVED MODEL
# ============================================================================
# Copy-paste this entire code into a single file: app.py
# Run with: streamlit run app.py
# ============================================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(page_title="MRI Tumor Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain MRI Tumor Classifier")
st.subheader("Upload an MRI image to get a prediction")

# ============================================================================
# LOAD MODEL (ONCE - CACHED)
# ============================================================================

@st.cache_resource
def load_model(model_path):
    """Load saved model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

# Sidebar - Model selection
st.sidebar.title("⚙️ Settings")

# Ask user for model path
model_path = st.sidebar.text_input(
    "Model Path",
    value="best_tumor_model.h5",
    help="Path to your saved .h5 model file"
)

# Class names
CLASSES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
IMG_SIZE = 256

# Load model
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.info("Please save your trained model as 'best_tumor_model.h5' first")
    st.stop()

model = load_model(model_path)

if model is None:
    st.stop()

st.sidebar.success(f"✅ Model loaded: {os.path.basename(model_path)}")

# ============================================================================
# FILE UPLOAD
# ============================================================================

st.markdown("---")
st.markdown("### 📂 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image (JPG, PNG, TIFF)",
    type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
    help="Upload a brain MRI scan"
)

if uploaded_file is not None:
    
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Process image
    st.markdown("---")
    st.markdown("### 🔍 Processing...")
    
    # Convert PIL to numpy
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    
    # Resize
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to RGB (required for ResNet50)
    img_rgb = np.repeat(img_normalized[np.newaxis, :, :, np.newaxis], 3, axis=-1)
    
    # Make prediction
    with st.spinner("Making prediction..."):
        prediction = model.predict(img_rgb, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx]) * 100
    
    st.markdown("---")
    st.markdown("### 🎯 PREDICTION RESULT")
    
    # Display main prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Predicted Class",
            CLASSES[predicted_class_idx],
            f"{confidence:.1f}%"
        )
    
    with col2:
        if confidence >= 80:
            st.metric("Confidence", "High ✅", "")
        elif confidence >= 60:
            st.metric("Confidence", "Medium ⚠️", "")
        else:
            st.metric("Confidence", "Low ❌", "")
    
    # Show confidence for all classes
    st.markdown("### 📊 Confidence Scores for All Classes")
    
    confidence_data = {
        CLASSES[i]: f"{prediction[0][i]*100:.1f}%"
        for i in range(len(CLASSES))
    }
    
    # Display as bar chart
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#1f77b4' if i == predicted_class_idx else '#d3d3d3' for i in range(len(CLASSES))]
    bars = ax.barh(CLASSES, prediction[0] * 100, color=colors)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_xlim([0, 100])
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed breakdown
    st.markdown("### 📋 Detailed Breakdown")
    
    breakdown = []
    for i, class_name in enumerate(CLASSES):
        conf = prediction[0][i] * 100
        breakdown.append({
            "Class": class_name,
            "Confidence": f"{conf:.2f}%",
            "Rank": i+1 if i == predicted_class_idx else ""
        })
    
    import pandas as pd
    df = pd.DataFrame(breakdown)
    df = df.sort_values('Confidence', ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Interpretation
    st.markdown("---")
    st.markdown("### 📌 Interpretation")
    
    if confidence >= 80:
        st.success(f"✅ High confidence prediction: **{CLASSES[predicted_class_idx]}**")
    elif confidence >= 60:
        st.warning(f"⚠️ Moderate confidence: **{CLASSES[predicted_class_idx]}**. Consider specialist review.")
    else:
        st.error(f"❌ Low confidence. Image may be unclear or unusual.")
    
    st.info("💡 Note: Always validate AI predictions with medical professionals before clinical use.")

else:
    st.info("👆 Upload an MRI image to get started")
    
    # Show example
    st.markdown("---")
    st.markdown("### 📚 How to use:")
    st.markdown("""
    1. **Prepare**: Have a brain MRI scan image (JPG, PNG, or TIFF)
    2. **Upload**: Click above to upload your image
    3. **Analyze**: Model automatically processes and predicts
    4. **Review**: Check confidence scores and class breakdown
    5. **Validate**: Always confirm with a medical professional
    
    ### 🎯 Expected Outputs:
    - **Predicted Class**: Glioma, Meningioma, Pituitary, or No Tumor
    - **Confidence Score**: How sure the model is (0-100%)
    - **All Scores**: Breakdown of all class probabilities
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85em;">
    <p>🧠 Brain MRI Tumor Classification | ResNet50 Transfer Learning</p>
    <p>Built with Streamlit & TensorFlow | For research and educational use</p>
</div>
""", unsafe_allow_html=True)
