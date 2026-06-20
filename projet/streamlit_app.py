import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt

# Define FeatureExtractor class (needed to load the pickled object)
class FeatureExtractor:
    """
    Extract handcrafted features from medical images
    """
    def __init__(self):
        pass
    
    def extract_glcm_features(self, image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Extract Gray-Level Co-occurrence Matrix (GLCM) features"""
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        glcm = graycomatrix(
            image_uint8, 
            distances=distances, 
            angles=angles,
            symmetric=True,
            normed=True
        )
        
        features = {}
        props = ['contrast', 'dissimilarity', 'homogeneity', 
                'energy', 'correlation', 'ASM']
        
        for prop in props:
            prop_values = graycoprops(glcm, prop)
            for i, dist in enumerate(distances):
                for j, angle in enumerate(angles):
                    features[f'glcm_{prop}_d{dist}_a{int(np.degrees(angle))}'] = prop_values[i, j]
        
        return features
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """Extract Local Binary Pattern (LBP) features"""
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
        
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        hist = hist.astype('float32')
        hist /= hist.sum() + 1e-7
        
        features = {}
        for i, val in enumerate(hist):
            features[f'lbp_bin_{i}'] = val
        
        return features
    
    def extract_histogram_features(self, image, n_bins=32):
        """Extract histogram-based features"""
        flat_image = image.flatten()
        
        hist, bins = np.histogram(flat_image, bins=n_bins)
        
        hist = hist.astype('float32')
        hist /= hist.sum() + 1e-7
        
        features = {}
        for i, val in enumerate(hist):
            features[f'hist_bin_{i}'] = val
        
        features['hist_mean'] = np.mean(flat_image)
        features['hist_std'] = np.std(flat_image)
        features['hist_skew'] = pd.Series(flat_image).skew()
        features['hist_kurtosis'] = pd.Series(flat_image).kurtosis()
        
        return features
    
    def extract_shape_features(self, image, threshold=0.5):
        """Extract shape-based features using contour detection"""
        _, binary = cv2.threshold((image * 255).astype(np.uint8), 
                                 threshold * 255, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['shape_area'] = area
            features['shape_perimeter'] = perimeter
            
            if perimeter > 0:
                features['shape_compactness'] = (4 * np.pi * area) / (perimeter ** 2)
            else:
                features['shape_compactness'] = 0
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['shape_aspect_ratio'] = w / h if h > 0 else 0
            features['shape_extent'] = area / (w * h) if w * h > 0 else 0
            
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            for i, hu in enumerate(hu_moments):
                features[f'shape_hu_{i+1}'] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        return features
    
    def extract_all_features(self, image):
        """Extract all features from an image"""
        all_features = {}
        
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image_2d = image.squeeze()
        else:
            image_2d = image
        
        all_features.update(self.extract_glcm_features(image_2d))
        all_features.update(self.extract_lbp_features(image_2d))
        all_features.update(self.extract_histogram_features(image_2d))
        all_features.update(self.extract_shape_features(image_2d))
        
        return all_features

# Set page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

# Title
st.title("🧠 Brain Tumor Classification System")
st.markdown("---")

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all saved models and preprocessors"""
    try:
        best_model = joblib.load("best_ml_model.joblib")
        scaler = joblib.load("feature_scaler.joblib")
        feature_extractor = joblib.load("feature_extractor.joblib")
        preprocessor = joblib.load("image_preprocessor.joblib")
        test_data = joblib.load("test_data.joblib")
        model_results = joblib.load("model_results.joblib")
        
        return best_model, scaler, feature_extractor, preprocessor, test_data, model_results
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure all model files are saved in the same directory as this script.")
        return None, None, None, None, None, None

# Load everything
best_model, scaler, feature_extractor, preprocessor, test_data, model_results = load_models()

if best_model is None:
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
                        ["Home", "Single Image Prediction", "Batch Prediction", "Model Performance", "Test Data Analysis"])

# ============== HOME PAGE ==============
if page == "Home":
    st.header("Welcome to Brain Tumor Classifier")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 About this System")
        st.write("""
        This application uses machine learning to classify brain MRI images into:
        - **Glioma Tumor**
        - **Meningioma Tumor**
        - **Pituitary Tumor**
        - **No Tumor**
        
        The model was trained on extracted texture and shape features including:
        - GLCM (Gray-Level Co-occurrence Matrix)
        - LBP (Local Binary Pattern)
        - Histogram Statistics
        - Shape Features
        """)
    
    with col2:
        st.subheader("📈 Model Performance")
        if model_results:
            st.metric("Accuracy", f"{model_results['accuracy']:.2%}")
            st.metric("Precision", f"{model_results['precision']:.2%}")
            st.metric("Recall", f"{model_results['recall']:.2%}")
            st.metric("F1-Score", f"{model_results['f1']:.2%}")

# ============== SINGLE IMAGE PREDICTION ==============
elif page == "Single Image Prediction":
    st.header("Single Image Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload MRI Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tif"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to array
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Save temporarily and preprocess
            temp_path = "temp_image.jpg"
            cv2.imwrite(temp_path, img_array)
            
            try:
                # Preprocess
                preprocessed_img = preprocessor.preprocess_image(temp_path)
                
                if preprocessed_img is not None:
                    # Extract features
                    features = feature_extractor.extract_all_features(preprocessed_img)
                    features_array = np.array(list(features.values())).reshape(1, -1)
                    
                    # Scale features
                    scaled_features = scaler.transform(features_array)
                    
                    # Predict
                    prediction = best_model.predict(scaled_features)[0]
                    probabilities = best_model.predict_proba(scaled_features)[0]
                    
                    with col2:
                        st.subheader("Prediction Results")
                        
                        class_names = test_data['classes']
                        predicted_class = class_names[prediction]
                        confidence = probabilities[prediction]
                        
                        st.metric("Predicted Class", predicted_class)
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        st.subheader("Probability Distribution")
                        
                        # Create probability chart
                        prob_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
                        st.bar_chart(prob_dict)
                        
                        st.subheader("Detailed Results")
                        for i, class_name in enumerate(class_names):
                            st.write(f"{class_name}: {probabilities[i]:.4f}")
                
                else:
                    st.error("Failed to preprocess image. Please ensure it's a valid MRI image.")
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")

# ============== BATCH PREDICTION ==============
elif page == "Batch Prediction":
    st.header("Batch Prediction")
    
    uploaded_files = st.file_uploader("Upload multiple MRI images", type=["jpg", "jpeg", "png", "tif"], 
                                      accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Batch"):
        results_list = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}")
            
            try:
                image = Image.open(uploaded_file)
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                temp_path = "temp_batch_image.jpg"
                cv2.imwrite(temp_path, img_array)
                
                preprocessed_img = preprocessor.preprocess_image(temp_path)
                
                if preprocessed_img is not None:
                    features = feature_extractor.extract_all_features(preprocessed_img)
                    features_array = np.array(list(features.values())).reshape(1, -1)
                    scaled_features = scaler.transform(features_array)
                    
                    prediction = best_model.predict(scaled_features)[0]
                    probabilities = best_model.predict_proba(scaled_features)[0]
                    
                    class_names = test_data['classes']
                    
                    results_list.append({
                        'Filename': uploaded_file.name,
                        'Predicted Class': class_names[prediction],
                        'Confidence': f"{probabilities[prediction]:.2%}"
                    })
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
            except Exception as e:
                results_list.append({
                    'Filename': uploaded_file.name,
                    'Predicted Class': 'Error',
                    'Confidence': str(e)
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        st.subheader("Batch Results")
        
        import pandas as pd
        df_results = pd.DataFrame(results_list)
        st.dataframe(df_results, use_container_width=True)

# ============== MODEL PERFORMANCE ==============
elif page == "Model Performance":
    st.header("Model Performance on Test Data")
    
    if model_results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_results['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{model_results['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{model_results['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{model_results['f1']:.2%}")
        
        st.subheader("Confusion Matrix")
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        cm = model_results['confusion_matrix']
        class_names = test_data['classes']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", color="black")
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)

# ============== TEST DATA ANALYSIS ==============
elif page == "Test Data Analysis":
    st.header("Test Data Analysis")
    
    if test_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Test Samples", len(test_data['y_test']))
        
        with col2:
            class_names = test_data['classes']
            unique, counts = np.unique(test_data['y_test'], return_counts=True)
            
        st.subheader("Class Distribution in Test Set")
        
        import pandas as pd
        
        dist_dict = {}
        for i, count in enumerate(counts):
            dist_dict[class_names[i]] = count
        
        st.bar_chart(dist_dict)
        
        df_dist = pd.DataFrame({
            'Class': [class_names[i] for i in unique],
            'Count': counts,
            'Percentage': [f"{count/len(test_data['y_test'])*100:.1f}%" for count in counts]
        })
        
        st.dataframe(df_dist, use_container_width=True)
