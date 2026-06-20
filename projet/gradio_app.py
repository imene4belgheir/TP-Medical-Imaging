import gradio as gr
import joblib
import numpy as np
from PIL import Image
import os
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

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

# Load models and preprocessors
try:
    best_model = joblib.load("best_ml_model.joblib")
    scaler = joblib.load("feature_scaler.joblib")
    feature_extractor = joblib.load("feature_extractor.joblib")
    preprocessor = joblib.load("image_preprocessor.joblib")
    test_data = joblib.load("test_data.joblib")
    model_results = joblib.load("model_results.joblib")
    
    class_names = test_data['classes']
    
    print("✓ All models loaded successfully!")
    
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Make sure all model files are saved in the same directory as this script.")
    exit()

# Prediction function
def predict_tumor(image):
    """
    Predict brain tumor from MRI image
    """
    if image is None:
        return "Please upload an image", {}
    
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_rgb = img_array
        elif img_array.shape[2] == 4:  # RGBA
            img_rgb = Image.fromarray(img_array).convert('RGB')
            img_rgb = np.array(img_rgb)
        else:  # RGB
            img_rgb = img_array
        
        # Save temporarily
        temp_path = "temp_gradio_image.jpg"
        Image.fromarray(img_rgb if len(img_rgb.shape) == 3 else np.stack([img_rgb]*3, axis=2)).save(temp_path)
        
        # Preprocess
        preprocessed_img = preprocessor.preprocess_image(temp_path)
        
        if preprocessed_img is None:
            return "Error: Could not preprocess image", {}
        
        # Extract features
        features = feature_extractor.extract_all_features(preprocessed_img)
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Predict
        prediction = best_model.predict(scaled_features)[0]
        probabilities = best_model.predict_proba(scaled_features)[0]
        
        predicted_class = class_names[prediction]
        confidence = probabilities[prediction]
        
        # Create probability dictionary
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        result_text = f"🧠 **Predicted Class:** {predicted_class}\n\n✓ **Confidence:** {confidence:.2%}"
        
        return result_text, prob_dict
        
    except Exception as e:
        return f"Error processing image: {str(e)}", {}

# Get model info
def get_model_info():
    info_text = f"""
    ## Model Performance Summary
    
    - **Accuracy:** {model_results['accuracy']:.2%}
    - **Precision:** {model_results['precision']:.2%}
    - **Recall:** {model_results['recall']:.2%}
    - **F1-Score:** {model_results['f1']:.2%}
    
    ### Test Set Statistics
    - **Total Test Samples:** {len(test_data['y_test'])}
    - **Classes:** {', '.join(class_names)}
    """
    return info_text

# Create Gradio interface
with gr.Blocks(title="Brain Tumor Classifier") as demo:
    gr.Markdown("# 🧠 Brain Tumor Classification System")
    gr.Markdown("Upload an MRI image to classify brain tumors")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload MRI Image")
            image_input = gr.Image(type="pil", label="MRI Image")
            submit_btn = gr.Button("Classify", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Prediction Results")
            prediction_output = gr.Markdown(label="Prediction")
            chart_output = gr.BarPlot(
                x="Class",
                y="Probability",
                title="Prediction Probabilities",
                x_label="Tumor Class",
                y_label="Probability",
                label="Class Probabilities"
            )
    
    # Model info
    gr.Markdown("---")
    gr.Markdown(get_model_info())
    
    # Connect button to function
    submit_btn.click(
        fn=predict_tumor,
        inputs=image_input,
        outputs=[prediction_output, chart_output]
    )
    
    # Example usage
    gr.Markdown("""
    ### How to use:
    1. Upload a brain MRI image (JPG, PNG, or TIF format)
    2. Click "Classify" button
    3. View the prediction and confidence level
    4. See probability distribution across all classes
    
    ### Supported Classes:
    - Glioma Tumor
    - Meningioma Tumor
    - Pituitary Tumor
    - No Tumor
    """)

if __name__ == "__main__":
    demo.launch(share=True)
