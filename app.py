import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Page configuration - Changed to wide layout
st.set_page_config(
    page_title="MRI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",  # Changed from "centered" to "wide"
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme with purple accents - Updated for full width
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    .block-container {
        padding-top: 1rem;
        margin-top: 0;
        max-width: 100%;  /* Changed from 800px to 100% */
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    header {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    .main-title {
        font-size: 2.5rem;
        text-align: center;
        background: linear-gradient(45deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: linear-gradient(to right, #1e1e2e, #2e2e4e);
        border: 2px dashed #a855f7;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        width: 100%;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2e2e4e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #a855f7;
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.2);
        max-width: 800px;
        margin: 0 auto;
    }
    
    .success-result {
        background: linear-gradient(135deg, #065f46, #047857);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
    }
    
    .warning-result {
        background: linear-gradient(135deg, #92400e, #d97706);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #facc15;
        margin: 1rem 0;
    }
    
    .info-section {
        background: rgba(168, 85, 247, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #a855f7;
        margin: 1rem 0;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #a855f7, #ec4899);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
    }
    
    /* Full width file uploader styling */
    .stFileUploader {
        width: 100% !important;
        margin: 1rem 0;
    }
    
    .stFileUploader > div {
        width: 100% !important;
    }
    
    .stFileUploader > div > div > div {
        background: linear-gradient(to right, #1e1e2e, #2e2e4e);
        border: 2px dashed #a855f7;
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0;
        width: 100% !important;
        max-width: 100% !important;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
    }
    
    .uploadedFile {
        border: 1px solid #a855f7;
        border-radius: 10px;
        background: rgba(168, 85, 247, 0.1);
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto;
    }
    
    /* Full width upload container */
    .upload-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0;
        margin: 1rem 0 2rem 0;
    }
    
    /* Upload section header styling */
    .upload-header {
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .footer-text {
        text-align: center;
        background: linear-gradient(45deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Center content sections while keeping upload full width */
    .centered-content {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Main title with glow effect
st.markdown('<div class="centered-content">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🧠 MRI Brain Tumor Detection</h1>', unsafe_allow_html=True)
st.markdown("---")

# Project description
st.markdown("""
<div class="info-section">
<h3>🎯 Welcome to the Brain Tumor Detection System</h3>
<p>Upload an MRI brain scan and our AI system will analyze it to determine if the brain is healthy or contains a tumor. This system uses advanced deep learning techniques for accurate medical image analysis.</p>
</div>
""", unsafe_allow_html=True)

# About This AI Model - Expandable Section
with st.expander("📊 About This AI Model", expanded=False):
    st.markdown("""
    **Model Architecture:** Convolutional Neural Network (CNN)
    
    **Training Data:**
    - **Tumor Cases:** Brain MRI scans with confirmed tumors
    - **Normal Cases:** Healthy brain MRI scans
    
    **Performance Metrics:**
    - **Accuracy:** 97.8%
    - **Trained for:** 18 epochs
    
    **Image Requirements:**
    - **Format:** PNG, JPG, JPEG, or DICOM
    - **Quality:** High resolution preferred
    - **Type:** Brain MRI scans only
    
    **Important Notes:**
    - Results are for assistance only
    - Does not replace medical examination
    - Always consult with a specialist radiologist
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Model architecture definition matching the trained model
def create_mri_model():
    """
    Create MRI model with the same architecture used in training
    """
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 1024, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.MaxPool2d(4, 4),
        nn.Flatten(),
        nn.Linear(7*7*1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return model

# Function to load the model
@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model with the same architecture used in training
        model = create_mri_model()
        
        # Load the trained model
        checkpoint = torch.load('best_mri_model_epoch_18_acc_97.80.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Model not found: {str(e)}")
        st.error("Make sure to place the file 'best_mri_model_epoch_18_acc_97.80.pth' in the same directory as the code.")
        return None, None

# Function to preprocess image
def preprocess_image(image):
    """
    Process image with the same method used in training
    """
    # Define transformations matching training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    img_tensor = transform(image)
    
    # Add batch dimension
    img_batch = img_tensor.unsqueeze(0)
    
    return img_batch

# Function for prediction
def predict_tumor(model, device, image):
    """
    Predict tumor presence in image using trained model
    """
    processed_image = preprocess_image(image)
    processed_image = processed_image.to(device)
    
    with torch.inference_mode():
        logits = model(processed_image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_indices = torch.max(logits, 1)
        confidence = torch.max(probabilities).item() * 100
    
    # Class names according to original dataset order
    # From code: MRI.classes will give classes sorted alphabetically
    # You may need to adjust according to actual classes
    
    # If your model has only two classes (tumor/no tumor)
    # Use this instead of the previous line:
    class_names = ["No Tumor", "Tumor"]  # Adjust order according to your model
    
    predicted_class = predicted_indices.item()
    result = class_names[predicted_class]
    
    return result, confidence

# Load model
model_data = load_model()
if model_data[0] is not None:
    model, device = model_data
else:
    model, device = None, None

# Image upload interface - FULL WIDTH (no container wrapper)
st.markdown('<h3 class="upload-header">📤 Upload MRI Scan</h3>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a0a0a0; margin-bottom: 2rem;">Select your brain MRI image for AI analysis</p>', unsafe_allow_html=True)

# Full width file uploader - no columns, no containers
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=['png', 'jpg', 'jpeg', 'dcm'],
    help="Supported formats: PNG, JPG, JPEG, DICOM. Max size: 200MB.",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Center the content for image display and results
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    
    # Display uploaded image in controlled width
    st.markdown("#### 🖼️ Uploaded Image:")
    
    image = Image.open(uploaded_file)
    
    # Center the image with controlled width
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    
    st.markdown("---")
    
    # Prediction section
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown("#### 🔬 Analysis & Prediction Result:")
    
    if model is not None:
        # Create columns for the analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Add analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("🤖 Analyzing image..."):
                    try:
                        result, confidence = predict_tumor(model, device, image)
                        
                        # Display result with different colors based on type
                        if "No Tumor" in result or "notumor" in result.lower():
                            st.markdown(f'<div class="success-result">✅ <strong>Result:</strong> {result}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="success-result">📊 <strong>Confidence:</strong> {confidence:.2f}%</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-result">⚠️ <strong>Result:</strong> {result}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="warning-result">📊 <strong>Confidence:</strong> {confidence:.2f}%</div>', unsafe_allow_html=True)
                        
                        # Display additional information
                        st.markdown("""
                        <div class="info-section">
                        ℹ️ <strong>Disclaimer:</strong> This AI tool is a support tool and does not replace professional medical advice. Please consult with a qualified medical professional for proper diagnosis and treatment.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.error("❌ Cannot analyze image without loading the model.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Additional information - centered content
st.markdown('<div class="centered-content">', unsafe_allow_html=True)
st.markdown("---")

# Important notes
st.markdown("""
<div class="info-section">
<h3>📋 Important Notes</h3>
<p><strong>Note:</strong> This model is trained to detect whether there is a tumor or not:</p>
<ul>
<li><strong>Tumor</strong> - Tumor detected</li>
<li><strong>No Tumor</strong> - No tumor detected</li>
</ul>
<br>
<ul>
<li>Ensure the image is clear and of high quality</li>
<li>The system is trained specifically on brain MRI images</li>
<li>Results are for assistance only and do not replace medical examination</li>
<li>In case of doubt, consult a specialist radiologist</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Usage instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    1. Click on **"Browse files"** to select an MRI image
    2. Choose an image in PNG, JPG, or DICOM format
    3. Click on **"Analyze Image"** to get the result
    4. The result will appear with the confidence percentage
    
    **Tips for best results:**
    - Use high-quality MRI brain scans
    - Ensure proper image orientation
    - Supported file formats: PNG, JPG, JPEG, DICOM
    - Maximum file size: 200MB
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
<p><strong>🧠 Brain Tumor Detection System</strong></p>
<p>🤖 Built with Streamlit and PyTorch | Powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)