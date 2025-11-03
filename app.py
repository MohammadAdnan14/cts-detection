import streamlit as st
import cv2
import numpy as np
from PIL import Image
from backend import process_image_complete
import io
import time

# Page config for amazing look
st.set_page_config(
    page_title="🦷 ToothGuard AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fantastic styling
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #0f4c75; text-align: center; margin-bottom: 2rem;}
    .upload-area {background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;}
    .result-card {background-color: #e8f4fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #0f4c75;}
    .warning-card {background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #ffc107;}
    .success-card {background-color: #d4edda; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #28a745;}
    .severe-text {font-size: 1.5rem; font-weight: bold; color: #dc3545; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);}
    .high-text {font-size: 1.4rem; font-weight: bold; color: #fd7e14;}
    .medium-text {font-size: 1.3rem; font-weight: bold; color: #ffc107;}
    .low-text {font-size: 1.2rem; font-weight: bold; color: #198754;}
    .none-text {font-size: 1.2rem; font-weight: bold; color: #6c757d;}
    .card-text {color: black !important;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for info
with st.sidebar:
    st.title("ℹ️ Info")
    st.info("""
    **Upload a dental X-ray** (JPG/PNG/JPEG only).\n
    **Processing**: 10-30s using edge detection & morphology.\n
    **Output**: Annotated image + Crack (YES/NO) + Severity.
    """)
    st.markdown("---")
    st.caption("By: Mohammad Adnan Dalal (42), Mohammad Faqueem Khan (43), Sankalp Choubey (49)")

# Main header
st.markdown('<h1 class="main-header">🦷 ToothGuard AI</h1>', unsafe_allow_html=True)
st.markdown("### Detect Cracks in Dental X-Rays Instantly")

# File uploader (X-ray only)
uploaded_file = st.file_uploader(
    "📁 Upload your Dental X-Ray Image",
    type=['jpg', 'jpeg', 'png'],
    help="Only grayscale X-ray images supported (auto-converted if needed)."
)

if uploaded_file is not None:
    # Validate: Ensure it's an image
    try:
        image = Image.open(uploaded_file)
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Convert to numpy array for backend
        img_array = np.array(image.convert('RGB'))  # Ensure RGB for cv2
        
        # Preview
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Upload", use_container_width=True)
        
        # Process button
        if st.button("🚀 Analyze for Cracks", type="primary"):
            with st.spinner("🔬 Processing X-Ray... (Advanced edge detection in progress)"):
                progress_bar = st.progress(0)
                
                # Simulate progress (adapt from your pipeline steps)
                for i in range(100):
                    time.sleep(0.05)  # Simulate work
                    progress_bar.progress(i + 1)
                
                # Run backend
                final_result, metrics = process_image_complete(img_array, uploaded_file.name)
                
                # Convert final_result to PIL for display
                final_pil = Image.fromarray(cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
                
                # Results section
                st.markdown("---")
                st.markdown('<h2>📊 Analysis Results</h2>', unsafe_allow_html=True)
                
                # Crack presence & severity
                col_a, col_b = st.columns(2)
                with col_a:
                    if metrics['num_cracks'] > 0:
                        st.markdown(f'<div class="result-card"><h3 class="card-text">🔴 Crack Detected: **YES**</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-card"><h3 class="card-text">🟢 Crack Detected: **NO**</h3></div>', unsafe_allow_html=True)
                
                with col_b:
                    severity = metrics['risk_level'] if metrics['num_cracks'] > 0 else 'NONE'
                    if severity == 'SEVERE':
                        st.markdown(f'<div class="warning-card"><h3 class="card-text severe-text">⚠️ Severity: **{severity}**</h3></div>', unsafe_allow_html=True)
                    elif severity == 'HIGH':
                        st.markdown(f'<div class="warning-card"><h3 class="card-text high-text">⚠️ Severity: **{severity}**</h3></div>', unsafe_allow_html=True)
                    elif severity == 'MEDIUM':
                        st.markdown(f'<div class="warning-card"><h3 class="card-text medium-text">⚠️ Severity: **{severity}**</h3></div>', unsafe_allow_html=True)
                    elif severity == 'LOW':
                        st.markdown(f'<div class="success-card"><h3 class="card-text low-text">ℹ️ Severity: **{severity}**</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-card"><h3 class="card-text none-text">ℹ️ Severity: **{severity}**</h3></div>', unsafe_allow_html=True)
                
                # Annotated image (like Cell 14)
                st.markdown('<h3>🖼️ Annotated Result</h3>', unsafe_allow_html=True)
                st.image(final_pil, caption="Cracks highlighted in red | Green boxes: Boundaries | Yellow dots: Centers | Magenta lines: Orientation", use_container_width=True)
                
                # Download button for annotated image
                buf = io.BytesIO()
                final_pil.save(buf, format='PNG')
                st.download_button(
                    label="💾 Download Annotated Image",
                    data=buf.getvalue(),
                    file_name=f"annotated_{uploaded_file.name}",
                    mime="image/png"
                )
                
    except Exception as e:
        st.error(f"❌ Invalid image: {str(e)}. Please upload a valid dental X-ray (JPG/PNG/JPEG).")

else:
    st.info("👆 Upload a dental X-ray image to get started!")