import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import datetime

# Import your existing backend logic
try:
    from face_analyzer import FaceAnalyzer
    from face_visualizer import FaceVisualizer
    from destiny_predictor import DestinyPredictor
except ImportError as e:
    st.error(
        f"Backend modules not found. Please ensure face_analyzer.py, face_visualizer.py, and destiny_predictor.py are in the same folder. Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Destiny Mirror AI",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS Styling (Optional but makes it look 'Destiny' themed) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    .stButton>button {
        background-color: #0EA5E9;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
    }
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        color: #38BDF8;
    }
    h1 {
        color: #38BDF8; 
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    h2, h3 {
        color: #F472B6;
    }
    </style>
""", unsafe_allow_html=True)


# Initialization
@st.cache_resource
def load_models():
    """
    Initialize the AI Core.
    Cached so it doesn't reload on every button click.
    """
    analyzer = FaceAnalyzer()
    visualizer = FaceVisualizer()
    predictor = DestinyPredictor()
    return analyzer, visualizer, predictor


try:
    analyzer, visualizer, predictor = load_models()
    ai_status = "🟢 AI Core Online"
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    ai_status = "🔴 AI Core Offline"

# --- Sidebar (VIP & Settings) ---
with st.sidebar:
    st.title(" Destiny Mirror")
    st.write(f"Status: {ai_status}")

    st.markdown("---")
    st.header(" VIP Access")
    is_vip = st.checkbox("Unlock VIP Mode ($5/mo)")

    if is_vip:
        st.success("VIP Unlocked! Detailed metrics enabled.")
    else:
        st.info("Subscribe to see detailed Love, Wealth, and Health breakdown.")

    st.markdown("---")
    st.write("About")
    st.caption("Advanced facial analysis and fortune-prediction using XGBoost + LightGBM.")
    st.caption("[GitHub Repository](https://github.com/Sarahyu-baby/destinyMirror)")

# --- Main Interface ---
st.title("DESTINY MIRROR")
st.markdown("### Reveal your fate through the geometry of your face.")

# 1. Camera Input
# This widget handles the webcam natively in the browser
img_file_buffer = st.camera_input("Take a picture to analyze")

if img_file_buffer is not None:
    # 2. Pre-processing
    # Convert the file buffer to an image format OpenCV can work with
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # Streamlit camera input is usually RGB, OpenCV expects BGR
    # However, MediaPipe (inside FaceAnalyzer) usually works with RGB.
    # We will convert to BGR for standard OpenCV consistency if your backend expects it,
    # or keep it RGB. Let's assume standard OpenCV BGR pipeline.
    frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing facial geometry..."):
        # 3. Analysis Pipeline (Replicating screens.py logic)
        stats = analyzer.process_image(frame_bgr)

        if stats:
            # Predict
            fortune_results = predictor.predict_fortune(stats)

            # Visualize
            lms_data = analyzer.landmarks_np
            custom_pts_data = analyzer.custom_points

            img_to_draw = frame_bgr.copy()
            img_with_dots = visualizer.draw_landmarks(img_to_draw, lms_data)
            final_visualized_img = visualizer.draw_custom_points(img_with_dots, custom_pts_data)

            # Convert back to RGB for display in Streamlit
            final_rgb = cv2.cvtColor(final_visualized_img, cv2.COLOR_BGR2RGB)

            # 4. Display Results
            st.image(final_rgb, caption="Analyzed Feature Map", use_container_width=True)

            st.divider()
            st.header(" Your Destiny Revealed")

            # Create a grid layout for cards
            col1, col2 = st.columns(2)

            # Iterating through results
            for idx, (category, data) in enumerate(fortune_results.items()):
                label = data['label']
                sentence = data['sentence']

                # Logic to hide/show based on VIP
                is_vip_content = "VIP" in label or "2" in category

                if is_vip_content and not is_vip:
                    continue  # Skip this card if user isn't VIP

                # Alternate columns
                target_col = col1 if idx % 2 == 0 else col2

                with target_col:
                    with st.container():
                        st.metric(label=category.replace("_", " ").title(), value=label)
                        st.info(sentence)

            # 5. Data Export (Replacing the 'Save' buttons)
            st.divider()
            st.subheader(" Archives")

            # CSV generation for download
            csv_data = pd.DataFrame(
                [{"Category": k, "Prediction": v['sentence']} for k, v in fortune_results.items()]
            ).to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Fortune Results (CSV)",
                data=csv_data,
                file_name=f"fortune_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            if is_vip:
                # Feature export
                feature_csv = pd.DataFrame(
                    list(stats.items()), columns=['Feature', 'Value']
                ).to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Facial Features (VIP)",
                    data=feature_csv,
                    file_name="face_features.csv",
                    mime="text/csv",
                )

        else:
            st.warning(" No face detected. Please look directly at the camera and try again.")