import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TB Detection", layout="centered")

IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Tuberculosis"]

# ---------------------------------------------------------
# LOAD MODEL SAFELY
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    try:
        model = keras.models.load_model(path, compile=False)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# PREPROCESS FUNCTION
# ---------------------------------------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32") / 255.0  # normalize 0–1
    img = np.expand_dims(img, axis=0)  # shape: (1,224,224,3)
    return img

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Tuberculosis Detection", "Creator Info"]
)

# ---------------------------------------------------------
# PAGE 1 — OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.title("📘 Project Overview")
    # Add an image
    st.image(
        "lungs.png",
        use_container_width=True
    )
    st.markdown(
        """
        ## 🩺 Tuberculosis Detection Using Deep Learning  
        This project uses a **VGG16 CNN model** to classify chest X-ray images into:  
        - **Normal**
        - **Tuberculosis (TB)**  

        ### 🔍 Project Highlights  
        - Uses transfer learning (VGG16)  
        - Achieves high accuracy on TB classification  
        - Accepts chest X-ray images and predicts probability  
        - Simple Streamlit interface  
        """
    )



# ---------------------------------------------------------
# PAGE 2 — TB DETECTION (YOUR ORIGINAL CODE)
# ---------------------------------------------------------
elif page == "Tuberculosis Detection":

    # (Your original code starts exactly here — untouched)

    st.sidebar.header("🔧 Model & Dataset Settings")
    model_path = st.sidebar.text_input("Model Path", "VGG16.keras")

    model, error = load_model(model_path)

    if model is None:
        st.sidebar.error(f"Model Load Error:\n{error}")
        st.stop()
    else:
        st.sidebar.success("Model Loaded Successfully")

    st.header("🩺 TB Detection From Chest X-ray")

    uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            try:
                inp = preprocess_image(img)
                pred = model.predict(inp)[0][0]  # sigmoid output

                label = CLASS_NAMES[1] if pred >= 0.5 else CLASS_NAMES[0]

                st.subheader(f"Prediction: **{label}**")
                st.write(f"Confidence Score: **{pred:.4f}**")

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    # (Your original code ends here — unchanged)

# ---------------------------------------------------------
# PAGE 3 — CREATOR INFO
# ---------------------------------------------------------
elif page == "Creator Info":
    st.title("👤 Creator Information")
    st.markdown(
        """
        ### 👋 Created By  
        **Karuna**  
        Aspiring Data Science Enthusiast  

        ### 📚 Skills  
        - Machine Learning  
        - Deep Learning  
        - Computer Vision  
        - Python | SQL | Streamlit  
        - Data Analysis & Visualization  

        ### 📧 Contact  
        Email: **karuna@example.com**  
        """
    )

    st.markdown("---")
    st.write("Thank you for exploring the TB Detection App!")
