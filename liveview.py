import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input


# 1. PAGE CONFIGURATION (RENDER THIS FIRST)

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ Live AI Waste Classification")
st.write("Hold an item up to your webcam to determine its hierarchical taxonomy and safety protocol.")


# 2. BULLETPROOF FILE PATHS

# This ensures it always finds the artifacts folder, no matter where your terminal is opened
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /  "D:\\ml\\test\\best_production_model.keras"
MAPPING_PATH = BASE_DIR / "class_mapping.json"


# 3. LOAD AI ARTIFACTS INTO MEMORY

@st.cache_resource
def load_system():
    # Pre-check if files exist so it doesn't crash silently
    if not MODEL_PATH.exists():
        return None, None, f"Model file not found at: {MODEL_PATH}"
    if not MAPPING_PATH.exists():
        return None, None, f"Mapping file not found at: {MAPPING_PATH}"
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(MAPPING_PATH, "r") as f:
            raw_map = json.load(f)
            class_map = {int(k): v for k, v in raw_map.items()}
        return model, class_map, None
    except Exception as e:
        return None, None, str(e)

# Render a beautiful UI spinner so you know the screen isn't frozen
with st.spinner("🧠 Mounting AI Brain into Memory (This takes 10-20 seconds on first load)..."):
    model, class_mapping, system_error = load_system()

# If the load failed, show a giant red error on the screen, not in the hidden terminal
if system_error:
    st.error(f"**Critical System Error:** {system_error}")
    st.stop()


# 4. HIERARCHY TRANSLATOR

def translate_to_hierarchy(folder_name: str):
    """Translates the raw folder name into your strict 4-tier data flow."""
    # Tier 1 & 2
    if folder_name.startswith('hw_'):
        risk = "Hazardous"
        color = "#e74c3c" # Red
        if 'ewaste' in folder_name: category = "E-Waste"
        elif 'chem' in folder_name: category = "Chemicals"
        elif 'med' in folder_name: category = "Medical"
        else: category = "Unknown Hazard"
    else:
        risk = "Non-Hazardous"
        color = "#2ecc71" # Green
        if 'comp' in folder_name and 'noncomp' not in folder_name: category = "Compostable"
        elif 'rec' in folder_name and 'nonrec' not in folder_name: category = "Recyclable"
        elif 'nonrec' in folder_name: category = "Landfill (Non-Recyclable)"
        else: category = "Unknown Safe"

    # Tier 3 (Clean up the exact item name)
    item_name = folder_name.split('_')[-1].capitalize()

    return risk, category, item_name, color


# 5. LIVE CAMERA INTERFACE

st.markdown("---")
# Streamlit's built-in webcam widget
camera_image = st.camera_input("Scan your waste item")

if camera_image is not None:
    # 1. Read and Display the Image
    image = Image.open(camera_image).convert('RGB')
    
    with st.spinner("Scanning item via EfficientNet..."):
        # 2. Preprocess exactly like the Kaggle pipeline
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # 3. Run Inference
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx]) * 100
        
        predicted_folder = class_mapping.get(predicted_idx, "Unknown")
        
        # 4. Map to Hierarchy
        risk, category, item_name, color = translate_to_hierarchy(predicted_folder)
        

    # 6. RENDER THE DATA FLOW RESULTS

    st.markdown("---")
    st.subheader("Classification Results")
    
    # Display the confidence score
    st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
    
    # Display the hierarchical flow
    st.markdown(f"### Master Data Flow Pathway:")
    flow_html = f"""
    <div style='padding: 20px; border-radius: 10px; background-color: #1e1e1e; border: 1px solid #444;'>
        <h4 style='color: white; margin: 0; font-family: monospace;'>
            🗑️ Waste &nbsp;➔&nbsp; 
            <span style='color: {color};'>{risk}</span> &nbsp;➔&nbsp; 
            <span style='color: lightblue;'>{category}</span> &nbsp;➔&nbsp; 
            <span style='color: gold;'>{item_name}</span>
        </h4>
        <p style='color: gray; margin-top: 15px; font-size: 14px;'>Target Node: <code>{predicted_folder}</code></p>
    </div>
    """
    st.markdown(flow_html, unsafe_allow_html=True)
    
    # Display actionable warning if Hazardous
    st.markdown("<br>", unsafe_allow_html=True)
    if risk == "Hazardous":
        st.error("⚠️ **HAZARDOUS MATERIAL DETECTED:** Do not place in standard bins. Follow specialized disposal protocols immediately.")
    else:
        st.success("✅ **SAFE MATERIAL:** Proceed with standard sorting protocols.")
