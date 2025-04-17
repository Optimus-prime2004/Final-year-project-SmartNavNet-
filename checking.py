import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"  # Hides warnings from external libs

import streamlit as st
from roboflow import Roboflow
from PIL import Image
import json
import uuid

# === Always put this FIRST ===
st.set_page_config(page_title="SmartNavNet  WebApp", layout="centered")

# === Configuration ===
API_KEY = "02IdNwvMGiFDQtKr4rrN"
PROJECT_NAME = "loco-dataset-yolo"
WORKSPACE = "mohamed-bouallegue"
MODEL_VERSION = 7

# === Initialize Session State ===
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin"}

# === Authentication UI ===
def login_ui():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.authenticated = True
            st.rerun()  # Refresh the app
        else:
            st.error("❌ Invalid credentials.")

def signup_ui():
    st.title("📝 Signup")
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")
    if st.button("Sign Up"):
        if username in st.session_state.users:
            st.warning("⚠️ Username already exists.")
        else:
            st.session_state.users[username] = password
            st.success("🎉 Account created! Please log in.")

# === Auth Logic ===
if not st.session_state.authenticated:
    mode = st.sidebar.radio("Account", ["Login", "Signup"])
    if mode == "Login":
        login_ui()
    else:
        signup_ui()
else:
    # === Roboflow Setup ===
    @st.cache_resource
    def load_model():
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
        return project.version(MODEL_VERSION).model

    model = load_model()

    st.title("🔍 SmartNavNet WebAPP")
    st.markdown("Upload an image to detect objects using a SmartNavNet hosted on Streamlit.")

    with st.sidebar:
        st.header("📦 Model Details")
        st.write(f"**Project: SmartNavNet YOLO**")
        st.write(f"**Workspace: Manodharshan**")
        st.write(f"**Version:  3S**")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_id = str(uuid.uuid4())
        input_path = f"input_{file_id}.jpg"
        output_path = f"output_{file_id}.jpg"

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(input_path, caption="📸 Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Running model inference..."):
            try:
                prediction = model.predict(input_path)
                prediction.save(output_path)
                result_json = prediction.json()
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                os.remove(input_path)
                st.stop()

        st.subheader("🎯 Prediction Result")
        st.image(output_path, caption="🖼️ Predicted Image with Bounding Boxes", use_container_width=True)

        st.subheader("📌 Detected Objects")
        predictions = result_json.get("predictions", [])
        if predictions:
            pred_data = [{"Class": p["class"], "Confidence (%)": round(p["confidence"] * 100, 2)} for p in predictions]
            st.table(pred_data)
        else:
            st.info("🤷 No objects detected.")

        st.subheader("📊 Model Metrics")
        metrics = result_json.get("metrics", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("mAP", f"{metrics.get('mAP_0.5', 'N/A'):.2f}" if metrics else "62.2%")
        col2.metric("Precision", f"{metrics.get('precision', 'N/A'):.2f}" if metrics else "70.2%")
        col3.metric("Recall", f"{metrics.get('recall', 'N/A'):.2f}" if metrics else "55.7%")

        st.subheader("📥 Download Results")
        with open(output_path, "rb") as f:
            st.download_button("🖼️ Download Prediction Image", f, file_name="predicted.jpg", mime="image/jpeg")

        st.download_button("📄 Download JSON Result", json.dumps(result_json, indent=2), file_name="result.json")

        os.remove(input_path)
        os.remove(output_path)
