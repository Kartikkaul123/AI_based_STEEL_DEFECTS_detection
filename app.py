import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image
import tempfile

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Steel Surface Inspection Console",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0e1117;
}

/* Main title */
h1 {
    font-family: 'Times New Roman', serif;
    color: #e6e6e6;
    letter-spacing: 1px;
}

/* Subtitles */
h2, h3 {
    font-family: 'Georgia', serif;
    color: #cfcfcf;
}

/* Normal text */
p, label, div {
    font-family: 'Arial', sans-serif;
    color: #d0d0d0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #151a23;
    border-right: 2px solid #2a2f3a;
}

/* Buttons */
.stButton>button {
    background-color: #2a2f3a;
    color: #ffffff;
    border-radius: 4px;
    border: 1px solid #444;
}
.stButton>button:hover {
    background-color: #3a3f4a;
}

/* Success / Error boxes */
div[data-testid="stAlert"] {
    border-radius: 4px;
    font-weight: bold;
}

/* Image border */
img {
    border: 2px solid #2f3440;
    border-radius: 4px;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("last.pt")

model = load_model()

# ================= SIDEBAR =================
st.sidebar.title("Inspection Settings")

mode = st.sidebar.radio(
    "Operation Mode",
    ["Image Inspection", "Live Camera Inspection", "Video Inspection"]
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1, 0.9, 0.65, 0.05
)

use_grayscale = st.sidebar.checkbox(
    "Monochrome Processing",
    value=True
)

camera_index = st.sidebar.selectbox("Camera Device Index", [0, 1, 2])
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 4, 2)

st.sidebar.markdown("---")
st.sidebar.caption("Industrial quality control console")

# ================= HEADER =================
st.title("Steel Surface Inspection Console")
st.caption("Visual quality verification system for steel manufacturing")

# =====================================================
# IMAGE INSPECTION
# =====================================================
if mode == "Image Inspection":

    uploaded_image = st.file_uploader(
        "Load steel surface image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        if use_grayscale:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        results = model.predict(
            image_np,
            imgsz=416,
            conf=confidence,
            max_det=20,
            verbose=False
        )

        annotated = results[0].plot()
        detections = results[0].boxes

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Surface")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Inspection Output")
            st.image(annotated, use_column_width=True)

        st.markdown("---")

        if len(detections) == 0:
            st.success("SURFACE STATUS: ACCEPTABLE")
        else:
            st.error(f"SURFACE STATUS: DEFECTS FOUND ({len(detections)})")

            st.subheader("Defect Summary")
            for box in detections:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                st.write(f"{model.names[cls_id]} — confidence {conf_score:.2f}")

# =====================================================
# LIVE CAMERA INSPECTION
# =====================================================
elif mode == "Live Camera Inspection":

    start = st.button("Start Inspection")
    stop = st.button("Stop")

    frame_box = st.image([])
    status_box = st.empty()
    fps_box = st.empty()

    if start:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_id = 0
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            if use_grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            results = model.predict(
                frame,
                imgsz=416,
                conf=confidence,
                max_det=20,
                verbose=False
            )

            annotated = results[0].plot()
            detections = results[0].boxes
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_box.image(annotated)

            if len(detections) == 0:
                status_box.success("STATUS: ACCEPTABLE")
            else:
                status_box.error(f"STATUS: DEFECTS FOUND ({len(detections)})")

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_box.info(f"Processing Rate: {fps:.1f} FPS")

            if stop:
                break

        cap.release()

# =====================================================
# VIDEO INSPECTION
# =====================================================
elif mode == "Video Inspection":

    st.subheader("Recorded Video Inspection")

    source = st.radio("Video Source", ["Upload File", "Direct Video URL"])

    video_path = None

    if source == "Upload File":
        uploaded_video = st.file_uploader(
            "Upload inspection video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(uploaded_video.read())
            video_path = temp.name

    else:
        video_path = st.text_input("Direct video file URL (.mp4 only)")

    start_video = st.button("Start Video Inspection")

    frame_box = st.image([])
    status_box = st.empty()
    fps_box = st.empty()

    if start_video and video_path:
        cap = cv2.VideoCapture(video_path)
        prev_time = time.time()
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            if use_grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            results = model.predict(
                frame,
                imgsz=416,
                conf=confidence,
                max_det=20,
                verbose=False
            )

            annotated = results[0].plot()
            detections = results[0].boxes
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_box.image(annotated)

            if len(detections) == 0:
                status_box.success("STATUS: ACCEPTABLE")
            else:
                status_box.error(f"STATUS: DEFECTS FOUND ({len(detections)})")

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_box.info(f"Processing Rate: {fps:.1f} FPS")

        cap.release()
        st.success("Video inspection completed")

# ================= FOOTER =================
st.markdown("""
<hr style="border:1px solid #2a2f3a">
<p style="text-align:center;color:#7a7a7a;font-size:13px;">
Steel Surface Inspection Console — Manufacturing Quality Control System
</p>
""", unsafe_allow_html=True)
