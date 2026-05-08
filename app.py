import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
from collections import defaultdict

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Vision Dashboard",
    layout="centered",
    page_icon="🎥"
)

# ---------------- UI ---------------- #
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

.stApp {
    background: radial-gradient(circle at top, #ffc8dd, #ffafcc, #ff8fab);
    color: #1f1f1f;
    font-family: 'Poppins', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* 🎨 TITLE - PASTEL PALETTE GRADIENT */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;

    color: #e76f9a;  /* 🎨 slightly darker pastel pink */

    letter-spacing: 1px;
}

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

/* CAMERA BOX */
.camera-container {
    width: 720px;
    margin: auto;
    padding: 14px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.35);
    border: 2px solid #ffafcc;
    box-shadow: 0 8px 25px rgba(255, 143, 171, 0.3);
    backdrop-filter: blur(10px);
}

/* VIDEO */
video {
    width: 100% !important;
    height: auto !important;
    aspect-ratio: 4 / 3 !important;
    object-fit: contain !important;
    border-radius: 14px !important;
}

/* BUTTONS */
button {
    background: linear-gradient(90deg, #ffafcc, #a2d2ff) !important;
    color: #1f1f1f !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* TEXT */
p, span {
    font-family: 'Poppins', sans-serif;
    color: #1f1f1f !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("<div class='title'>🎥 Live Object Detection Tracking</div>", unsafe_allow_html=True)

# ---------------- MODEL ---------------- #
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- STATE ---------------- #
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# ---------------- CALLBACK ---------------- #
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 480))

    results = model.predict(img, conf=0.5, verbose=False)
    annotated = img.copy()

    counts = defaultdict(int)

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:

            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])

            counts[name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 143, 171),
                2
            )

            cv2.putText(
                annotated,
                f"{name} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1
            )

    # ---------------- TRANSPARENT HUD ---------------- #
    overlay = annotated.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (320, 220),
        (0, 0, 0),
        -1
    )

    alpha = 0.25
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

    # HUD TEXT
    y = 40

    cv2.putText(
        annotated,
        "DETECTED OBJECTS",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 143, 171),
        1
    )

    for k, v in list(counts.items())[:6]:
        cv2.putText(
            annotated,
            f"{k}: {v}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y += 25

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------- CAMERA UI ---------------- #
st.markdown("<div class='camera-container'>", unsafe_allow_html=True)

if st.session_state.camera_on:
    webrtc_streamer(
        key="ai-camera",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
else:
    st.warning("🔴 Camera OFF — Click ON to start")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BUTTONS ---------------- #
col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 TURN ON CAMERA"):
        st.session_state.camera_on = True
        st.rerun()

with col2:
    if st.button("🔴 TURN OFF CAMERA"):
        st.session_state.camera_on = False
        st.rerun()