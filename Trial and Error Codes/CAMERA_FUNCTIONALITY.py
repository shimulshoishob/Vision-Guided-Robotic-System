import streamlit as st
import cv2
import time
from ultralytics import YOLO

# ==================== YOLO MODEL ====================
MODEL_PATH = r"D:/EWU/10th Semester/CSE475/LABS/best.pt"
st.sidebar.markdown("### 📦 Loading YOLO model...")
model = YOLO(MODEL_PATH)
st.sidebar.success("✅ YOLO model loaded")

# ==================== SESSION STATE ====================
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "camera_index" not in st.session_state:
    st.session_state.camera_index = 0

# ==================== CAMERA SELECTION ====================
st.sidebar.markdown("### 📷 Select Camera (MSMF Only)")

camera_index = st.sidebar.radio(
    "Choose Iriun / USB camera index:",
    [0, 1],
    index=0
)
st.session_state.camera_index = camera_index

debug_mode = st.sidebar.checkbox("Debug Mode (FPS + classes)", True)

# ==================== UI LAYOUT ====================
st.title("🤖 Real-Time Vegetable Detection")
start = st.button("▶️ Start Camera")
stop = st.button("⏹️ Stop Camera")

frame_display = st.empty()

# ==================== CAMERA CONTROL ====================
if start:
    st.session_state.camera_running = True

if stop:
    st.session_state.camera_running = False

# ==================== CAMERA LOOP ====================
if st.session_state.camera_running:

    cap = cv2.VideoCapture(st.session_state.camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        st.error("❌ Cannot open camera — MSMF backend failed.")
        st.session_state.camera_running = False

    prev_time = time.time()

    while st.session_state.camera_running:

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Cannot read frame.")
            break

        results = model(frame, conf=0.8, imgsz=640, verbose=False)

        curr = time.time()
        fps = 1 / (curr - prev_time)
        prev_time = curr

        latest_class_name = None
        if len(results[0].boxes) > 0:
            # Take the class of the last detected box
            latest_cls_index = int(results[0].boxes.cls[-1])
            latest_class_name = results[0].names[latest_cls_index]
        
        if debug_mode:
            st.sidebar.write(f"FPS: {fps:.1f}")
            st.sidebar.write(f"Latest Detected: {latest_class_name if latest_class_name else 'None'}")

        annotated = results[0].plot()

        # Display video
        frame_display.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        # Display latest detection class in main UI
        if latest_class_name:
            st.markdown(f"### 🥬 Latest Detection: {latest_class_name}")
        else:
            st.markdown("### 👀 No detections yet")

    cap.release()
    frame_display.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True,
        caption="Camera stopped"
    )
else:
    st.info("Press ▶️ Start Camera to begin.")