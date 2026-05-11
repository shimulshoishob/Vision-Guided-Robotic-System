import streamlit as st
import cv2
import time
import threading
import yaml
from ultralytics import YOLO
import xarm

# ==================== CONFIG ====================
YAML_PATH = r"D:/EWU/10th Semester/CSE475/LABS/data.yaml"
MODEL_PATH = r"D:/EWU/10th Semester/CSE475/LABS/best.pt"

GRIPPER_OPEN = 150
GRIPPER_CLOSE = 400
HOME = {1:100, 2:500, 3:250, 4:820, 5:640, 6:500}
LEFT_LIMIT = 900
RIGHT_LIMIT = 100
box_positions = [int(LEFT_LIMIT - i*(LEFT_LIMIT - RIGHT_LIMIT)/8) for i in range(9)]
BOX_OFFSETS = {
    0:(50,30,-60,-240),1:(40,30,-100,-270),2:(30,100,-200,-350),
    3:(20,100,-200,-350),4:(30,100,-200,-350),5:(-20,100,-200,-350),
    6:(-30,100,-200,-350),7:(-40,30,-100,-270),8:(-50,30,-60,-240)
}
duration_map = {1:500,2:500,3:500,4:600,5:700,6:500,"default":500}

# ==================== LOAD YAML ====================
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

all_classes = cfg["names"]
fresh_classes = [c for c in all_classes if "Fresh" in c]
damaged_classes = [c for c in all_classes if "Damaged" in c]
class_to_box = {fresh_classes[i]: i for i in range(len(fresh_classes))}
for c in damaged_classes:
    class_to_box[c] = 8

# ==================== CONNECT ARM ====================
try:
    robot = xarm.Controller('USB')
    print("✅ Connected to xArm 1S")
except:
    st.error("❌ Could not connect to xArm")
    st.stop()

# ==================== GLOBAL VARIABLES ====================
latest_frame = None
latest_class = None
automation_running = False

# ==================== HELPER FUNCTIONS ====================
def clamp(x): return max(0,min(1000,int(x)))
def reverse_bound(home,offset):
    t = home + offset
    return clamp(home - offset if t<0 or t>1000 else t)

def move_servo(sid,pos,d=None):
    if d is None: d = duration_map.get(sid,500)
    robot.setPosition(sid,clamp(pos),duration=d,wait=False)
    time.sleep(d/1000)

def move_multi(moves):
    dmax = 0
    for sid,pos in moves.items():
        d = duration_map.get(sid,500)
        robot.setPosition(sid,clamp(pos),duration=d,wait=False)
        dmax = max(d,dmax)
    time.sleep(dmax/1000)

def move_home(open_gripper=True):
    # Move links 3,4,5 simultaneously
    move_multi({3:HOME[3],4:HOME[4],5:HOME[5]})
    move_servo(2, HOME[2])
    move_servo(6, HOME[6])
    if open_gripper: move_servo(1, GRIPPER_OPEN)

def pick_and_place(box_idx):
    # PICK
    link2 = reverse_bound(HOME[2],0)
    link3 = reverse_bound(HOME[3],-30)
    link4 = reverse_bound(HOME[4],20)
    link5 = reverse_bound(HOME[5],-190)
    pick_position = box_positions[4]

    move_servo(6,pick_position)
    move_servo(1,GRIPPER_OPEN)
    move_multi({3:link3,4:link4,5:link5})
    move_servo(2,link2)
    move_servo(1,reverse_bound(HOME[1],-GRIPPER_CLOSE))
    move_multi({3:HOME[3],4:HOME[4],5:HOME[5]})
    move_servo(2,HOME[2])
    move_servo(6,HOME[6])

    # PLACE
    off = BOX_OFFSETS[box_idx]
    link2 = reverse_bound(HOME[2],off[0])
    link3 = reverse_bound(HOME[3],off[1])
    link4 = reverse_bound(HOME[4],off[2])
    link5 = reverse_bound(HOME[5],off[3])

    move_servo(6,box_positions[box_idx])
    move_multi({3:link3,4:link4,5:link5})
    move_servo(2,link2)
    move_servo(1,GRIPPER_OPEN)
    time.sleep(0.25)
    move_home()

# ==================== LOAD YOLO ====================
st.sidebar.markdown("### 📦 Loading YOLO model...")
model = YOLO(MODEL_PATH)
st.sidebar.success("✅ YOLO model loaded")

camera_index = st.sidebar.radio("Select Camera Index:", [0,1], index=0)
debug_mode = st.sidebar.checkbox("Debug Mode (FPS + classes)", True)

# ==================== STREAMLIT UI ====================
st.title("🤖 Automated Vegetable Sorting Dashboard")
start_btn = st.button("▶️ Start Automation")
stop_btn = st.button("⏹️ Stop Automation")
frame_display = st.empty()
latest_class_display = st.empty()
status_display = st.empty()

# ==================== AUTOMATION LOOP ====================
def automation_loop(cam_idx):
    global latest_frame, latest_class, automation_running
    cap = cv2.VideoCapture(cam_idx)
    prev_time = time.time()

    while automation_running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=0.7, imgsz=640)[0]

        latest_detected = None
        for box in results.boxes.data.tolist():
            x1,y1,x2,y2,score,cls_idx = box
            if score < 0.7:
                continue
            cls_name = all_classes[int(cls_idx)]
            latest_detected = cls_name
            pick_and_place(class_to_box[cls_name])
            break

        latest_frame = results.plot()
        latest_class = latest_detected

        if debug_mode:
            curr_time = time.time()
            fps = 1/(curr_time-prev_time)
            prev_time = curr_time
            st.sidebar.write(f"FPS: {fps:.1f}")

    cap.release()
    move_home()

# ==================== BUTTON HANDLERS ====================
if start_btn and not automation_running:
    automation_running = True

    # Move arm to HOME simultaneously and open gripper
    move_home(open_gripper=True)
    time.sleep(0.2)  # stabilization delay

    # Start automation thread
    threading.Thread(target=automation_loop, args=(camera_index,), daemon=True).start()

if stop_btn:
    automation_running = False
    move_home()
    st.success("✅ Automation stopped and arm returned home")

# ==================== UI MONITORING LOOP ====================
while True:
    if latest_frame is not None:
        frame_display.image(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)
    if latest_class is not None:
        latest_class_display.markdown(f"### 🥬 Latest Detection: {latest_class} (≥70%)")
        status_display.success("Automation running...")
    else:
        status_display.info("Waiting for vegetables...")
    time.sleep(0.05)
