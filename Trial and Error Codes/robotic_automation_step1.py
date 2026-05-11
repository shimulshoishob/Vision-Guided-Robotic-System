import cv2
import time
import yaml
from pathlib import Path
from enum import Enum
import threading
from queue import Queue

from ultralytics import YOLO
import xarm

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/best.pt")
YAML_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/data.yaml")
CAMERA_INDEX = 1
CONF_THRESH = 0.7

# ============================================================
# FSM STATES
# ============================================================

class SystemState(Enum):
    IDLE = 0
    DETECTING = 1
    ARM_BUSY = 2
    SHUTDOWN = 3

# ============================================================
# ROBOT PARAMS
# ============================================================

HOME = {1:100, 2:500, 3:250, 4:820, 5:640, 6:500}
NUM_BOXES = 9
LEFT_LIMIT = 900
RIGHT_LIMIT = 100

box_positions = [int(LEFT_LIMIT - i*(LEFT_LIMIT-RIGHT_LIMIT)/(NUM_BOXES-1)) for i in range(NUM_BOXES)]
GRIPPER_OPEN = 150
GRIPPER_CLOSE = 250

BOX_OFFSETS = {
    0:(50,30,-60,-250), 1:(40,30,-100,-280), 2:(30,100,-200,-360),
    3:(20,100,-200,-360), 4:(30,100,-200,-360), 5:(-20,100,-200,-360),
    6:(-30,100,-200,-360), 7:(-40,30,-100,-280), 8:(-50,30,-60,-250)
}

def build_duration_map(general=500, link4=600, link5=700):
    return {1:general,2:general,3:general,4:link4,5:link5,6:general,"default":general}

# ============================================================
# LOAD CLASSES
# ============================================================

if not YAML_PATH.exists():
    cfg = {"names":[]}
else:
    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)

all_classes = cfg.get("names", [])

fresh_classes = [c for c in all_classes if "Fresh" in c]
damaged_classes = [c for c in all_classes if "Damaged" in c]

class_to_box = {}
for i, c in enumerate(fresh_classes):
    class_to_box[c] = i
for c in damaged_classes:
    class_to_box[c] = 8
if len(fresh_classes) > 8:
    for c in fresh_classes[8:]:
        class_to_box[c] = 7

# ============================================================
# CONNECT ROBOT
# ============================================================

try:
    robot = xarm.Controller("USB")
    print("[Robot] Connected")
except:
    robot = None
    print("[Robot] NOT connected")

# ============================================================
# ROBOT HELPERS
# ============================================================

def clamp_target(pos):
    return max(0, min(1000, int(pos)))

def reverse_if_out_of_bounds(home, offset):
    t = home + offset
    if t < 0 or t > 1000:
        t = home - offset
    return clamp_target(t)

def move_servo(servo_id, position, duration_ms):
    pos = clamp_target(position)
    if robot:
        robot.setPosition(servo_id, pos, duration=duration_ms, wait=False)
    time.sleep(duration_ms / 1000)

def move_multiple(servos_targets, duration_map):
    max_d = 0
    for sid, pos in servos_targets.items():
        d = duration_map.get(sid, duration_map["default"])
        if robot:
            robot.setPosition(sid, clamp_target(pos), duration=d, wait=False)
        max_d = max(max_d, d)
    time.sleep(max_d / 1000)

def move_all_home(duration_map, open_gripper=True):
    move_multiple({5:HOME[5],4:HOME[4],3:HOME[3]}, duration_map)
    move_servo(2, HOME[2], duration_map.get(2,500))
    if open_gripper:
        move_servo(1, GRIPPER_OPEN, duration_map.get(1,500))
    move_servo(6, HOME[6], duration_map.get(6,500))

# ============================================================
# PICK & PLACE FUNCTION
# ============================================================

def pick_and_place_class(cls_name):
    duration_map = build_duration_map()
    if cls_name not in class_to_box:
        return False
    box_idx = class_to_box[cls_name]
    pick_idx = 4

    # PICK
    move_servo(6, box_positions[pick_idx], duration_map.get(6,500))
    move_servo(1, GRIPPER_OPEN, duration_map.get(1,500))
    move_multiple({
        5:reverse_if_out_of_bounds(HOME[5],-200),
        4:reverse_if_out_of_bounds(HOME[4],20),
        3:reverse_if_out_of_bounds(HOME[3],-30)
    }, duration_map)
    move_servo(2, reverse_if_out_of_bounds(HOME[2],50), duration_map.get(2,500))
    move_servo(1, GRIPPER_CLOSE, duration_map.get(1,500))

    # RETURN
    move_multiple({5:HOME[5],4:HOME[4],3:HOME[3]}, duration_map)
    move_servo(2, HOME[2], duration_map.get(2,500))
    move_servo(6, HOME[6], duration_map.get(6,500))

    # PLACE
    off = BOX_OFFSETS.get(box_idx, BOX_OFFSETS[4])
    move_servo(6, box_positions[box_idx], duration_map.get(6,500))
    move_multiple({
        5:reverse_if_out_of_bounds(HOME[5],off[3]),
        4:reverse_if_out_of_bounds(HOME[4],off[2]),
        3:reverse_if_out_of_bounds(HOME[3],off[1])
    }, duration_map)
    move_servo(2, reverse_if_out_of_bounds(HOME[2],off[0]), duration_map.get(2,500))
    move_servo(1, GRIPPER_OPEN, duration_map.get(1,500))
    time.sleep(0.15)
    move_all_home(duration_map)
    return True

# ============================================================
# LOAD YOLO
# ============================================================

yolo_model = YOLO(str(MODEL_PATH))
print("[YOLO] Model loaded")

# ============================================================
# THREAD-SAFE QUEUE FOR DETECTIONS
# ============================================================

det_queue = Queue(maxsize=1)  # only keep latest
robot_active = threading.Event()  # control start/stop

def arm_worker():
    while True:
        cls_name = det_queue.get()
        if cls_name is None:
            break
        # Wait until robot is active
        robot_active.wait()
        print(f"[ARM THREAD] Picking {cls_name}")
        pick_and_place_class(cls_name)
        det_queue.task_done()

# Start arm thread
arm_thread = threading.Thread(target=arm_worker, daemon=True)
arm_thread.start()

# ============================================================
# MAIN LOOP — LIVE DISPLAY + YOLO DETECTION
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)
prev_time = time.time()

print("[System] Starting live detection loop")
print("Press 's' to START robot, 'p' to PAUSE robot, 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    results = yolo_model(frame, conf=CONF_THRESH, imgsz=640)[0]
    annotated_frame = results.plot()

    detected_class = None
    for box in results.boxes.data.tolist():
        _,_,_,_,score,cls_idx = box
        if score < CONF_THRESH:
            continue
        cls_idx = int(cls_idx)
        if cls_idx < len(all_classes):
            detected_class = all_classes[cls_idx]
            break

    # Send to robot queue if not full
    if detected_class in class_to_box and det_queue.empty():
        det_queue.put(detected_class)
        print(f"[DETECT] Queued {detected_class}")

    # Show FPS
    fps = 1/(time.time()-prev_time)
    prev_time = time.time()
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live Detection", annotated_frame)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):  # Start robot
        robot_active.set()
        print("[CONTROL] Robot STARTED")
    elif key == ord('p'):  # Pause robot
        robot_active.clear()
        print("[CONTROL] Robot PAUSED")

# ============================================================
# CLEAN EXIT
# ============================================================

cap.release()
cv2.destroyAllWindows()
det_queue.put(None)  # signal thread to exit
arm_thread.join()
move_all_home(build_duration_map())
print("[System] Shutdown complete")
