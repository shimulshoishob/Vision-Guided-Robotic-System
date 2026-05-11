# ============================================================
# SINGLE FILE ROBOTIC AUTOMATION — BASELINE
# (Arm logic preserved exactly as given)
# ============================================================

import cv2
import time
import threading
import yaml
from pathlib import Path
from enum import Enum

from ultralytics import YOLO
import xarm

# ============================================================
# CONFIG PATHS
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
# ROBOT & MOTION PARAMETERS (UNCHANGED)
# ============================================================

HOME = {1:100, 2:500, 3:250, 4:820, 5:640, 6:500}
NUM_BOXES = 9
LEFT_LIMIT = 900
RIGHT_LIMIT = 100

box_positions = [
    int(LEFT_LIMIT - i*(LEFT_LIMIT-RIGHT_LIMIT)/(NUM_BOXES-1))
    for i in range(NUM_BOXES)
]

GRIPPER_OPEN = 150
GRIPPER_CLOSE = 250

BOX_OFFSETS = {
    0:(50,30,-60,-250), 1:(40,30,-100,-280), 2:(30,100,-200,-360),
    3:(20,100,-200,-360), 4:(30,100,-200,-360), 5:(-20,100,-200,-360),
    6:(-30,100,-200,-360), 7:(-40,30,-100,-280), 8:(-50,30,-60,-250)
}

def build_duration_map(general=500, link4=600, link5=700):
    return {
        1:general, 2:general, 3:general,
        4:link4, 5:link5, 6:general,
        "default":general
    }

# ============================================================
# LOAD CLASS YAML (UNCHANGED LOGIC)
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
# ROBOT HELPERS (UNCHANGED)
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
# PICK & PLACE (UNCHANGED)
# ============================================================

def pick_and_place_class(duration_map, cls_name):
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

# ============================================================
# MAIN CONTROL LOOP (FSM)
# ============================================================

state = SystemState.IDLE
cap = cv2.VideoCapture(CAMERA_INDEX)

print("[System] Starting main loop")

while True:

    if state == SystemState.IDLE:
        state = SystemState.DETECTING
        continue

    if state == SystemState.DETECTING:

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results = yolo_model(frame, conf=CONF_THRESH, imgsz=640)[0]

        detected_class = None
        for box in results.boxes.data.tolist():
            _,_,_,_,score,cls_idx = box
            if score < CONF_THRESH:
                continue
            cls_idx = int(cls_idx)
            if cls_idx < len(all_classes):
                detected_class = all_classes[cls_idx]
                break

        if detected_class in class_to_box:
            print(f"[Detect] {detected_class}")
            state = SystemState.ARM_BUSY
            target_class = detected_class

    if state == SystemState.ARM_BUSY:
        print("[FSM] Arm busy — detection locked")
        pick_and_place_class(build_duration_map(), target_class)
        state = SystemState.DETECTING

    if state == SystemState.SHUTDOWN:
        break

# ============================================================
# CLEAN EXIT
# ============================================================

cap.release()
move_all_home(build_duration_map())
print("[System] Shutdown complete")
