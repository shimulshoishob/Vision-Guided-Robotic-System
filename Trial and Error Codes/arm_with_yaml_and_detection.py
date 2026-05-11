import threading
import time
import xarm
import cv2
from ultralytics import YOLO
import yaml

# ----------------------
# Load YAML
# ----------------------
yaml_path = "D:/EWU/10th Semester/CSE475/LABS/data.yaml"
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

all_classes = cfg['names']
fresh_classes = [c for c in all_classes if "Fresh" in c]
damaged_classes = [c for c in all_classes if "Damaged" in c]

# 8 fresh boxes + 1 damaged box
class_to_box = {fresh_classes[i]: i for i in range(len(fresh_classes))}
for c in damaged_classes:
    class_to_box[c] = 8

# ----------------------
# xArm Setup
# ----------------------
HOME = {1:100,2:500,3:250,4:820,5:640,6:500}
NUM_BOXES = 9
LEFT_LIMIT = 900
RIGHT_LIMIT = 100
box_positions = [int(LEFT_LIMIT - i * (LEFT_LIMIT - RIGHT_LIMIT) / (NUM_BOXES - 1)) for i in range(NUM_BOXES)]

GRIPPER_OPEN = 150
GRIPPER_CLOSE = 400

BOX_OFFSETS = {
    0:(50,30,-60,-240),
    1:(40,30,-100,-270),
    2:(30,100,-200,-340),
    3:(20,100,-200,-340),
    4:(0,100,-200,-340),
    5:(-20,100,-200,-340),
    6:(-30,100,-200,-340),
    7:(-40,30,-100,-270),
    8:(-50,30,-60,-240)
}

try:
    robot = xarm.Controller('USB')
    print("✅ Connected to xArm")
except Exception as e:
    print(f"❌ Failed to connect to xArm: {e}")
    exit(1)

# ----------------------
# Thread globals
# ----------------------
AUTOMATION_RUNNING = False
STOP_REQUESTED = False
ACTION_LOCK = threading.Lock()  # ensures one pick/place at a time

# ----------------------
# Duration
# ----------------------
def build_duration_map(general=500, link4=600, link5=700):
    return {1:general,2:general,3:general,4:link4,5:link5,6:general,"default":general}

duration_map = build_duration_map()

# ----------------------
# Helper functions
# ----------------------
def clamp_target(pos):
    return max(0,min(1000,int(pos)))

def reverse_if_out_of_bounds(home, offset):
    target = home+offset
    if target<0 or target>1000:
        target = home-offset
    return clamp_target(target)

def move_servo(servo_id, pos, duration_ms):
    pos = clamp_target(pos)
    robot.setPosition(servo_id, pos, duration=duration_ms, wait=False)
    time.sleep(duration_ms/1000)

def move_multiple(servos_targets, duration_map):
    max_d = max([duration_map.get(s, duration_map.get("default",500)) for s in servos_targets])
    for s, p in servos_targets.items():
        d = duration_map.get(s, duration_map.get("default",500))
        robot.setPosition(s, clamp_target(p), duration=d, wait=False)
    time.sleep(max_d/1000)

def move_all_home(open_gripper=True):
    move_multiple({5:HOME[5],4:HOME[4],3:HOME[3]}, duration_map)
    move_servo(2, HOME[2], duration_map[2])
    if open_gripper:
        move_servo(1, GRIPPER_OPEN, duration_map[1])
    move_servo(6, HOME[6], duration_map[6])

def pick_and_place(box_idx):
    # Only one action at a time
    with ACTION_LOCK:
        link2 = reverse_if_out_of_bounds(HOME[2], 0)
        link3 = reverse_if_out_of_bounds(HOME[3], -30)
        link4 = reverse_if_out_of_bounds(HOME[4], 20)
        link5 = reverse_if_out_of_bounds(HOME[5], -190)

        # Pick
        move_servo(6, box_positions[4], duration_map[6])
        move_servo(1, GRIPPER_OPEN, duration_map[1])
        move_multiple({5:link5,4:link4,3:link3}, duration_map)
        move_servo(2, link2, duration_map[2])
        move_servo(1, reverse_if_out_of_bounds(HOME[1], -GRIPPER_CLOSE), duration_map[1])
        move_multiple({5:HOME[5],4:HOME[4],3:HOME[3]}, duration_map)
        move_servo(2, HOME[2], duration_map[2])
        move_servo(6, HOME[6], duration_map[6])

        # Place
        link2 = reverse_if_out_of_bounds(HOME[2], BOX_OFFSETS[box_idx][0])
        link3 = reverse_if_out_of_bounds(HOME[3], BOX_OFFSETS[box_idx][1])
        link4 = reverse_if_out_of_bounds(HOME[4], BOX_OFFSETS[box_idx][2])
        link5 = reverse_if_out_of_bounds(HOME[5], BOX_OFFSETS[box_idx][3])

        move_servo(6, box_positions[box_idx], duration_map[6])
        move_multiple({5:link5,4:link4,3:link3}, duration_map)
        move_servo(2, link2, duration_map[2])
        move_servo(1, GRIPPER_OPEN, duration_map[1])
        time.sleep(0.15)
        move_all_home()

# ----------------------
# YOLO model
# ----------------------
model_path = "D:/EWU/10th Semester/CSE475/LABS/best.pt"
model = YOLO(model_path)

# ----------------------
# Automation thread
# ----------------------
def automation_loop():
    global AUTOMATION_RUNNING, STOP_REQUESTED
    cap = cv2.VideoCapture(1)  # Replace 0 with Iriun URL if needed

    while AUTOMATION_RUNNING:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[1]

        detected = False
        for r in results.boxes.data.tolist():
            x1,y1,x2,y2,score,cls = r
            cls_name = all_classes[int(cls)]
            if cls_name not in class_to_box:
                continue
            if score < 0.7:
                continue

            box_idx = class_to_box[cls_name]
            detected = True

            # Draw bbox
            cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame,f"{cls_name} {score:.2f}",(int(x1),int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            # Trigger pick and place only once per detection
            pick_and_place(box_idx)
            if STOP_REQUESTED:
                AUTOMATION_RUNNING = False
                STOP_REQUESTED = False
                break

        # Show live frame
        cv2.imshow("xArm Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            AUTOMATION_RUNNING = False
            break

        # If no object detected, do nothing (wait)
        if not detected:
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    move_all_home()

# ----------------------
# Control functions
# ----------------------
def start_automation():
    global AUTOMATION_RUNNING
    if AUTOMATION_RUNNING:
        print("Already running")
        return
    AUTOMATION_RUNNING = True
    t = threading.Thread(target=automation_loop, daemon=True)
    t.start()
    print("✅ Automation started. Press 'q' in window to exit.")

def stop_automation():
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("⏹ Stop requested. Will finish current pick/place safely.")

# ----------------------
# Main menu
# ----------------------
if __name__ == "__main__":
    while True:
        print("\nSelect action:")
        print("1. START live automated pick & place")
        print("2. STOP automation (safe)")
        print("3. Move all to HOME")
        print("4. EXIT")
        choice = input("Enter choice [1-4]: ").strip()
        if choice == "1":
            start_automation()
        elif choice == "2":
            stop_automation()
        elif choice == "3":
            move_all_home()
            print("Returned HOME")
        elif choice == "4":
            stop_automation()
            print("Exiting...")
            break
        else:
            print("Invalid input. Try again.")
