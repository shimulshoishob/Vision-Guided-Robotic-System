"""
xArm 1S Pick-and-Place Console Automation (NO SERVO 3)
"""

import time
import xarm
import yaml

# =========================================================
# Load YAML class info
# =========================================================
yaml_path = "D:/EWU/10th Semester/CSE475/LABS/data.yaml"

with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

all_classes = cfg['names']
fresh_classes = [c for c in all_classes if "Fresh" in c]
damaged_classes = [c for c in all_classes if "Damaged" in c]

# Class mapping
class_to_box = {fresh_classes[i]: i for i in range(len(fresh_classes))}
for c in damaged_classes:
    class_to_box[c] = 8

# =========================================================
# CONFIG (SERVO 3 REMOVED)
# =========================================================
HOME = {1: 100, 2: 500, 4: 820, 5: 640, 6: 500}

NUM_BOXES = 9
LEFT_LIMIT = 900
RIGHT_LIMIT = 100

box_positions = [
    int(LEFT_LIMIT - i * (LEFT_LIMIT - RIGHT_LIMIT) / (NUM_BOXES - 1))
    for i in range(NUM_BOXES)
]

GRIPPER_OPEN = 150
GRIPPER_CLOSE = 410

# (link2, link4, link5)
BOX_OFFSETS = {
    0: (50, -150, -260),
    1: (40, -190, -290),
    2: (30, -300, -370),
    3: (20, -300, -370),
    4: (30, -320, -350),
    5: (-20, -300, -370),
    6: (-30, -300, -370),
    7: (-40, -190, -290),
    8: (-50, -150, -260)
}

# =========================================================
# CONNECT ROBOT
# =========================================================
try:
    robot = xarm.Controller('USB')
    print("✅ Connected to xArm (USB)")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    exit(1)

# =========================================================
# HELPERS
# =========================================================
def clamp_target(pos):
    return max(0, min(1000, int(pos)))

def reverse_if_out_of_bounds(home, offset):
    target = home + offset
    if target < 0 or target > 1000:
        target = home - offset
    return clamp_target(target)

def move_servo(servo_id, position, duration_ms):
    robot.setPosition(servo_id, clamp_target(position), duration=duration_ms, wait=False)
    time.sleep(duration_ms / 1000.0)

def move_multiple(servos_targets, duration_map):
    max_d = 0
    for sid, pos in servos_targets.items():
        d = duration_map.get(sid, duration_map.get("default", 500))
        robot.setPosition(sid, clamp_target(pos), duration=d, wait=False)
        max_d = max(max_d, d)
    time.sleep(max_d / 1000.0)

def build_duration_map(general=500, link4=600, link5=700):
    return {
        1: general,
        2: general,
        4: link4,
        5: link5,
        6: general,
        "default": general
    }

# =========================================================
# SEQUENCES (NO SERVO 3)
# =========================================================
def move_all_home(duration_map, open_gripper=True):
    move_multiple({5: HOME[5], 4: HOME[4]}, duration_map)
    move_servo(2, HOME[2], duration_map[2])
    if open_gripper:
        move_servo(1, GRIPPER_OPEN, duration_map[1])
    move_servo(6, HOME[6], duration_map[6])

def pick_and_place_class(duration_map, cls_name):
    if cls_name not in class_to_box:
        print("❌ Class not recognized.")
        return

    box_idx = class_to_box[cls_name]

    # ---------------- PICK ----------------
    link2_t = HOME[2]
    link4_t = reverse_if_out_of_bounds(HOME[4], 20)
    link5_t = reverse_if_out_of_bounds(HOME[5], -190)

    pick_idx = 4

    move_servo(6, box_positions[pick_idx], duration_map[6])
    move_servo(1, GRIPPER_OPEN, duration_map[1])

    move_multiple({
        5: link5_t,
        4: link4_t
    }, duration_map)

    move_servo(2, link2_t, duration_map[2])

    move_servo(1, GRIPPER_CLOSE, duration_map[1])

    move_multiple({
        5: HOME[5],
        4: HOME[4]
    }, duration_map)

    move_servo(2, HOME[2], duration_map[2])
    move_servo(6, HOME[6], duration_map[6])

    print(f"✅ Picked object for '{cls_name}'")

    # ---------------- PLACE ----------------
    off = BOX_OFFSETS[box_idx]

    link2_t = reverse_if_out_of_bounds(HOME[2], off[0])
    link4_t = reverse_if_out_of_bounds(HOME[4], off[1])
    link5_t = reverse_if_out_of_bounds(HOME[5], off[2])

    move_servo(6, box_positions[box_idx], duration_map[6])

    move_multiple({
        5: link5_t,
        4: link4_t
    }, duration_map)

    move_servo(2, link2_t, duration_map[2])

    move_servo(1, GRIPPER_OPEN, duration_map[1])
    time.sleep(0.2)

    move_all_home(duration_map)

    print(f"✅ Placed in box {box_idx} and returned HOME")

# =========================================================
# MAIN MENU
# =========================================================
if __name__ == "__main__":
    duration_map = build_duration_map()

    while True:
        print("\nSelect action:")
        print("1. Move all to HOME")
        print("2. Manual PICK & PLACE")
        print("3. EXIT")

        choice = input("Enter choice [1-3]: ").strip()

        if choice == "1":
            move_all_home(duration_map)
            print("Returned to HOME")

        elif choice == "2":
            while True:
                print("\nSelect class:")
                for i, cls in enumerate(fresh_classes):
                    print(f"{i+1}. {cls}")
                print("9. Damaged")
                print("0. Back")

                cls_choice = input("Choice: ").strip()

                if cls_choice == "0":
                    break

                if cls_choice in [str(i+1) for i in range(9)]:
                    idx = int(cls_choice) - 1
                    if idx < 8:
                        cls_name = fresh_classes[idx]
                    else:
                        cls_name = damaged_classes[0] if damaged_classes else "Damaged"

                    pick_and_place_class(duration_map, cls_name)
                else:
                    print("Invalid choice")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid input")