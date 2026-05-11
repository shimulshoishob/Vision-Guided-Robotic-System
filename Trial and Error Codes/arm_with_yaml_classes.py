


"""
xArm 1S Pick-and-Place Console Automation
- Pick object and place class-wise in one step for manual control
- Class-wise boxes: 8 fresh + 1 damaged
- Safe HOME return after placement
"""

import threading
import time
import xarm
import yaml

# =========================================================
# Load YAML class info
# =========================================================
yaml_path = "D:/EWU/10th Semester/CSE475/LABS/data.yaml"  # replace with your actual yaml path
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

all_classes = cfg['names']
fresh_classes = [c for c in all_classes if "Fresh" in c]
damaged_classes = [c for c in all_classes if "Damaged" in c]

# Map class to box index
# Fresh classes: boxes 0-7
# Damaged classes: box 8
class_to_box = {fresh_classes[i]: i for i in range(len(fresh_classes))}
for c in damaged_classes:
    class_to_box[c] = 8

# =========================================================
# Global configuration
# =========================================================
HOME = {1: 100, 2: 500, 3: 250, 4: 820, 5: 640, 6: 500}
NUM_BOXES = 9
LEFT_LIMIT = 900
RIGHT_LIMIT = 100
box_positions = [int(LEFT_LIMIT - i * (LEFT_LIMIT - RIGHT_LIMIT) / (NUM_BOXES - 1)) for i in range(NUM_BOXES)]

GRIPPER_OPEN = 150
GRIPPER_CLOSE = 410

BOX_OFFSETS = {
    0: (50, 30, -60, -240),
    1: (40, 30, -100, -270),
    2: (30, 100, -200, -350),
    3: (20, 100, -200, -350),
    4: (30, 100, -200, -350),
    5: (-20, 100, -200, -350),
    6: (-30, 100, -200, -350),
    7: (-40, 30, -100, -270),
    8: (-50, 30, -60, -240)
}

# =========================================================
# Connect to robot
# =========================================================
try:
    robot = xarm.Controller('USB')
    print("✅ Connected to xArm (USB)")
except Exception as e:
    print(f"❌ Failed to connect to xArm: {e}")
    exit(1)

# =========================================================
# Helpers
# =========================================================
def clamp_target(pos):
    return max(0, min(1000, int(pos)))

def reverse_if_out_of_bounds(home, offset):
    target = home + offset
    if target < 0 or target > 1000:
        target = home - offset
    return clamp_target(target)

def move_servo(servo_id, position, duration_ms):
    pos = clamp_target(position)
    robot.setPosition(servo_id, pos, duration=duration_ms, wait=False)
    time.sleep(duration_ms / 1000.0)

def move_multiple(servos_targets, duration_map):
    max_d = 0
    for sid, pos in servos_targets.items():
        d = duration_map.get(sid, duration_map.get("default", 500))
        robot.setPosition(sid, clamp_target(pos), duration=d, wait=False)
        if d > max_d:
            max_d = d
    time.sleep(max_d / 1000.0)

def build_duration_map(general=500, link4=600, link5=700):
    return {1: general, 2: general, 3: general, 4: link4, 5: link5, 6: general, "default": general}

# =========================================================
# Sequences
# =========================================================
def move_all_home(duration_map, open_gripper=True):
    move_multiple({5: HOME[5], 4: HOME[4], 3: HOME[3]}, duration_map)
    move_servo(2, HOME[2], duration_map.get(2, duration_map["default"]))
    if open_gripper:
        move_servo(1, GRIPPER_OPEN, duration_map.get(1, duration_map["default"]))
    move_servo(6, HOME[6], duration_map.get(6, duration_map["default"]))

def pick_and_place_class(duration_map, cls_name):
    if cls_name not in class_to_box:
        print("❌ Class not recognized.")
        return

    box_idx = class_to_box[cls_name]
    # ----------------
    # Pick object (fixed pick zone)
    # ----------------
    link2_t = reverse_if_out_of_bounds(HOME[2], 0)
    link3_t = reverse_if_out_of_bounds(HOME[3], -30)
    link4_t = reverse_if_out_of_bounds(HOME[4], 20)
    link5_t = reverse_if_out_of_bounds(HOME[5], -190)

    pick_idx = 4  # default pick zone
    move_servo(6, box_positions[pick_idx], duration_map[6])
    move_servo(1, GRIPPER_OPEN, duration_map[1])
    move_multiple({5: link5_t, 4: link4_t, 3: link3_t}, duration_map)
    move_servo(2, link2_t, duration_map[2])
    move_servo(1, reverse_if_out_of_bounds(HOME[1], -GRIPPER_CLOSE), duration_map[1])
    move_multiple({5: HOME[5], 4: HOME[4], 3: HOME[3]}, duration_map)
    move_servo(2, HOME[2], duration_map[2])
    move_servo(6, HOME[6], duration_map[6])
    print(f"✅ Picked object for class '{cls_name}'.")

    # ----------------
    # Place in class-specific box
    # ----------------
    link2_t = reverse_if_out_of_bounds(HOME[2], BOX_OFFSETS[box_idx][0])
    link3_t = reverse_if_out_of_bounds(HOME[3], BOX_OFFSETS[box_idx][1])
    link4_t = reverse_if_out_of_bounds(HOME[4], BOX_OFFSETS[box_idx][2])
    link5_t = reverse_if_out_of_bounds(HOME[5], BOX_OFFSETS[box_idx][3])

    move_servo(6, box_positions[box_idx], duration_map[6])
    move_multiple({5: link5_t, 4: link4_t, 3: link3_t}, duration_map)
    move_servo(2, link2_t, duration_map[2])
    move_servo(1, GRIPPER_OPEN, duration_map[1])
    time.sleep(0.15)
    move_all_home(duration_map)
    print(f"✅ Placed object in box for class '{cls_name}' and returned HOME.")

# =========================================================
# Main menu
# =========================================================
if __name__ == "__main__":
    duration_map = build_duration_map()
    while True:
        print("\nSelect action:")
        print("1. START automation (not implemented)")
        print("2. STOP automation (not implemented)")
        print("3. Move all to HOME")
        print("4. MANUAL PICK & PLACE (class-wise)")
        print("5. EXIT")
        choice = input("Enter choice [1-5]: ").strip()

        if choice == "1":
            print("⚠️ Automation not implemented in this snippet")
        elif choice == "2":
            print("⚠️ Stop automation not implemented in this snippet")
        elif choice == "3":
            move_all_home(duration_map)
            print("Returned to HOME")
        elif choice == "4":
            # Show class-wise menu
            while True:
                print("\nSelect class to pick & place:")
                for i, cls in enumerate(fresh_classes):
                    print(f"{i+1}. {cls} (Fresh)")
                print(f"9. Damaged Vegetables (all classes)")
                print("0. Back to Main Menu")
                cls_choice = input("Enter choice [0-9]: ").strip()
                if cls_choice == "0":
                    break
                if cls_choice in [str(i+1) for i in range(9)]:
                    cls_idx = int(cls_choice)-1
                    if cls_idx < 8:
                        cls_name = fresh_classes[cls_idx]
                    else:
                        cls_name = "Damaged"  # generic for all damaged
                    pick_and_place_class(duration_map, cls_name)
                else:
                    print("Invalid choice. Try again.")
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid input. Try again.")
