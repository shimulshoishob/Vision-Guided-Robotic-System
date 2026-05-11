"""
Automated Streamlit Dashboard for xArm 1S Pick-and-Place
Loops through pick → place (boxes 1–9) → home until stop is pressed.
"""

import streamlit as st
import time
import threading
import xarm

# -------------------------------
# 1. GLOBAL CONFIG
# -------------------------------

HOME = {1: 100, 2: 500, 3: 250, 4: 820, 5: 640, 6: 500}
LEFT_LIMIT = 900
RIGHT_LIMIT = 100
NUM_BOXES = 9
SLOW_DURATION = 500

box_positions = [int(LEFT_LIMIT - i * (LEFT_LIMIT - RIGHT_LIMIT) / (NUM_BOXES - 1))
                 for i in range(NUM_BOXES)]

GRIPPER_OPEN = 150
GRIPPER_CLOSE = 200

# BOX OFFSETS
BOX_OFFSETS = {
    0: (50, 30, -60, -250),
    1: (40, 30, -100, -280),
    2: (30, 100, -200, -360),
    3: (20, 100, -200, -360),
    4: (30, 100, -200, -360),
    5: (-20, 100, -200, -360),
    6: (-30, 100, -200, -360),
    7: (-40, 30, -100, -280),
    8: (-50, 30, -60, -250)
}

# GLOBAL FLAG FOR AUTOMATION
if "run_automation" not in st.session_state:
    st.session_state.run_automation = False

# -------------------------------
# 2. CONNECT ROBOT
# -------------------------------

try:
    robot = xarm.Controller('USB')
    st.success("✅ Connected to xArm via USB")
except Exception as e:
    st.error(f"❌ Failed to connect to xArm: {e}")
    st.stop()


# -------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------

def reverse_if_out_of_bounds(home, offset):
    target = home + offset
    if target < 0 or target > 1000:
        target = home - offset
    return max(0, min(1000, target))

def move_servo(servo_id, position, duration=SLOW_DURATION):
    position = max(0, min(1000, position))
    robot.setPosition(servo_id, position, duration=duration, wait=False)
    time.sleep(duration / 1000)

def move_all_home():
    for servo_id in [5, 2, 3, 4]:
        move_servo(servo_id, HOME[servo_id])
    move_servo(1, HOME[1])
    move_servo(6, HOME[6])


# -------------------------------
# 4. PICK FUNCTION
# -------------------------------

def pick_object():
    link1_home = HOME[1]
    idx = 4  # picking position (center)

    link2_target = reverse_if_out_of_bounds(HOME[2], 50)
    link3_target = reverse_if_out_of_bounds(HOME[3], -30)
    link4_target = reverse_if_out_of_bounds(HOME[4], 20)
    link5_target = reverse_if_out_of_bounds(HOME[5], -200)

    move_servo(6, box_positions[idx])
    move_servo(1, GRIPPER_OPEN)
    move_servo(4, link4_target)
    move_servo(5, link5_target)
    move_servo(2, link2_target)
    move_servo(3, link3_target)

    move_servo(1, reverse_if_out_of_bounds(link1_home, -GRIPPER_CLOSE))
    time.sleep(0.2)
    move_all_home()


# -------------------------------
# 5. PLACE FUNCTION
# -------------------------------

def place_object(box_idx, offsets):
    move_servo(6, box_positions[box_idx])

    link2_target = reverse_if_out_of_bounds(HOME[2], offsets[0])
    link3_target = reverse_if_out_of_bounds(HOME[3], offsets[1])
    link4_target = reverse_if_out_of_bounds(HOME[4], offsets[2])
    link5_target = reverse_if_out_of_bounds(HOME[5], offsets[3])

    move_servo(3, link3_target)
    move_servo(4, link4_target)
    move_servo(5, link5_target)
    move_servo(2, link2_target)

    move_servo(1, GRIPPER_OPEN)
    time.sleep(0.2)
    move_all_home()


# -------------------------------
# 6. AUTOMATION LOOP
# -------------------------------

def automation_loop():
    while st.session_state.run_automation:
        for box in range(NUM_BOXES):
            if not st.session_state.run_automation:
                break

            move_all_home()
            pick_object()
            place_object(box, BOX_OFFSETS[box])

    # After STOP → go home
    move_all_home()


# -------------------------------
# 7. STREAMLIT UI
# -------------------------------

st.title("🤖 xArm 1S Automated Pick-and-Place System")

if st.button("🏁 START Automation"):
    st.session_state.run_automation = True
    threading.Thread(target=automation_loop, daemon=True).start()
    st.success("🚀 Automation Started")

if st.button("🛑 STOP Automation"):
    st.session_state.run_automation = False
    st.warning("⛔ Automation Stopping... Returning to HOME")


st.subheader("Manual Control (Optional)")
if st.button("🏠 Move All to HOME"):
    move_all_home()
    st.success("✅ All servos moved to HOME")