"""
============================================================
   xArm 1S – Full 6-Servo Pick-and-Place Sweep (Fixed)
============================================================

This script demonstrates full pick-and-place motion control of the HiWonder
xArm 1S. It does the following:

    1. Connects to the robot
    2. Moves ALL 6 servos to a known HOME position
    3. Performs a pick-and-place style sweep across 9 box positions
       (base rotation = link6)
    4. Returns all servos to HOME at the end

Author: YOUR NAME
Date: TODAY
============================================================
"""

import time
import xarm

# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

HOME = {
    1: 100,     # Gripper open
    2: 500,     # link2 neutral
    3: 250,     # link3 lifted
    4: 820,     # link4 lifted
    5: 640,     # link5 neutral
    6: 500      # link6 center/home (base facing forward)
}

LEFT_LIMIT  = 900
RIGHT_LIMIT = 100
NUM_BOXES = 9
SLOW_DURATION = 500  # ms

box_positions = [
    int(LEFT_LIMIT - i * (LEFT_LIMIT - RIGHT_LIMIT) / (NUM_BOXES - 1))
    for i in range(NUM_BOXES)
]

# ============================================================
# 2. ROBOT CONNECTION
# ============================================================

def connect_robot():
    try:
        robot = xarm.Controller('USB')
        print("✅ Connected to xArm via USB\n")
        return robot
    except Exception as e:
        print("❌ Failed to connect to xArm:", e)
        exit()

robot = connect_robot()

# ============================================================
# 3. SERVO HELPER FUNCTIONS
# ============================================================

def reverse_if_out_of_bounds(home, offset):
    """
    Calculate servo target relative to home.
    If target exceeds limits (0-1000), reverse direction.
    """
    target = home + offset
    if target < 0 or target > 1000:
        target = home - offset  # reverse direction
        print(f"⚠️ Servo target out of bounds. Reversing direction: {target}")
    return max(0, min(1000, target))

def move_servo(servo_id, position, duration=SLOW_DURATION):
    position = max(0, min(1000, position))
    print(f"→ Moving servo {servo_id} to {position} (duration={duration}ms)")
    robot.setPosition(servo_id, position, duration=duration, wait=False)
    time.sleep(duration / 1000)
    print(f"   ✔ Servo {servo_id} reached {position}\n")

def move_all_home():
    print("🏠 Moving ALL servos to HOME position...\n")
    for servo_id, pos in HOME.items():
        move_servo(servo_id, pos)
    print("✔ All servos are now at HOME\n")
    time.sleep(1)

# ============================================================
# 4. FULL PICK-AND-PLACE SWEEP
# ============================================================

def sweep_all_links_pick_place():
    print(f"📦 Sweeping all 6 servos with pick-and-place motion...\n")

    link1_home = HOME[1]
    GRIPPER_OPEN = 150
    GRIPPER_CLOSE = 200

    for idx, link6_pos in enumerate(box_positions, start=1):
        print(f"📍 Box {idx}: Preparing pick")
        
        # Step 1: Calculate all other servo targets safely
        
        link2_target = reverse_if_out_of_bounds(HOME[2], 30)
        link3_target = reverse_if_out_of_bounds(HOME[3], 100)
        link4_target = reverse_if_out_of_bounds(HOME[4], -200)
        link5_target = reverse_if_out_of_bounds(HOME[5], -360)
        
        # Step 2: Move link6 to box position
        move_servo(6, link6_pos, duration=SLOW_DURATION)
        print(f"   → Link6 moved to position {link6_pos}")
        # Step 3: Close gripper to grip target
        link1_target = reverse_if_out_of_bounds(link1_home, -GRIPPER_CLOSE)
        move_servo(1, link1_target, duration=SLOW_DURATION)
        print(f"   → Gripper closed to pick")

        # Step 4: Move link4 and link5 to picking positions
        move_servo(3, link3_target, duration=SLOW_DURATION)
        move_servo(4, link4_target, duration=SLOW_DURATION)
        move_servo(5, link5_target, duration=SLOW_DURATION)
        # Step 5: Move link2 and link3 to picking positions
        move_servo(2, link2_target, duration=SLOW_DURATION)
        
        
        # Step 6: Open gripper to place position
        move_servo(1, GRIPPER_OPEN, duration=SLOW_DURATION)
        print(f"   → Gripper opened to pick")
        time.sleep(0.2)

        print(f"📍 Box {idx}: Preparing to pick")
        # Step 6: Return link2-5 to HOME (gripper stays at grip target)
        move_servo(5, HOME[5], duration=SLOW_DURATION)
        move_servo(3, HOME[3], duration=SLOW_DURATION)
        move_servo(2, HOME[2], duration=SLOW_DURATION)
        move_servo(4, HOME[4], duration=SLOW_DURATION)
        print(f"   ← Link2-5 returned to HOME")

        time.sleep(0.2)  # small pause before next box

    # Step 7: After full sweep, return gripper to HOME
    move_servo(1, link1_home, duration=SLOW_DURATION)
    print(f"✅ Gripper returned to HOME")

    print("🏁 Full place sweep complete!\n")


# ============================================================
# 5. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("\n=====================================")
    print("   xArm Full 6-Servo Pick-and-Place Sweep")
    print("=====================================\n")

    move_all_home()
    sweep_all_links_pick_place()

    # Return all servos to HOME at the end
    print("➡ Returning all servos to HOME...\n")
    for servo_id, pos in HOME.items():
        move_servo(servo_id, pos, duration=SLOW_DURATION)

    print("✅ Sequence complete. Robot is back at HOME.\n")
