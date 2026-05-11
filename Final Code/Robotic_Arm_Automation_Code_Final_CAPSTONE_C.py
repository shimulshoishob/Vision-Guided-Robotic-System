import cv2
import numpy as np
import time
import xarm
from threading import Thread, Lock
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import yaml

MODEL_PATH = "D:/EWU/10th Semester/CSE475/LABS/best.pt"
YAML_PATH  = "D:/EWU/10th Semester/CSE475/LABS/data.yaml"

model = YOLO(MODEL_PATH)

cfg = yaml.safe_load(open(YAML_PATH))
all_classes = cfg["names"]

selected_id = None
input_buffer = ""
detection_dict = {}

# =========================
# TRACK HISTORY (FOR TRAIL)
# =========================
track_history = {}
MAX_TRAIL = 60   # number of past points

# =========================
# PICK STATE MACHINE
# =========================
robot_busy = False
current_target = None
pick_attempts = 0
retry_pending = False
stop_pick = False
last_pick_time = 0
task_id = 0
task_lock = Lock()
COOLDOWN = 1.2   # seconds
# =========================
# NEW STABILITY + TRACKING FIXES
# =========================
TRACK_THRESH = 0.2
PICK_THRESH = 0.7
SMOOTH_ALPHA = 0.7
STABLE_FRAMES = 5
MAX_LOST = 10

smooth_centers = {}
stable_counter = {}
lost_tracks = {}
last_results = None
# =========================
# 🔥 ADVANCED VALIDATION
# =========================
missing_counter = {}
MISSING_THRESHOLD = 3

VERIFY_TIME = 0.3
verify_start_time = None
# =========================
# 📊 METRICS SYSTEM
# =========================
metrics = {
    "start_time": time.time(),
    "frame_count": 0,

    # detection
    "detections_total": 0,
    "stable_detections": 0,

    # tracking
    "track_lost": 0,

    # pick
    "total_attempts": 0,       # includes retries
    "total_pick_cycles": 0,    # already exists but use properly
    "successful_picks": 0,
    "retry_counts": [],
    "max_retry_fail": 0,

    # timing
    "pick_durations": [],
    "last_pick_start": 0,
    "event_log": [],  # (timestamp, success/fail)

    # trajectory
    "trajectory_lengths": [],
    "valid_trajectory": 0,

    # energy
    "battery_voltage": 7.5,  # constant assumption
    # ===== NEW JOURNAL METRICS =====
    "disappearance_success": 0,
    "class_stats": {},
    "tasks_last_minute": []
}
MAX_RETRY = 3
VERIFY_DELAY = 1.2


# =========================================================
# PLACEMENT SYSTEM
# =========================================================

fresh_classes = [c for c in all_classes if "Fresh" in c]
damaged_classes = [c for c in all_classes if "Damaged" in c]

class_to_box = {fresh_classes[i]: i for i in range(len(fresh_classes))}
for c in damaged_classes:
    class_to_box[c] = 8

NUM_BOXES = 9
LEFT_LIMIT = 900
RIGHT_LIMIT = 100

box_positions = [
    int(LEFT_LIMIT - i * (LEFT_LIMIT - RIGHT_LIMIT) / (NUM_BOXES - 1))
    for i in range(NUM_BOXES)
]

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

def reverse_if_out_of_bounds(home, offset):
    target = home + offset
    if target < 0 or target > 1000:
        target = home - offset
    return int(max(0, min(1000, target)))

# =========================================================
# CONFIG
# =========================================================

HOME = {2:500, 4:820, 5:700, 6:500}
MOVE_TIME = 1200
CAMERA_INDEX = 1

WORK_W = 28
WORK_H = 14

ROBOT_ORIGIN_X = 14
ROBOT_ORIGIN_Y = -12

SERVO_LIMITS = {
    4: (400, 950),
    5: (275, 520),
    6: (200, 700)
}

SERVO6_MIN = 100
SERVO6_MAX = 850
SERVO6_CENTER = 500

R_SCALE = 1.05

# =========================================================
# GRIPPER CONFIG
# =========================================================

GRIPPER_ID = 1
GRIP_OPEN = 100
GRIP_CLOSE = 510

GRIP_CLOSE_TIME = 800
GRIP_OPEN_TIME = 400

GRIP_DELAY_BEFORE_CLOSE = 0.6
RETURN_DELAY = 0.4


# =========================================================
# CONNECT
# =========================================================

robot = xarm.Controller("USB")
print("Robot Connected")

# =========================================================
# SAFE FUNCTIONS
# =========================================================

def clamp_safe(sid, val):
    low, high = SERVO_LIMITS.get(sid, (0,1000))
    return int(max(low, min(high, val)))

def move_multiple_raw(targets):
    for sid, pos in targets.items():
        robot.setPosition(sid, pos, duration=MOVE_TIME, wait=False)
    time.sleep(MOVE_TIME/1000)

# =========================================================
# GRIPPER
# =========================================================

def grip_open():
    print("Gripper: OPEN")
    robot.setPosition(GRIPPER_ID, GRIP_OPEN, duration=GRIP_OPEN_TIME, wait=True)
    time.sleep(0.3)

def grip_close_slow(r_norm=None):

    duration = GRIP_CLOSE_TIME

    if r_norm is not None and r_norm < 0.35:
        duration = 1200  # slower close

    robot.setPosition(GRIPPER_ID, GRIP_CLOSE, duration=duration, wait=True)
    time.sleep(0.4)

# =========================================================
# HOME
# =========================================================

def move_home():
    move_multiple_raw({2: HOME[2], 4:HOME[4], 5:HOME[5], 6:HOME[6]})

move_home()
grip_open()

# =========================================================
# COORDINATE TRANSFORM
# =========================================================

def workspace_to_robot(wx, wy):
    return wx - ROBOT_ORIGIN_X, wy - ROBOT_ORIGIN_Y


# =========================================================
#  PRETTY PRINT (NEW)
# =========================================================

def pretty_print_coords(wx, wy, rx, ry):

    # -----------------------------
    # Workspace Camera interpretation
    # -----------------------------
    x_dir = "RIGHT" if wx >= WORK_W/2 else "LEFT"
    y_dir = "NEAR" if wy < WORK_H/2 else "FAR"

    # -----------------------------
    # Robot interpretation
    # -----------------------------
    rx_dir = "LEFT" if rx >= 0 else "RIGHT"

    # -----------------------------
    # Metrics
    # -----------------------------
    dist = np.sqrt(rx**2 + ry**2)
    angle = np.degrees(np.arctan2(-rx, ry))

    # -----------------------------
    # PRINT
    # -----------------------------
    print("\n========== COORDINATES ==========")

    # Exact coordinates
    print(f"Workspace (X,Y): ({wx:.2f}, {wy:.2f})")
    print(f"Robot (X,Y):     ({rx:.2f}, {ry:.2f})")

    # Human-readable
    print(f"Workspace Dir -> X: {x_dir}, Y: {y_dir}")
    print(f"Robot Dir     -> {rx_dir} {abs(rx):.2f}, Forward {ry:.2f}")

    # Motion metrics
    print(f"Distance: {dist:.2f}")
    print(f"Angle: {angle:.2f}°")

    print("=================================\n")

# =========================================================
# ARM CONTROL
# =========================================================

def move_arm(rx, ry, pick_mode=False):

    print("\n------ TARGET ------")
    print("RX:", rx, "RY:", ry)

    if abs(ry) < 0.1:
        ry = 0.1

    # Distance
    r = np.sqrt(rx**2 + ry**2)
    r *= R_SCALE

    r_min = 10
    r_max = 30
    r_norm = (r - r_min) / (r_max - r_min)
    r_norm = np.clip(r_norm, 0, 1)

    # =============================
    # SERVO 6
    # =============================
    angle = np.degrees(np.arctan2(-rx, ry))
    angle = np.clip(angle, -90, 90)

    if angle >= 0:
        base_target = SERVO6_CENTER - (angle / 90.0) * (SERVO6_CENTER - SERVO6_MIN)
    else:
        base_target = SERVO6_CENTER + (abs(angle) / 90.0) * (SERVO6_MAX - SERVO6_CENTER)

    base_target = int(clamp_safe(6, base_target))

    # =============================
    # SERVO 5
    # =============================
    r_corrected = r_norm ** 1.35
    s5_min, s5_max = SERVO_LIMITS[5]
    reach_target = s5_max - r_corrected * (s5_max - s5_min)
    reach_target = int(clamp_safe(5, reach_target))

    # =============================
    # SERVO 4 (TUNED SAFE VERSION)
    # =============================
    angle_norm = abs(angle) / 90.0

    h_base = 1 - r_norm

    near_gate = np.clip((0.4 - r_norm) / 0.4, 0, 1)
    center_boost = (1 - angle_norm) * (1 - r_norm)**2 * near_gate

    angle_weight = (1 - r_norm)
    h_angle = angle_weight * angle_norm * 0.25

    h = h_base + 0.7 * center_boost + h_angle
    h = np.clip(h, 0, 1)

    h = 1 - (1 - h) ** 1.5

    SAFE_MIN = 560
    SAFE_MAX = 920

    elbow_target = SAFE_MIN + h * (SAFE_MAX - SAFE_MIN)
    elbow_target -= 20
    if r_norm > 0.9:
        elbow_target += 10

    elbow_target = int(clamp_safe(4, elbow_target))

    # =========================
    # PICK MODE ADJUSTMENT 

    if pick_mode:
        

        # -------------------------
        # 🔥 ADAPTIVE PICK OFFSET
        # -------------------------
        if r_norm < 0.35:
            # VERY NEAR → minimal downward push
            elbow_target += 5
            reach_target -= 10

        elif r_norm < 0.6:
            # MID RANGE
            elbow_target += 15
            reach_target -= 15

        else:
            # FAR → full reach
            elbow_target += 25
            reach_target -= 20


    # =========================
    # 🔥 GROUND SAFETY (AFTER ADJUSTMENT)
    # =========================
    if r_norm < 0.35:

        SAFE_GROUND = 600 + (0.35 - r_norm) * 120

        # clamp AFTER pick adjustment
        if elbow_target < SAFE_GROUND:
            print("⚠️ Ground safety activated")
            elbow_target = int(SAFE_GROUND)
    # DEBUG
    print("Servo 6:", base_target)
    print("Servo 5:", reach_target)
    print("Servo 4:", elbow_target)

    # MOVE
    robot.setPosition(6, base_target, duration=MOVE_TIME, wait=False)
    robot.setPosition(5, reach_target, duration=MOVE_TIME, wait=False)
    robot.setPosition(4, elbow_target, duration=MOVE_TIME, wait=False)

    time.sleep(MOVE_TIME / 1000)


def place_object(label_id):

    print("Placing:", label_id)

    # extract class name (remove _1, _2)
    cls_name = "_".join(label_id.split("_")[:-1])

    if cls_name not in class_to_box:
        print("Unknown class → skipping placement")
        return

    box_idx = class_to_box[cls_name]

    # offsets
    off = BOX_OFFSETS[box_idx]

    link2_t = reverse_if_out_of_bounds(500, off[0])
    link4_t = reverse_if_out_of_bounds(820, off[1])
    link5_t = reverse_if_out_of_bounds(640, off[2])

    # move base
    robot.setPosition(6, box_positions[box_idx], duration=800, wait=True)

    # move arm down
    robot.setPosition(5, link5_t, duration=800, wait=False)
    robot.setPosition(4, link4_t, duration=800, wait=False)
    time.sleep(0.8)

    robot.setPosition(2, link2_t, duration=600, wait=True)

    # release
    grip_open()
    time.sleep(0.3)

    # return home
    move_home()

    print(f"✅ Placed in box {box_idx}")
    
# =========================================================
# ARUCO + UI
# =========================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
cap = cv2.VideoCapture(CAMERA_INDEX)

workspace_rect = None
workspace_locked = False
clicked_point = None

def mouse(event,x,y,flags,param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x,y)

cv2.namedWindow("Workspace")
cv2.setMouseCallback("Workspace", mouse)

def order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp(img, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    w = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    h = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w,h))


def execute_pick(label_id, cx, cy, w, h, my_task_id):

    global robot_busy, pick_attempts, last_pick_time, stop_pick, task_id, metrics

    # =========================
    # 🔥 TRUE ATTEMPT COUNT
    # =========================
    metrics["total_attempts"] += 1

    robot_busy = True

    wx = (cx / w) * WORK_W
    wy = (cy / h) * WORK_H
    rx, ry = workspace_to_robot(wx, wy)

    print(f"\n🚀 Attempt {pick_attempts+1} → {label_id}")

    # =========================
    # RETRY OPEN
    # =========================
    if pick_attempts > 0:
        grip_open()
        time.sleep(0.3)

    # =========================
    # STOP CHECK BEFORE MOVE
    # =========================
    if my_task_id != task_id:
        robot_busy = False
        return

    # =========================
    # MOVE (IK + PICK MODE HANDLED HERE)
    # =========================
    move_arm(rx, ry, pick_mode=True)

    # =========================
    # STOP AFTER MOVE
    # =========================
    if my_task_id != task_id:
        robot_busy = False
        return

    time.sleep(GRIP_DELAY_BEFORE_CLOSE)

    # =========================
    # 🔥 SINGLE SAFE HEIGHT CONTROL (ONLY ONE Z CONTROL)
    # =========================
    current_elbow = robot.getPosition(4)

    SAFE_MIN_PICK = 620   # tune if needed

    if current_elbow < SAFE_MIN_PICK:
        print("🛡️ Adjusting height before grip")

        robot.setPosition(
            4,
            clamp_safe(4, SAFE_MIN_PICK),
            duration=200,
            wait=True
        )

    # =========================
    # STOP BEFORE CLOSE
    # =========================
    if my_task_id != task_id:
        robot_busy = False
        return

    # =========================
    # CLOSE
    # =========================
    grip_close_slow()

    # =========================
    # STOP AFTER CLOSE
    # =========================
    if my_task_id != task_id:
        grip_open()
        robot_busy = False
        return

    time.sleep(RETURN_DELAY)

    # =========================
    # RETURN HOME
    # =========================
    move_home()

    # =========================
    # 🔋 READ BATTERY (SAFE POINT)
    # =========================
    try:
        metrics["battery_voltage"] = robot.getBatteryVoltage()
        print("🔋 Voltage:", metrics["battery_voltage"])
    except:
        pass
    robot_busy = False
    last_pick_time = time.time()
    
def validate_pick(label_id):

    global verify_start_time

    if label_id is None:
        return False

    # -------------------------
    # 1. Disappearance check (multi-frame)
    # -------------------------
    if label_id not in missing_counter:
        missing_counter[label_id] = 0

    if label_id not in detection_dict:
        missing_counter[label_id] += 1
    else:
        missing_counter[label_id] = 0

    disappeared = missing_counter[label_id] >= MISSING_THRESHOLD

    # -------------------------
    # 2. Trajectory check
    # -------------------------
    try:
        track_id = int(label_id.split("_")[-1])
    except:
        return False

    points = track_history.get(track_id, [])

    if len(points) < 5:
        return False

    # cumulative motion
    traj_len = 0
    for i in range(1, len(points)):
        traj_len += np.linalg.norm(
            np.array(points[i]) - np.array(points[i-1])
        )

    traj_ok = traj_len > 15

    # -------------------------
    # 3. Time validation
    # -------------------------
    if disappeared or traj_ok:

        if verify_start_time is None:
            verify_start_time = time.time()
            return False

        if time.time() - verify_start_time > VERIFY_TIME:
            return True

    elif not disappeared:
        verify_start_time = None
        return False
    return False

def delayed_retry(label_id, w, h):
    global current_target, robot_busy, retry_pending

    print("⏳ Waiting before retry...")
    time.sleep(VERIFY_DELAY)   # or use VERIFY_DELAY
    
    with task_lock:
        local_task = task_id
    # =========================
    # CANCEL IF TARGET CHANGED
    # =========================
    if current_target != label_id:
        retry_pending = False
        return

    # =========================
    # SAFE COPY (THREAD SAFE)
    # =========================
    local_dict = detection_dict.copy()

    # =========================
    # CANCEL IF OBJECT GONE
    # =========================
    if label_id not in local_dict:
        print("🚫 Object gone → cancel retry")
        retry_pending = False
        return

    # =========================
    # CANCEL IF ROBOT BUSY
    # =========================
    if robot_busy:
        retry_pending = False
        return

    # =========================
    # EXECUTE RETRY
    # =========================
    cx, cy, _ = local_dict[label_id]

    print("🔄 Retry executing...")
    
        
    if local_task != task_id:
        retry_pending = False
        return
    Thread(
        target=execute_pick,
        args=(label_id, cx, cy, w, h, task_id),
        daemon=True
    ).start()

    # =========================
    # RESET RETRY FLAG
    # =========================
    retry_pending = False
    
    
def calculate_metrics(metrics):

    now = time.time()
    elapsed = now - metrics["start_time"]

    # =========================
    # BASIC PERFORMANCE
    # =========================
    fps = metrics["frame_count"]/elapsed if elapsed > 0 else 0

    success_rate = metrics["successful_picks"]/metrics["total_pick_cycles"] if metrics["total_pick_cycles"] else 0
    attempts_per_success = (
    metrics["total_attempts"] / metrics["total_pick_cycles"]
    if metrics["total_pick_cycles"] else 0
)
    retry_avg = np.mean(metrics["retry_counts"]) if metrics["retry_counts"] else 0

    avg_pick = np.mean(metrics["pick_durations"]) if metrics["pick_durations"] else 0

    # =========================
    # TRAJECTORY
    # =========================
    traj_valid = metrics["valid_trajectory"]/len(metrics["trajectory_lengths"]) if metrics["trajectory_lengths"] else 0

    avg_traj = np.mean(metrics["trajectory_lengths"]) if metrics["trajectory_lengths"] else 0

    # =========================
    # DETECTION
    # =========================
    stability = metrics["stable_detections"]/metrics["detections_total"] if metrics["detections_total"] else 0

    # =========================
    # ⚡ ENERGY
    # =========================
    CURRENT = 1.5  # realistic average load

    voltage = metrics.get("battery_voltage", 7.5)

    total_energy = 0
    for i, d in enumerate(metrics["pick_durations"]):
        v = metrics["pick_voltages"][i] if i < len(metrics.get("pick_voltages", [])) else voltage
        total_energy += v * CURRENT * d

    energy_per_pick = total_energy / metrics["successful_picks"] if metrics["successful_picks"] else 0

    energy_efficiency = total_energy / metrics["total_pick_cycles"] if metrics["total_pick_cycles"] else 0

    # =========================
    # THROUGHPUT (FIXED)
    # =========================
    # NOTE: use timestamps instead of durations (important fix)
    if "tasks_last_minute" in metrics:
        ppm = len([t for t in metrics["tasks_last_minute"] if t >= now-60])
    else:
        ppm = 0
    t5 = len([t for t in metrics["tasks_last_minute"] if t >= now - 300])
    # =========================
    # 📊 JOURNAL METRICS
    # =========================

    # Detection Stability Ratio
    det_stability_ratio = stability

    # Track Loss Rate
    track_loss_rate = metrics["track_lost"]/metrics["detections_total"] if metrics["detections_total"] else 0

    # Trajectory Mean
    traj_mean = avg_traj

    # Valid Trajectory Ratio
    traj_valid_ratio = traj_valid

    # Disappearance Confirmation Rate
    disappearance_rate = (
        metrics["disappearance_success"] /
        (metrics["successful_picks"] + metrics.get("failed_cycles", 0))
        if (metrics["successful_picks"] + metrics.get("failed_cycles", 0)) else 0
    )

    # Retry Rate
    retry_rate = retry_avg

    # Max Retry Failure Rate
    max_retry_fail_rate = (
        metrics["max_retry_fail"]/metrics["total_pick_cycles"]
        if metrics["total_pick_cycles"] else 0
    )
    class_summary = {}

    for cls, data in metrics["class_stats"].items():

        total = data["success"] + data["fail"]

        success_rate = (data["success"] / total) if total > 0 else 0

        class_summary[cls] = {
            "success": data["success"],
            "fail": data["fail"],
            "rate": success_rate
        }

    # =========================
    # RETURN ALL
    # =========================
    return {
        # existing
        "fps": fps,
        "success": success_rate,
        "retry": retry_avg,
        "avg_pick": avg_pick,
        "traj_valid": traj_valid,
        "avg_traj": avg_traj,
        "stability": stability,
        "energy": energy_per_pick,
        "ppm": ppm,
        "t5": t5,
        "attempts_per_success": attempts_per_success,
        # new journal metrics
        "det_stability_ratio": det_stability_ratio,
        "track_loss_rate": track_loss_rate,
        "traj_mean": traj_mean,
        "traj_valid_ratio": traj_valid_ratio,
        "disappearance_rate": disappearance_rate,
        "retry_rate": retry_rate,
        "max_retry_fail_rate": max_retry_fail_rate,
        "class_summary": class_summary,
        "energy_efficiency": energy_efficiency
    }

def init_class(metrics, cls_name):
    if cls_name not in metrics["class_stats"]:
        metrics["class_stats"][cls_name] = {
            "success": 0,
            "fail": 0
        }   
def draw_graph(img, data, x, y, w=250, h=100, color=(0,255,0), label=""):

    if len(data) < 2:
        return

    data = np.array(data)
    data = data - np.min(data)
    if np.max(data) > 0:
        data = data / np.max(data)

    pts = []

    for i, val in enumerate(data):
        px = int(x + i * (w / len(data)))
        py = int(y + h - val * h)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], color, 2)

    cv2.rectangle(img, (x, y), (x+w, y+h), (200,200,200), 1)

    cv2.putText(img, label, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
def draw_bar_chart(img, data_dict, x, y, w=400, h=200):

    if not data_dict:
        return

    classes = list(data_dict.keys())[:6]
    values = [data_dict[c]["success"] for c in classes]

    max_val = max(values) if max(values) > 0 else 1
    bar_w = int(w / len(classes))

    # 🎨 predefined distinct colors
    colors = [
        (0,255,255),   # yellow
        (255,0,0),     # blue
        (0,255,0),     # green
        (0,0,255),     # red
        (255,0,255),   # purple
        (255,255,0),   # cyan
        (0,128,255),   # orange-ish
        (128,0,255)    # violet
    ]

    for i, (cls, val) in enumerate(zip(classes, values)):

        bar_h = int((val / max_val) * h)

        x1 = x + i * bar_w
        y1 = y + h - bar_h

        color = colors[i % len(colors)]

        # draw bar
        cv2.rectangle(img, (x1, y1), (x1 + bar_w - 5, y + h),
                      color, -1)

        # label
        cv2.putText(img, cls[:6],
                    (x1, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,255,255), 1)

        # value on top (optional but useful)
        cv2.putText(img, str(val),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1)

    # border
    cv2.rectangle(img, (x, y), (x+w, y+h), (200,200,200), 1)

    cv2.putText(img, "Class Success",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,255,255), 1)
    
def draw_event_graph(img, events, x, y, w=400, h=120):

    if len(events) < 2:
        return

    max_len = 100
    events = events[-max_len:]

    step = w / max_len

    for i, val in enumerate(events):

        px = int(x + i * step)

        if val == 1:
            color = (0,255,0)  # success
        elif val == -1:
            color = (0,0,255)  # fail
        else:
            continue

        cv2.line(img, (px, y+h), (px, y+h-40), color, 2)

    cv2.rectangle(img, (x, y), (x+w, y+h), (200,200,200), 1)
    cv2.putText(img, "Pick Events", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
# =========================================================
# MAIN LOOP (UPDATED WITH METRICS)
# =========================================================

print("Show markers 1,2,3,4 then press ENTER")

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()

    if not workspace_locked:

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)
        marker_map = {}

        if ids is not None:
            ids = ids.flatten()

            for i, id in enumerate(ids):
                pts = corners[i][0].astype(int)

                for j in range(4):
                    cv2.line(display, tuple(pts[j]), tuple(pts[(j+1)%4]), (0,255,0), 2)

                for pt in pts:
                    cv2.circle(display, tuple(pt), 6, (0,0,255), -1)
                cx = int(np.mean(pts[:,0]))
                cy = int(np.mean(pts[:,1]))

                cv2.putText(
                    display,
                    f"ID: {id}",
                    (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
                if id == 1: marker_map[id] = tuple(pts[0])
                elif id == 2: marker_map[id] = tuple(pts[1])
                elif id == 3: marker_map[id] = tuple(pts[2])
                elif id == 4: marker_map[id] = tuple(pts[3])

        if all(i in marker_map for i in [1,2,3,4]):

            rect = order_points(np.array([
                marker_map[1], marker_map[2],
                marker_map[3], marker_map[4]
            ], dtype="float32"))

            workspace_rect = rect

            for i in range(4):
                cv2.line(display,
                         tuple(rect[i].astype(int)),
                         tuple(rect[(i+1)%4].astype(int)),
                         (0,255,255), 3)

        cv2.putText(display,"Press ENTER to lock",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    else:

        warped = warp(frame, workspace_rect)
        h,w,_ = warped.shape

        # =========================
        # BYTE TRACK DETECTION
        # =========================

        if robot_busy and last_results is not None:
            results = last_results
        else:
            results = model.track(
                warped,
                conf=TRACK_THRESH,
                persist=True,
                verbose=False
            )[0]
            last_results = results

        detection_dict.clear()
        serial_map = {}

        serial_id = 1

        if results.boxes.id is not None:

            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()

            current_ids = set(ids)

            for box, track_id, cls, score in zip(boxes, ids, classes, scores):

                # =========================
                # METRICS: detection count
                # =========================
                metrics["detections_total"] += 1

                if score < 0.2:
                    continue

                x1, y1, x2, y2 = box

                cx_raw = (x1 + x2) / 2
                cy_raw = (y1 + y2) / 2

                prev = smooth_centers.get(track_id, [cx_raw, cy_raw])

                cx = int(SMOOTH_ALPHA * prev[0] + (1 - SMOOTH_ALPHA) * cx_raw)
                cy = int(SMOOTH_ALPHA * prev[1] + (1 - SMOOTH_ALPHA) * cy_raw)

                smooth_centers[track_id] = [cx, cy]

                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append((cx, cy))

                if len(track_history[track_id]) > MAX_TRAIL:
                    track_history[track_id].pop(0)

                # =========================
                # TRAJECTORY METRICS
                # =========================
                if len(track_history[track_id]) >= 2:
                    p1 = track_history[track_id][0]
                    p2 = track_history[track_id][-1]

                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    metrics["trajectory_lengths"].append(dist)

                    if dist > 8:
                        metrics["valid_trajectory"] += 1

                label = all_classes[cls]
                label_id = f"{label}_{track_id}"

                if label_id not in stable_counter:
                    stable_counter[label_id] = 0

                stable_counter[label_id] += 1

                # =========================
                # STABILITY METRIC
                # =========================
                if stable_counter[label_id] >= STABLE_FRAMES:
                    metrics["stable_detections"] += 1

                detection_dict[label_id] = (cx, cy, score)
                # Reset missing counter when detected
                if label_id in missing_counter:
                    missing_counter[label_id] = 0
                serial_map[serial_id] = label_id

                color = colors(cls, True)

                cv2.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(warped, (cx, cy), 5, color, -1)

                cv2.putText(
                    warped,
                    f"{serial_id}. {label_id} ({score:.2f})",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # trajectory drawing
                points = track_history.get(track_id, [])
                for i in range(1, len(points)):
                    cv2.line(warped, points[i-1], points[i], (0,255,255), 2)

                serial_id += 1

            # =========================
            # TRACK LOST METRIC
            # =========================
            lost_ids = set(smooth_centers.keys()) - current_ids

            metrics["track_lost"] += len(lost_ids)

            # remove permanently lost tracks
            for tid in lost_ids:
                smooth_centers.pop(tid, None)

        # =========================
        # PICK EXECUTION
        # =========================

        if selected_id is not None and not robot_busy and current_target is None:

            if selected_id in serial_map:

                label_id = serial_map[selected_id]

                if label_id in detection_dict:

                    cx, cy, score = detection_dict[label_id]

                    if stable_counter[label_id] >= STABLE_FRAMES and score >= PICK_THRESH:

                        current_target = label_id   # 🔥 LOCK FIRST

                        metrics["total_pick_cycles"] += 1
                        
                        metrics["last_pick_start"] = time.time()

                        with task_lock:
                            task_id += 1
                            current_task = task_id

                        pick_attempts = 0
                        missing_counter.clear()
                        verify_start_time = None

                        Thread(
                            target=execute_pick,
                            args=(label_id, cx, cy, w, h, current_task),
                            daemon=True
                        ).start()

            selected_id = None

        # =========================
        # SUCCESS / RETRY LOGIC
        # =========================

        if (
            current_target is not None
            and not robot_busy
            and not retry_pending
            and (time.time() - last_pick_time > COOLDOWN)
        ):

            # Safety
            if current_target is None:
                continue
            
            success = validate_pick(current_target)

            # =========================
            # SUCCESS
            # =========================
            if success:
                
                print(f"✅ SUCCESS → {current_target}")

                # =========================
                # METRICS: success + duration
                # =========================
                metrics["successful_picks"] += 1
                cls_name = "_".join(current_target.split("_")[:-1])

                init_class(metrics, cls_name)
                metrics["class_stats"][cls_name]["success"] += 1
                metrics["disappearance_success"] += 1
                
                
                metrics["retry_counts"].append(pick_attempts)
                now = time.time()

                metrics["tasks_last_minute"] = [
                    t for t in metrics["tasks_last_minute"]
                    if t >= now - 60
                ]

                metrics["tasks_last_minute"].append(now)

                duration = time.time() - metrics["last_pick_start"]

                try:
                    voltage = robot.getBatteryVoltage()
                except:
                    voltage = metrics.get("battery_voltage", 7.5)

                metrics["pick_durations"].append(duration)
                metrics["event_log"].append(1)
                # store voltage per pick (NEW)
                if "pick_voltages" not in metrics:
                    metrics["pick_voltages"] = []

                metrics["pick_voltages"].append(voltage)

                place_object(current_target)
                time.sleep(0.4)

                
                # -------------------------
                # RESET STATE
                # -------------------------
                with task_lock:
                    task_id += 1

                current_target = None
                pick_attempts = 0
                retry_pending = False
                verify_start_time = None
                missing_counter.clear()
                selected_id = None

                print("🔄 Ready for next selection")

            # =========================
            # RETRY
            # =========================
            else:

                if pick_attempts < MAX_RETRY:

                    pick_attempts += 1
                    retry_pending = True

                    # =========================
                    # METRICS: retry
                    # =========================
                    
                    
                    print(f"❌ Retry {pick_attempts}")

                    Thread(
                        target=delayed_retry,
                        args=(current_target, w, h),
                        daemon=True
                    ).start()

                else:
                    print("🚫 Max retries reached → resetting system")
                    metrics["failed_cycles"] = metrics.get("failed_cycles", 0) + 1
                    cls_name = "_".join(current_target.split("_")[:-1])

                    init_class(metrics, cls_name)
                    metrics["class_stats"][cls_name]["fail"] += 1
                    metrics["retry_counts"].append(pick_attempts)
                    metrics["max_retry_fail"] += 1
                    metrics["event_log"].append(-1)
                    with task_lock:
                        task_id += 1

                    move_home()
                    time.sleep(0.5)
                    grip_open()

                    # -------------------------
                    # HARD RESET
                    # -------------------------
                    current_target = None
                    pick_attempts = 0
                    retry_pending = False
                    verify_start_time = None
                    missing_counter.clear()
                    selected_id = None

                    print("🔄 Ready for next selection")

        # =========================
        # RESIZE + METRICS OVERLAY (BALANCED)
        # =========================

        # =========================
        # RESIZE + CANVAS SETUP
        # =========================

        resized = cv2.resize(warped, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

        h_res, w_res, _ = resized.shape

        # Panel sizes
        left_w = 220
        right_w = 300

        # Create canvas
        canvas = np.zeros((h_res + 200, w_res + left_w + right_w, 3), dtype=np.uint8)

        # Place workspace in CENTER
        canvas[0:h_res, left_w:left_w + w_res] = resized

        # Draw separators
        cv2.line(canvas, (left_w, 0), (left_w, h_res), (80,80,80), 2)
        cv2.line(canvas, (left_w + w_res, 0), (left_w + w_res, h_res), (80,80,80), 2)

        metrics["frame_count"] += 1

        live = calculate_metrics(metrics)

        # =========================
        # 📊 GRAPH DATA UPDATE
        # =========================
        if "fps_hist" not in metrics:
            metrics["fps_hist"] = []
            metrics["success_hist"] = []
            metrics["stability_hist"] = []
            metrics["track_loss_hist"] = []
            metrics["energy_hist"] = []
            metrics["attempts_hist"] = []

        metrics["fps_hist"].append(live["fps"])
        metrics["success_hist"].append(live["success"] * 100)

        metrics["stability_hist"].append(live["stability"] * 100)
        metrics["track_loss_hist"].append(live.get("track_loss_rate", 0) * 100)
        metrics["energy_hist"].append(live["energy"])
        metrics["attempts_hist"].append(live["attempts_per_success"])

        # limit history
        for key in ["fps_hist", "success_hist", "stability_hist",
                    "track_loss_hist", "energy_hist", "attempts_hist"]:
            metrics[key] = metrics[key][-100:]

        # =========================
        # LEFT PANEL (SYSTEM)
        # =========================
        left_x = 10
        y_left = 20

        def draw_left(text, gap=0):
            global y_left
            y_left += gap  # extra spacing before line
            cv2.putText(canvas, text, (left_x, y_left),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            y_left += 20   # normal line spacing

        draw_left("=== SYSTEM ===", gap=5)
        draw_left(f"FPS: {live['fps']:.1f}")
        draw_left(f"PPM: {live['ppm']}")
        draw_left(f"Tasks/5min: {live['t5']}")

        draw_left("=== PICK ===", gap=10)
        draw_left(f"Success: {live['success']*100:.1f}%")
        draw_left(f"Retry Avg: {live['retry']:.2f}")
        draw_left(f"Avg Pick: {live['avg_pick']:.2f}s")

        draw_left(f"Total Attempts: {metrics['total_attempts']}")
        draw_left(f"Pick Cycles: {metrics['total_pick_cycles']}")
        draw_left(f"Success Count: {metrics['successful_picks']}")
        draw_left(f"Max Retry Fail: {metrics['max_retry_fail']}")
        draw_left(f"Attempts/Task: {live['attempts_per_success']:.2f}")

        draw_left("=== VISION ===", gap=10)
        draw_left(f"Stability: {live['stability']*100:.1f}%")
        draw_left(f"Track Loss: {live.get('track_loss_rate',0)*100:.1f}%")

        draw_left("=== TRAJECTORY ===", gap=10)
        draw_left(f"Valid: {live['traj_valid']*100:.1f}%")
        draw_left(f"Avg Dist: {live['avg_traj']:.1f}")

        draw_left("=== ENERGY ===", gap=10)
        draw_left(f"Energy/Pick: {live['energy']:.1f}J")
        draw_left(f"Eff: {live.get('energy_efficiency',0):.1f}J")

        draw_left("=== DETECTION ===", gap=10)
        draw_left(f"Total: {metrics['detections_total']}")
        draw_left(f"Stable: {metrics['stable_detections']}")
        draw_left(f"Lost: {metrics['track_lost']}")
        # =========================
        # CLASS PERFORMANCE
        # =========================
        y_class = y_left + 10

        cv2.putText(canvas, "=== CLASS PERFORMANCE ===",
                    (left_x, y_class),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,0), 1)

        y_class += 18

        sorted_classes = sorted(
            live["class_summary"].items(),
            key=lambda x: (x[1]["success"] + x[1]["fail"]),
            reverse=True
        )

        for cls, data in sorted_classes:
            total = data["success"] + data["fail"]

            text = f"{cls[:18]}: S={data['success']} F={data['fail']} T={total} ({data['rate']*100:.1f}%)"

            cv2.putText(canvas, text,
                        (left_x, y_class),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255,255,0), 1)

            y_class += 16

        # =========================
        # RIGHT PANEL (GRAPHS)
        # =========================
        right_x = left_w + w_res + 20

        # Graphs at TOP
        graph_y = 20

        # =========================
        # 📊 DRAW GRAPHS (SERIAL)
        # =========================

        gap = 140  # same spacing style you used

        draw_graph(canvas, metrics["fps_hist"], right_x, graph_y,
                color=(0,255,0), label="FPS")

        draw_graph(canvas, metrics["success_hist"], right_x, graph_y + gap,
                color=(0,200,255), label="Success %")

        draw_graph(canvas, metrics["stability_hist"], right_x, graph_y + gap*2,
                color=(255,255,0), label="Stability %")

        draw_graph(canvas, metrics["track_loss_hist"], right_x, graph_y + gap*3,
                color=(0,0,255), label="Track Loss %")

        draw_graph(canvas, metrics["energy_hist"], right_x, graph_y + gap*4,
                color=(255,0,255), label="Energy")

        draw_graph(canvas, metrics["attempts_hist"], right_x, graph_y + gap*5,
                color=(0,255,150), label="Attempts/Task")

        
        # =========================
        # 📊 BOTTOM AREA VISUALIZATION
        # =========================

        bottom_y = h_res + 40
        center_x = left_w + 50

        # ---- BAR CHART (LEFT BOTTOM)
        draw_bar_chart(canvas,
                    live["class_summary"],
                    center_x,
                    bottom_y,
                    w=400,
                    h=180)

        # ---- EVENT GRAPH (RIGHT BOTTOM)
        draw_event_graph(canvas,
                        metrics["event_log"],
                        center_x + 450,
                        bottom_y,
                        w=400,
                        h=120)
        # # =========================
        # # RIGHT PANEL (VISION TEXT)
        # # =========================
        # y_right = graph_y + 300

        # def draw_right(text):
        #     global y_right
        #     cv2.putText(canvas, text, (right_x, y_right),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        #     y_right += 18

        # draw_right("=== VISION ===")
        # draw_right(f"Stability: {live['stability']*100:.1f}%")
        # draw_right(f"Track Loss: {live.get('track_loss_rate',0)*100:.1f}%")

        # draw_right("=== TRAJECTORY ===")
        # draw_right(f"Valid: {live['traj_valid']*100:.1f}%")
        # draw_right(f"Avg Dist: {live['avg_traj']:.1f}")

        # draw_right("=== ENERGY ===")
        # draw_right(f"Energy/Pick: {live['energy']:.1f}J")
        # draw_right(f"Eff: {live.get('energy_efficiency',0):.1f}J")

        # draw_right("=== DETECTION ===")
        # draw_right(f"Total: {metrics['detections_total']}")
        # draw_right(f"Stable: {metrics['stable_detections']}")
        # draw_right(f"Lost: {metrics['track_lost']}")

        # =========================
        # SHOW FINAL OUTPUT
        # =========================
        cv2.imshow("Real-Time Dashboard", canvas)

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key >= ord('0') and key <= ord('9'):
        input_buffer += chr(key)
        print("Typing:", input_buffer)

    elif key == 13:
        if input_buffer != "":
            selected_id = int(input_buffer)
            input_buffer = ""

        if workspace_rect is not None:
            workspace_locked = True
            print("Workspace locked")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

move_home()
grip_open()
print("Finished")