import cv2
import numpy as np
import time
import xarm

# =========================================================
# CONFIG
# =========================================================

HOME = {4:820, 5:640, 6:500}
MOVE_TIME = 1200
CAMERA_INDEX = 1

WORK_W = 28
WORK_H = 14

ROBOT_ORIGIN_X = 14
ROBOT_ORIGIN_Y = -12

SERVO_LIMITS = {
    4: (400, 900),
    5: (290, 520),
    6: (200, 700)
}

SERVO6_MIN = 100
SERVO6_MAX = 850
SERVO6_CENTER = 500

MIN_R = 10
MAX_R = 30

R_SCALE = 1.05

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
# HOME
# =========================================================

def move_home():
    move_multiple_raw({4:HOME[4], 5:HOME[5], 6:HOME[6]})

move_home()

# =========================================================
# COORDINATE TRANSFORM
# =========================================================

def workspace_to_robot(wx, wy):
    return wx - ROBOT_ORIGIN_X, wy - ROBOT_ORIGIN_Y

# =========================================================
# ARM CONTROL
# =========================================================

def move_arm(rx, ry):

    print("\n------ TARGET ------")
    print("RX:", rx, "RY:", ry)

    if abs(ry) < 0.1:
        ry = 0.1

    # -----------------------------
    # DISTANCE
    # -----------------------------
    r = np.sqrt(rx**2 + ry**2)
    r *= R_SCALE
    r = np.clip(r, MIN_R, MAX_R)

    print("Distance r:", r)

    # =====================================================
    # SERVO 6 (UNCHANGED - PERFECT)
    # =====================================================
    angle = np.degrees(np.arctan2(-rx, ry))
    angle = np.clip(angle, -90, 90)

    if angle >= 0:
        base_target = SERVO6_CENTER - (angle / 90.0) * (SERVO6_CENTER - SERVO6_MIN)
    else:
        base_target = SERVO6_CENTER + (abs(angle) / 90.0) * (SERVO6_MAX - SERVO6_CENTER)

    base_target = int(clamp_safe(6, base_target))

    # =====================================================
    # 🔥 SERVO 5 (FINAL STABLE FIX)
    # =====================================================

    ratio = (r - MIN_R) / (MAX_R - MIN_R)
    ratio = np.clip(ratio, 0, 1)

    angle_norm = abs(angle) / 90.0

    # 🔥 diagonal compensation (stable)
    diag_boost = angle_norm * (ratio ** 2)

    final_ratio = ratio + 0.6 * diag_boost
    final_ratio = np.clip(final_ratio, 0, 1)

    MIN_REACH = 290
    MAX_REACH = 520

    reach_target = MAX_REACH - final_ratio * (MAX_REACH - MIN_REACH)
    reach_target = int(clamp_safe(5, reach_target))

    # =====================================================
    # SERVO 4 (UNCHANGED)
    # =====================================================
    bend_ratio = (1 - ratio) ** 0.5

    MIN_BEND = 600
    MAX_BEND = 900

    elbow_target = int(MIN_BEND + bend_ratio * (MAX_BEND - MIN_BEND))
    elbow_target = clamp_safe(4, elbow_target)

    # -----------------------------
    # DEBUG
    # -----------------------------
    print("Angle:", angle)
    print("Servo 6:", base_target)
    print("Servo 5:", reach_target)
    print("Servo 4:", elbow_target)

    # -----------------------------
    # MOVE
    # -----------------------------
    robot.setPosition(6, base_target, duration=MOVE_TIME, wait=False)
    robot.setPosition(5, reach_target, duration=MOVE_TIME, wait=False)
    robot.setPosition(4, elbow_target, duration=MOVE_TIME, wait=False)

    time.sleep(MOVE_TIME / 1000)

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

# =========================================================
# MAIN LOOP
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

        if clicked_point:

            px,py = clicked_point

            wx = (px/w)*WORK_W
            wy = (py/h)*WORK_H

            rx,ry = workspace_to_robot(wx,wy)

            print("WX WY:", wx, wy)
            print("RX RY:", rx, ry)

            move_arm(rx, ry)
            clicked_point = None

        cv2.imshow("Workspace", warped)

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1)

    if key == 13 and workspace_rect is not None:
        workspace_locked = True
        print("Workspace locked")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

move_home()
print("Finished")