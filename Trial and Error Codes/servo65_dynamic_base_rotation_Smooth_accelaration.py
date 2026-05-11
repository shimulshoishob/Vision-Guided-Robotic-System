import cv2
import numpy as np
import time
import xarm

# =========================================================
# CONFIG
# =========================================================

HOME = {5:500, 6:500}
MOVE_TIME = 1200
CAMERA_INDEX = 1

WORK_W = 28
WORK_H = 16

# REAL CALIBRATED ORIGIN
ROBOT_ORIGIN_X = 14
ROBOT_ORIGIN_Y = -8.5

# SERVO LIMITS (safety)
SERVO_LIMITS = {
    5: (200, 500),   # reach
    6: (200, 800)    # base
}

# SERVO CALIBRATION
SERVO_CENTER = 500
SERVO_SCALE = 4.44   # base

# REACH CALIBRATION (YOU MAY TUNE THESE)
MIN_R = 10     # nearest reachable distance
MAX_R = 22     # farthest reachable distance

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

def move_multiple(targets):
    for sid, pos in targets.items():
        pos = clamp_safe(sid, pos)
        robot.setPosition(sid, pos, duration=MOVE_TIME, wait=False)
    time.sleep(MOVE_TIME/1000)

# =========================================================
# HOME
# =========================================================

def move_home():
    move_multiple({5:HOME[5], 6:HOME[6]})

move_home()

# =========================================================
# COORDINATE TRANSFORM
# =========================================================

def workspace_to_robot(wx, wy):
    rx = wx - ROBOT_ORIGIN_X
    ry = wy - ROBOT_ORIGIN_Y
    return rx, ry

# =========================================================
# COMBINED CONTROL (SERVO 6 + 5)
# =========================================================

def move_arm(rx, ry):

    print("RX:", rx, "RY:", ry)

    # -----------------------------
    # SAFETY: avoid division issue
    # -----------------------------
    if abs(ry) < 0.1:
        ry = 0.1

    # -----------------------------
    # REACH LIMIT (CLAMP)
    # -----------------------------
    r = np.sqrt(rx**2 + ry**2)

    if r > MAX_R:
        scale = MAX_R / r
        rx *= scale
        ry *= scale
        print("Clamped to reachable zone")

    # -----------------------------
    # TARGET CALCULATION
    # -----------------------------
    angle = np.degrees(np.arctan(rx / ry))
    base_target = int(SERVO_CENTER + angle * SERVO_SCALE)

    r = np.sqrt(rx**2 + ry**2)
    ratio = (r - MIN_R) / (MAX_R - MIN_R)
    ratio = np.clip(ratio, 0, 1)

    reach_target = int(500 - ratio * 300)

    # -----------------------------
    # APPLY LIMITS
    # -----------------------------
    base_target = clamp_safe(6, base_target)
    reach_target = clamp_safe(5, reach_target)

    print("Angle:", angle)
    print("Distance r:", r)
    print("Servo 6:", base_target)
    print("Servo 5:", reach_target)

    # -----------------------------
    # STABLE MOVEMENT (NO FLOODING)
    # -----------------------------
    MOVE_DURATION = 1200  # ms

    robot.setPosition(6, base_target, duration=MOVE_DURATION, wait=False)
    robot.setPosition(5, reach_target, duration=MOVE_DURATION, wait=False)

    time.sleep(MOVE_DURATION / 1000)

# =========================================================
# ARUCO SETUP
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

    wA = np.linalg.norm(br-bl)
    wB = np.linalg.norm(tr-tl)
    maxW = int(max(wA,wB))

    hA = np.linalg.norm(tr-br)
    hB = np.linalg.norm(tl-bl)
    maxH = int(max(hA,hB))

    dst = np.array([
        [0,0],[maxW-1,0],
        [maxW-1,maxH-1],[0,maxH-1]
    ],dtype="float32")

    M = cv2.getPerspectiveTransform(rect,dst)
    return cv2.warpPerspective(img,M,(maxW,maxH))

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
            for i,id in enumerate(ids):
                c = corners[i][0]
                cx = int(np.mean(c[:,0]))
                cy = int(np.mean(c[:,1]))
                marker_map[id] = (cx,cy)
                cv2.circle(display,(cx,cy),5,(0,255,0),-1)

        if all(i in marker_map for i in [1,2,3,4]):

            rect = np.array([
                marker_map[1],
                marker_map[2],
                marker_map[3],
                marker_map[4]
            ],dtype="float32")

            cv2.polylines(display,[rect.astype(int)],True,(255,255,0),2)
            workspace_rect = rect

        cv2.putText(display,"Press ENTER to lock",
                    (20,40),cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,0,255),2)

    else:

        warped = warp(frame, workspace_rect)
        h,w,_ = warped.shape

        if clicked_point:

            px,py = clicked_point

            wx = (px/w)*WORK_W
            wy = (py/h)*WORK_H

            rx,ry = workspace_to_robot(wx,wy)

            print("WX WY:",wx,wy)
            print("RX RY:",rx,ry)

            move_arm(rx, ry)

            clicked_point = None

        cv2.imshow("Workspace", warped)

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1)

    if key == 13 and workspace_rect is not None:
        workspace_locked = True
        print("Workspace locked")

    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

move_home()
print("Finished")