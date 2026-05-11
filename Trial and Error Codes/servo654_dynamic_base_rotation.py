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
WORK_H = 16

ROBOT_ORIGIN_X = 14
ROBOT_ORIGIN_Y = -8.5

SERVO_LIMITS = {
    4: (400, 900),   # elbow
    5: (300, 480),   # reach
    6: (200, 700)    # base
}

SERVO_CENTER = 520
SERVO_SCALE = 4.44

MIN_R = 10
MAX_R = 26

X_WEIGHT = 0.65

# =========================================================
# CONNECT
# =========================================================

robot = xarm.Controller("USB")
print("Robot Connected")

# =========================================================
# SAFE FUNCTIONS (USED ONLY FOR MOTION)
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
# RAW MOVE (NO LIMITS → FOR HOME ONLY)
# =========================================================

def move_multiple_raw(targets):
    for sid, pos in targets.items():
        robot.setPosition(sid, pos, duration=MOVE_TIME, wait=False)
    time.sleep(MOVE_TIME/1000)

# =========================================================
# HOME (FULLY INDEPENDENT)
# =========================================================

def move_home():
    move_multiple_raw({4:HOME[4], 5:HOME[5], 6:HOME[6]})

move_home()

# =========================================================
# COORDINATE TRANSFORM
# =========================================================

def workspace_to_robot(wx, wy):
    rx = wx - ROBOT_ORIGIN_X
    ry = wy - ROBOT_ORIGIN_Y
    return rx, ry

# =========================================================
# MAIN CONTROL (UNCHANGED LOGIC)
# =========================================================

def move_arm(rx, ry):

    print("\n------ TARGET ------")
    print("RX:", rx, "RY:", ry)

    if abs(ry) < 0.1:
        ry = 0.1

    # -----------------------------
    # DISTANCE (corrected)
    # -----------------------------
    r = np.sqrt((rx * X_WEIGHT)**2 + ry**2)

    if r > MAX_R:
        scale = MAX_R / r
        rx *= scale
        ry *= scale
        r = MAX_R
        print("Clamped")

    print("Distance r:", r)

    # -----------------------------
    # SERVO 6 (ANGLE)
    # -----------------------------
    angle = np.degrees(np.arctan2(rx, ry))
    base_target = int(SERVO_CENTER + angle * SERVO_SCALE)

    # -----------------------------
    # SERVO 5 (REACH)
    # -----------------------------
    ratio = (r - MIN_R) / (MAX_R - MIN_R)
    ratio = np.clip(ratio, 0, 1)

    MIN_REACH = 300
    MAX_REACH = 480

    reach_target = int(MAX_REACH - ratio * (MAX_REACH - MIN_REACH))

    if reach_target <= MIN_REACH + 5:
        reach_target = MIN_REACH + 5

    # -----------------------------
    # SERVO 4 (ELBOW)
    # -----------------------------
    bend_ratio = (1 - ratio) ** 0.5

    MIN_BEND = 600
    MAX_BEND = 900

    elbow_target = int(MIN_BEND + bend_ratio * (MAX_BEND - MIN_BEND))

    # -----------------------------
    # CLAMP (ONLY FOR MOTION)
    # -----------------------------
    base_target = clamp_safe(6, base_target)
    reach_target = clamp_safe(5, reach_target)
    elbow_target = clamp_safe(4, elbow_target)

    # -----------------------------
    # DEBUG
    # -----------------------------
    print("Angle:", angle)
    print("Servo 6:", base_target)
    print("Servo 5:", reach_target)
    print("Servo 4:", elbow_target)

    # -----------------------------
    # SYNC MOVEMENT (STABLE)
    # -----------------------------
    robot.setPosition(6, base_target, duration=MOVE_TIME, wait=False)
    robot.setPosition(5, reach_target, duration=MOVE_TIME, wait=False)
    robot.setPosition(4, elbow_target, duration=MOVE_TIME, wait=False)

    time.sleep(MOVE_TIME / 1000)

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

            for i, id in enumerate(ids):

                c = corners[i][0]
                pts = c.astype(int)

                # -----------------------------
                # DRAW MARKER BORDER
                # -----------------------------
                for j in range(4):
                    pt1 = tuple(pts[j])
                    pt2 = tuple(pts[(j+1)%4])
                    cv2.line(display, pt1, pt2, (0,255,0), 2)

                # -----------------------------
                # DRAW CORNER POINTS + LABELS
                # -----------------------------
                corner_names = ["TL","TR","BR","BL"]

                for j, pt in enumerate(pts):
                    cv2.circle(display, tuple(pt), 6, (0,0,255), -1)

                    cv2.putText(display, corner_names[j],
                                (pt[0]+5, pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,0), 1)

                # -----------------------------
                # USE TRUE CORNERS FOR WORKSPACE
                # -----------------------------
                if id == 1:
                    marker_map[id] = tuple(pts[0])  # TL
                elif id == 2:
                    marker_map[id] = tuple(pts[1])  # TR
                elif id == 3:
                    marker_map[id] = tuple(pts[2])  # BR
                elif id == 4:
                    marker_map[id] = tuple(pts[3])  # BL

                # -----------------------------
                # DRAW ID AT CENTER (ONLY VISUAL)
                # -----------------------------
                cx = int(np.mean(c[:,0]))
                cy = int(np.mean(c[:,1]))

                cv2.putText(display, f"ID:{id}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,255), 2)

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