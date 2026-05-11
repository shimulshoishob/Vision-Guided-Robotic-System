
# ===========================================================
# VISION GUIDED ROBOTIC SORTING SYSTEM (FULL FIXED VERSION)
# ============================================================

import cv2
import time
import yaml
import random
import numpy as np

from pathlib import Path
from enum import Enum
from threading import Thread, Lock

from ultralytics import YOLO
from supervision import ByteTrack, Detections

import xarm


# ================= CONFIG =================

MODEL_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/best.pt")
YAML_PATH  = Path(r"D:/EWU/10th Semester/CSE475/LABS/data.yaml")

CAMERA_INDEX = 1
CONF_THRESH  = 0.7

WORKSPACE_W = 240
WORKSPACE_H = 150

ROBOT_OFFSET_X = 120
ROBOT_OFFSET_Y = 60

SUPPLY_VOLTAGE = 7.4
CURRENT_DRAW   = 6.0
POWER_W        = SUPPLY_VOLTAGE * CURRENT_DRAW


# ================= STATES =================

class SystemState(Enum):
    STOPPED   = 0
    DETECTING = 1
    ARM_BUSY  = 2


state = SystemState.STOPPED
lock  = Lock()
stop  = False


# ================= ROBOT CONFIG =================

HOME={1:100,2:500,3:250,4:820,5:640,6:500}

NUM_BOXES=9
LEFT_LIMIT,RIGHT_LIMIT=900,100

box_positions=[int(LEFT_LIMIT-i*(LEFT_LIMIT-RIGHT_LIMIT)/(NUM_BOXES-1)) for i in range(NUM_BOXES)]

GRIPPER_OPEN=150
GRIPPER_CLOSE=450

BOX_OFFSETS={
0:(50,30,-60,-250),
1:(40,30,-100,-280),
2:(30,100,-200,-360),
3:(20,100,-200,-360),
4:(30,100,-200,-360),
5:(-20,100,-200,-360),
6:(-30,100,-200,-360),
7:(-40,30,-100,-280),
8:(-50,30,-60,-250)
}


def build_duration_map(general=500):
    return {1:general,2:general,3:general,4:general,5:general,6:general,"default":general}


# ================= LOAD CLASSES =================

cfg=yaml.safe_load(open(YAML_PATH)) if YAML_PATH.exists() else {"names":[]}
all_classes=cfg.get("names",[])

fresh=[c for c in all_classes if "Fresh" in c]
damaged=[c for c in all_classes if "Damaged" in c]

class_to_box={}
for i,c in enumerate(fresh): class_to_box[c]=i
for c in damaged: class_to_box[c]=8

CLASS_COLORS={cls:(random.randint(50,255),random.randint(50,255),random.randint(50,255)) for cls in all_classes}


# ================= CONNECT ROBOT =================

try:
    robot=xarm.Controller("USB")
    print("[Robot] Connected")
except:
    robot=None
    print("[Robot] NOT connected")


# ================= ROBOT HELPERS =================

def clamp(pos):
    return max(0,min(1000,int(pos)))

def move_servo(sid,pos,dur):

    if state == SystemState.STOPPED:
        return

    if robot:
        robot.setPosition(sid,clamp(pos),duration=dur,wait=False)

    time.sleep(dur/1000)


def move_multiple(targets,dmap):

    if state == SystemState.STOPPED:
        return

    max_d=0

    for sid,pos in targets.items():

        d=dmap.get(sid,dmap["default"])

        if robot:
            robot.setPosition(sid,clamp(pos),duration=d,wait=False)

        max_d=max(max_d,d)

    time.sleep(max_d/1000)


def go_home():

    move_multiple({5:HOME[5],4:HOME[4],3:HOME[3]},build_duration_map())
    move_servo(2,HOME[2],500)
    move_servo(1,GRIPPER_OPEN,500)
    move_servo(6,HOME[6],500)



# ================= ARUCO CALIBRATION =================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

H = None


def calibrate_workspace(cap):

    global H

    print("[Calibration] Searching for markers...")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        display = frame.copy()

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

        pixel_points = [None]*4

        if ids is not None:

            ids = ids.flatten()

            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            for i, marker_id in enumerate(ids):

                c = corners[i][0]

                center = c.mean(axis=0).astype(int)

                # draw marker center
                cv2.circle(display, tuple(center), 6, (0,255,0), -1)

                # draw marker label
                cv2.putText(
                    display,
                    f"ID {marker_id}",
                    (center[0]+10, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2
                )

                if marker_id < 4:
                    pixel_points[marker_id] = center

        # SAFE CHECK (FIXED BUG)
        if all(p is not None for p in pixel_points):

            pts = np.array(pixel_points, dtype=np.int32)

            # draw workspace overlay
            overlay = display.copy()

            cv2.fillPoly(overlay, [pts], (255,200,0))

            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

            cv2.polylines(display, [pts], True, (255,255,0), 2)

            # compute homography
            pixel_points_np = np.array(pixel_points, dtype=np.float32)

            world_points = np.array([
                [0,0],
                [WORKSPACE_W,0],
                [WORKSPACE_W,WORKSPACE_H],
                [0,WORKSPACE_H]
            ], dtype=np.float32)

            H, _ = cv2.findHomography(pixel_points_np, world_points)

            # create warped view
            warped = cv2.warpPerspective(frame, H, (600,400))

            cv2.imshow("Warped Workspace", warped)

            cv2.putText(
                display,
                "Calibration ready - press ENTER",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        else:

            cv2.putText(
                display,
                "Place 4 ArUco markers (IDs 0,1,2,3)",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

        cv2.imshow("Calibration View", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 13 and H is not None:  # ENTER
            print("[Calibration] Locked")
            break

        if key == 27:  # ESC
            break



# ================= PIXEL → WORLD =================

def pixel_to_world(px,py):

    p=np.array([[[px,py]]],dtype=np.float32)

    world=cv2.perspectiveTransform(p,H)

    return world[0][0]


# ================= INVERSE KINEMATICS =================

L1=120
L2=120

def IK(x,y):

    r=np.sqrt(x*x+y*y)

    cos2=(r*r-L1*L1-L2*L2)/(2*L1*L2)
    cos2=np.clip(cos2,-1,1)

    t2=np.arccos(cos2)
    t1=np.arctan2(y,x)-np.arctan2(L2*np.sin(t2),L1+L2*np.cos(t2))

    return np.degrees(t1),np.degrees(t2)


def angle_to_servo(a):
    return int(500+a*5.5)


# ================= YOLO + TRACKER =================

model=YOLO(str(MODEL_PATH))
tracker=ByteTrack()


# ================= METRICS =================

metrics={
"total_sorted":0,
"pick_attempts":0,
"successful_picks":0,
"pick_durations":[],
"detections_total":0,
"tasks_last_minute":[],
"energy_log":[]
}


prev_time=time.time()


# ================= TRACKED OBJECTS =================

tracked={}
target_id=None


# ================= VISUAL SERVO =================

def visual_servo(obj):

    cx,cy=obj["center"]

    for _ in range(2):

        wx,wy=pixel_to_world(cx,cy)

        rx=wx+ROBOT_OFFSET_X
        ry=wy+ROBOT_OFFSET_Y

        t1,t2=IK(rx,ry)

        s2=angle_to_servo(t1)
        s3=angle_to_servo(t2)

        move_multiple({2:s2,3:s3},build_duration_map())

        time.sleep(0.2)


# ================= PICK =================

def pick_object(obj):

    x1,y1,x2,y2=obj["bbox"]

    w=x2-x1
    h=y2-y1

    wrist=550 if w>h else 450

    move_servo(5,wrist,400)

    visual_servo(obj)

    move_servo(1,GRIPPER_CLOSE,600)

    go_home()

    return True


# ================= PLACE =================

def place_box(cls):

    idx=class_to_box.get(cls,4)

    off=BOX_OFFSETS[idx]

    move_multiple({
        6:box_positions[idx],
        5:HOME[5]+off[3],
        4:HOME[4]+off[2],
        3:HOME[3]+off[1],
        2:HOME[2]+off[0]
    },build_duration_map())

    move_servo(1,GRIPPER_OPEN,500)

    go_home()


# ================= AUTOMATION THREAD =================

def automation():

    global target_id,state

    while True:

        if state==SystemState.ARM_BUSY and target_id in tracked:

            obj=tracked[target_id]

            metrics["pick_attempts"]+=1

            start=time.time()

            ok=pick_object(obj)

            if ok:
                place_box(obj["class"])

            dur=time.time()-start

            metrics["pick_durations"].append(dur)
            metrics["total_sorted"]+=1
            metrics["successful_picks"]+=1

            now=time.time()

            metrics["tasks_last_minute"].append(now)

            energy=POWER_W*(dur/3600)
            metrics["energy_log"].append((now,energy))

            target_id=None
            state=SystemState.DETECTING

        time.sleep(0.02)


Thread(target=automation,daemon=True).start()


# ================= METRIC CALC =================

def calculate_metrics():

    global prev_time

    now=time.time()

    fps=1/(now-prev_time)
    prev_time=now

    tasks_per_min=len([t for t in metrics["tasks_last_minute"] if t>=now-60])

    energy_min=sum(e for t,e in metrics["energy_log"] if t>=now-60)
    energy_5min=sum(e for t,e in metrics["energy_log"] if t>=now-300)

    success_rate=metrics["successful_picks"]/metrics["pick_attempts"] if metrics["pick_attempts"] else 0

    return fps,tasks_per_min,energy_min,energy_5min,success_rate


# ================= MAIN LOOP =================

cap=cv2.VideoCapture(CAMERA_INDEX)

calibrate_workspace(cap)

print("[System] Press S to start/stop")

while True:

    ret,frame=cap.read()

    if not ret:
        continue

    if state!=SystemState.ARM_BUSY:
        tracked.clear()

    if state!=SystemState.STOPPED:

        results=model(frame,conf=CONF_THRESH)[0]

        boxes=[]
        scores=[]
        cls_ids=[]

        for b in results.boxes.data.tolist():

            x1,y1,x2,y2,sc,ci=b

            if sc<CONF_THRESH:
                continue

            boxes.append([x1,y1,x2,y2])
            scores.append(sc)
            cls_ids.append(int(ci))

        if len(boxes)>0:

            metrics["detections_total"]+=len(boxes)

            det=Detections(
                xyxy=np.array(boxes),
                confidence=np.array(scores),
                class_id=np.array(cls_ids)
            )

            tracks=tracker.update_with_detections(det)

            for xyxy,tid,cid in zip(tracks.xyxy,tracks.tracker_id,tracks.class_id):

                x1,y1,x2,y2=map(int,xyxy)

                cls=all_classes[cid]

                cx=(x1+x2)/2
                cy=(y1+y2)/2

                tracked[tid]={
                "class":cls,
                "center":(cx,cy),
                "bbox":(x1,y1,x2,y2)
                }

                col=CLASS_COLORS.get(cls,(0,255,0))

                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,f"{cls} ID:{tid}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

    items=list(tracked.items())

    for i,(tid,obj) in enumerate(items):

        txt=f"{i+1}:{obj['class']}"

        cv2.putText(frame,txt,(10,200+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    fps,tpm,emin,e5min,success=calculate_metrics()

    cv2.putText(frame,f"FPS:{fps:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.putText(frame,f"Tasks/min:{tpm}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    cv2.putText(frame,f"Success:{success*100:.1f}%",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

    cv2.putText(frame,f"Energy/min:{emin:.3f}Wh",(400,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1)
    cv2.putText(frame,f"Energy/5min:{e5min:.3f}Wh",(400,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1)

    cv2.putText(frame,f"STATE: {state.name}",(10,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)

    cv2.imshow("Detection",frame)

    k=cv2.waitKey(1)&0xFF

    if k==27:
        break

    if k in [ord('s'),ord('S')]:

        with lock:

            if state==SystemState.STOPPED:
                print("[System] START")
                state=SystemState.DETECTING
            else:
                print("[System] STOP")
                state=SystemState.STOPPED
                target_id=None

    if k in [ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:

        idx=int(chr(k))-1

        if idx<len(items):

            target_id=items[idx][0]
            state=SystemState.ARM_BUSY


cap.release()
cv2.destroyAllWindows()

print("[System] Shutdown")
