# ============================================================
# VISION GUIDED ROBOTIC VEGETABLE SORTING SYSTEM
# FULL FINAL VERSION
# ============================================================

import cv2, time, yaml, random
import numpy as np
from pathlib import Path
from enum import Enum
from threading import Thread, Lock

from ultralytics import YOLO
from supervision import ByteTrack, Detections
import xarm


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/best.pt")
YAML_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/data.yaml")

CAMERA_INDEX = 1
CONF_THRESH = 0.7

SUPPLY_VOLTAGE = 7.4
CURRENT_DRAW = 6.0
POWER_W = SUPPLY_VOLTAGE * CURRENT_DRAW

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


# ============================================================
# STATES
# ============================================================

class SystemState(Enum):
    STOPPED=0
    DETECTING=1
    ARM_BUSY=2


state = SystemState.STOPPED


# ============================================================
# ROBOT CONFIG
# ============================================================

HOME={1:100,2:500,3:250,4:820,5:640,6:500}

GRIPPER_OPEN=150
GRIPPER_CLOSE=450

try:
    robot=xarm.Controller("USB")
    print("[Robot] Connected")
except:
    robot=None
    print("[Robot] NOT connected")


def clamp(v):
    return max(0,min(1000,int(v)))


def move_servo(i,pos,d=500):
    if robot:
        robot.setPosition(i,clamp(pos),duration=d,wait=False)
    time.sleep(d/1000)


def go_home():
    for sid,pos in HOME.items():
        move_servo(sid,pos)


# ============================================================
# LOAD CLASSES
# ============================================================

cfg=yaml.safe_load(open(YAML_PATH))
all_classes=cfg["names"]

CLASS_COLORS={cls:(random.randint(50,255),random.randint(50,255),random.randint(50,255)) for cls in all_classes}


# ============================================================
# YOLO + BYTE TRACK
# ============================================================

model=YOLO(str(MODEL_PATH))
tracker=ByteTrack()


# ============================================================
# CALIBRATION VARIABLES
# ============================================================

calibrated=False
workspace_rect=None


# ============================================================
# PERSPECTIVE FUNCTIONS
# ============================================================

def order_points(pts):

    rect=np.zeros((4,2),dtype="float32")

    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return rect


def four_point_transform(image,pts):

    rect=order_points(pts)

    (tl,tr,br,bl)=rect

    widthA=np.linalg.norm(br-bl)
    widthB=np.linalg.norm(tr-tl)
    maxWidth=max(int(widthA),int(widthB))

    heightA=np.linalg.norm(tr-br)
    heightB=np.linalg.norm(tl-bl)
    maxHeight=max(int(heightA),int(heightB))

    dst=np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]],dtype="float32")

    M=cv2.getPerspectiveTransform(rect,dst)

    warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    return warped


# ============================================================
# WORKSPACE DETECTION
# ============================================================

def detect_workspace(frame):

    corners,ids,_=cv2.aruco.detectMarkers(frame,aruco_dict)

    display=frame.copy()

    marker_dict={}

    if ids is not None:

        ids=ids.flatten()

        for i,id in enumerate(ids):

            c=corners[i][0]

            cx=int(np.mean(c[:,0]))
            cy=int(np.mean(c[:,1]))

            marker_dict[id]=(cx,cy)

            cv2.circle(display,(cx,cy),8,(0,255,0),-1)
            cv2.putText(display,str(id),(cx+10,cy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.putText(display,f"{len(marker_dict)} markers found",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    if all(i in marker_dict for i in [1,2,3,4]):

        rect=np.array([
            marker_dict[1],
            marker_dict[2],
            marker_dict[3],
            marker_dict[4]
        ],dtype="float32")

        overlay=display.copy()

        cv2.fillPoly(overlay,[rect.astype(int)],(255,215,0))

        display=cv2.addWeighted(overlay,0.35,display,0.65,0)

        cv2.polylines(display,[rect.astype(int)],True,(255,255,0),2)

        return rect,display

    return None,display


# ============================================================
# METRICS
# ============================================================

metrics={
"total_sorted":0,
"pick_attempts":0,
"successful_picks":0,
"detections_total":0,
"detections_used":0,
"pick_durations":[],
"tasks_last_minute":[],
"energy_log":[]
}

prev_time=time.time()


# ============================================================
# TRACKED OBJECTS
# ============================================================

tracked={}
target_id=None


# ============================================================
# PICK FUNCTION
# ============================================================

def pick_object(obj):

    x1,y1,x2,y2=obj["bbox"]

    w=x2-x1
    h=y2-y1

    wrist=550 if w>h else 450

    move_servo(5,wrist,400)

    move_servo(1,GRIPPER_CLOSE,600)

    go_home()

    return True


# ============================================================
# AUTOMATION THREAD
# ============================================================

def automation():

    global state,target_id

    while True:

        if state==SystemState.ARM_BUSY and target_id in tracked:

            obj=tracked[target_id]

            metrics["pick_attempts"]+=1

            start=time.time()

            ok=pick_object(obj)

            dur=time.time()-start

            metrics["pick_durations"].append(dur)

            energy=POWER_W*(dur/3600)

            metrics["energy_log"].append((time.time(),energy))

            metrics["tasks_last_minute"].append(time.time())

            metrics["total_sorted"]+=1

            if ok:
                metrics["successful_picks"]+=1

            target_id=None

            state=SystemState.DETECTING

        time.sleep(0.02)


Thread(target=automation,daemon=True).start()


# ============================================================
# MAIN LOOP
# ============================================================

cap=cv2.VideoCapture(CAMERA_INDEX)

print("Place ArUco markers and press ENTER")

while True:

    ret,frame=cap.read()

    if not ret:
        continue


    if not calibrated:

        rect,display=detect_workspace(frame)

        cv2.putText(display,"Press ENTER to confirm calibration",(10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Camera",display)

        k=cv2.waitKey(1)&0xFF

        if k in [13,10] and rect is not None:

            workspace_rect=rect
            calibrated=True

            print("Workspace locked")

        continue


    warped=four_point_transform(frame,workspace_rect)

    display=frame.copy()

    if state!=SystemState.STOPPED:

        results=model(warped,conf=CONF_THRESH)[0]

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

            det=Detections(
                xyxy=np.array(boxes),
                confidence=np.array(scores),
                class_id=np.array(cls_ids)
            )

            tracks=tracker.update_with_detections(det)

            tracked.clear()

            for xyxy,tid,cid in zip(tracks.xyxy,tracks.tracker_id,tracks.class_id):

                x1,y1,x2,y2=map(int,xyxy)

                cx=(x1+x2)/2
                cy=(y1+y2)/2

                cls=model.names[cid]

                tracked[tid]={
                    "class":cls,
                    "bbox":(x1,y1,x2,y2),
                    "center":(cx,cy)
                }

                label=f"ID{tid} {cls}"

                cv2.rectangle(warped,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(warped,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)


    cv2.imshow("Camera",display)
    cv2.imshow("Workspace",warped)

    k=cv2.waitKey(1)&0xFF

    if k==27:
        break

    if k in [ord('s'),ord('S')]:

        if state==SystemState.STOPPED:
            state=SystemState.DETECTING
            print("Automation Started")

        else:
            state=SystemState.STOPPED
            print("Automation Stopped")


    if k>=ord('0') and k<=ord('9'):

        selected=int(chr(k))

        if selected in tracked:

            target_id=selected

            state=SystemState.ARM_BUSY

            print("Picking ID:",selected)


cap.release()
cv2.destroyAllWindows()

go_home()

print("System Shutdown")