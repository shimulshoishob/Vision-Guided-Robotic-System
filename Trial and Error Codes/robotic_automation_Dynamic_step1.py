# ============================================================
# VISION GUIDED ROBOTIC VEGETABLE SORTING SYSTEM
# YOLO + BYTE TRACK + ARUCO + VISUAL SERVO + METRICS
# ============================================================

import cv2
import time
import yaml
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
YAML_PATH  = Path(r"D:/EWU/10th Semester/CSE475/LABS/data.yaml")

CAMERA_INDEX = 1
CONF_THRESH  = 0.7

SUPPLY_VOLTAGE = 7.4
CURRENT_DRAW = 6.0
POWER_W = SUPPLY_VOLTAGE * CURRENT_DRAW

WORKSPACE_W = 240
WORKSPACE_H = 150


# ============================================================
# SYSTEM STATE
# ============================================================

class SystemState(Enum):

    STOPPED=0
    DETECTING=1
    ARM_BUSY=2


state = SystemState.STOPPED


# ============================================================
# ROBOT CONFIG
# ============================================================

HOME = {1:100,2:500,3:250,4:820,5:640,6:500}

GRIPPER_OPEN  = 150
GRIPPER_CLOSE = 450


try:

    robot = xarm.Controller("USB")
    print("[Robot] Connected")

except:

    robot=None
    print("[Robot] Not connected")


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
# ARUCO CALIBRATION
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


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
        [0,maxHeight-1]
    ],dtype="float32")

    M=cv2.getPerspectiveTransform(rect,dst)

    warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    return warped


def detect_workspace(frame):

    corners,ids,_ = cv2.aruco.detectMarkers(frame,aruco_dict)

    if ids is None:

        return None,frame

    ids=ids.flatten()

    centers=[]

    for i,id in enumerate(ids):

        c=corners[i][0]

        cx=int(np.mean(c[:,0]))
        cy=int(np.mean(c[:,1]))

        centers.append([cx,cy])

        cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
        cv2.putText(frame,str(id),(cx+10,cy),
        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)


    if len(centers)==4:

        pts=np.array(centers)

        rect=order_points(pts)

        overlay=frame.copy()

        cv2.fillPoly(overlay,[rect.astype(int)],(255,200,0))

        frame=cv2.addWeighted(overlay,0.3,frame,0.7,0)

        cv2.polylines(frame,[rect.astype(int)],True,(255,255,0),2)

        warped=four_point_transform(frame,rect)

        return warped,frame

    return None,frame


# ============================================================
# YOLO + TRACKER
# ============================================================

model=YOLO(str(MODEL_PATH))

tracker=ByteTrack()


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
"detection_latency":[],

"tasks_last_minute":[],

"energy_per_task":[],
"energy_log":[]

}

prev_time=time.time()


def calculate_metrics():

    global prev_time

    now=time.time()

    fps=1/(now-prev_time)

    prev_time=now

    tasks_per_min=len([t for t in metrics["tasks_last_minute"] if t>=now-60])

    avg_pick=np.mean(metrics["pick_durations"]) if metrics["pick_durations"] else 0

    energy_task=np.mean(metrics["energy_per_task"]) if metrics["energy_per_task"] else 0

    energy_min=sum(e for t,e in metrics["energy_log"] if t>=now-60)

    success=metrics["successful_picks"]/metrics["pick_attempts"] if metrics["pick_attempts"] else 0

    precision=metrics["successful_picks"]/metrics["detections_used"] if metrics["detections_used"] else 0

    recall=metrics["detections_used"]/metrics["detections_total"] if metrics["detections_total"] else 0

    accuracy=metrics["successful_picks"]/metrics["detections_total"] if metrics["detections_total"] else 0

    return fps,tasks_per_min,avg_pick,energy_task,energy_min,success,precision,recall,accuracy


# ============================================================
# TRACKED OBJECTS
# ============================================================

tracked={}
target_id=None

lock=Lock()


# ============================================================
# VISUAL SERVO PICK
# ============================================================

def pick_object(obj,cap):

    x1,y1,x2,y2=obj["bbox"]

    cx=(x1+x2)/2
    cy=(y1+y2)/2

    w=x2-x1
    h=y2-y1

    wrist=550 if w>h else 450

    move_servo(5,wrist,400)

    time.sleep(0.5)

    ret,frame=cap.read()

    if ret:

        results=model(frame,conf=CONF_THRESH)[0]

    move_servo(1,GRIPPER_CLOSE,600)

    go_home()

    return True


# ============================================================
# AUTOMATION THREAD
# ============================================================

def automation(cap):

    global state,target_id

    while True:

        if state==SystemState.ARM_BUSY and target_id in tracked:

            obj=tracked[target_id]

            metrics["pick_attempts"]+=1

            start=time.time()

            ok=pick_object(obj,cap)

            dur=time.time()-start

            metrics["pick_durations"].append(dur)

            energy=POWER_W*(dur/3600)

            metrics["energy_per_task"].append(energy)

            metrics["energy_log"].append((time.time(),energy))

            metrics["tasks_last_minute"].append(time.time())

            metrics["total_sorted"]+=1

            if ok:

                metrics["successful_picks"]+=1

            target_id=None

            state=SystemState.DETECTING

        time.sleep(0.02)


# ============================================================
# MAIN PROGRAM
# ============================================================

cap=cv2.VideoCapture(CAMERA_INDEX)

Thread(target=automation,args=(cap,),daemon=True).start()

print("Place 4 ArUco markers to calibrate workspace")


while True:

    ret,frame=cap.read()

    if not ret:

        continue


    warped,display=detect_workspace(frame)


    if warped is not None and state!=SystemState.STOPPED:


        start=time.time()

        results=model(warped,conf=CONF_THRESH)[0]

        metrics["detection_latency"].append(time.time()-start)


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

            tracked.clear()


            for xyxy,tid,cid in zip(tracks.xyxy,tracks.tracker_id,tracks.class_id):

                x1,y1,x2,y2=map(int,xyxy)

                class_name=model.names[cid]

                tracked[tid]={

                    "class_name":class_name,
                    "bbox":(x1,y1,x2,y2)

                }

                label=f"ID{tid} {class_name}"

                cv2.rectangle(warped,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(warped,label,(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)


    fps,tpm,avg_pick,energy_task,energy_min,success,precision,recall,accuracy=calculate_metrics()


    cv2.putText(display,f"FPS:{fps:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.putText(display,f"Tasks/min:{tpm}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

    cv2.putText(display,f"AvgPick:{avg_pick:.2f}s",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),1)

    cv2.putText(display,f"Success:{success*100:.1f}%",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),1)


    cv2.putText(display,f"Energy/task:{energy_task:.3f}Wh",(400,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,200),1)

    cv2.putText(display,f"Energy/min:{energy_min:.3f}Wh",(400,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,200),1)


    cv2.putText(display,f"Precision:{precision*100:.1f}%",(400,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.putText(display,f"Recall:{recall*100:.1f}%",(400,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.putText(display,f"Accuracy:{accuracy*100:.1f}%",(400,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)


    cv2.imshow("Camera View",display)

    if warped is not None:

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

        selected_id=int(chr(k))

        if selected_id in tracked:

            target_id=selected_id

            state=SystemState.ARM_BUSY

            print("Picking ID:",selected_id)


cap.release()

cv2.destroyAllWindows()

go_home()

print("System Shutdown")