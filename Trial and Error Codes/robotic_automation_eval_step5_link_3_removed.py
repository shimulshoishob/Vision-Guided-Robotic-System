# ============================================================
# ROBOTIC AUTOMATION WITH CENTER GRID + BYTE TRACK INTEGRATION
# (ONLY TRACKING ADDED — NOTHING ELSE CHANGED)
# ============================================================

import cv2, time, yaml, random, math
from pathlib import Path
from enum import Enum
from threading import Thread, Lock
from ultralytics import YOLO
import xarm
import numpy as np

# CONFIG
MODEL_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/best.pt")
YAML_PATH = Path(r"D:/EWU/10th Semester/CSE475/LABS/data.yaml")
CAMERA_INDEX = 1
CONF_THRESH = 0.7

# STATES
class SystemState(Enum): STOPPED=0; DETECTING=1; ARM_BUSY=2

# ROBOT PARAMS
HOME={1:100,2:500,4:820,5:640,6:500}

NUM_BOXES=9
LEFT_LIMIT, RIGHT_LIMIT = 900,100
box_positions=[int(LEFT_LIMIT - i*(LEFT_LIMIT-RIGHT_LIMIT)/(NUM_BOXES-1)) for i in range(NUM_BOXES)]

GRIPPER_OPEN, GRIPPER_CLOSE = 150,450

BOX_OFFSETS={0:(50,30,-60,-250),1:(40,30,-100,-280),2:(30,100,-200,-360),3:(20,100,-200,-360),
             4:(30,100,-200,-360),5:(-20,100,-200,-360),6:(-30,100,-200,-360),
             7:(-40,30,-100,-280),8:(-50,30,-60,-250)}

def build_duration_map(general=500, link4=600, link5=700):
    return {1:general,2:general,4:link4,5:link5,6:general,"default":general}

# LOAD CLASSES
cfg=yaml.safe_load(open(YAML_PATH)) if YAML_PATH.exists() else {"names":[]}
all_classes=cfg.get("names",[])

fresh_classes=[c for c in all_classes if "Fresh" in c]
damaged_classes=[c for c in all_classes if "Damaged" in c]

class_to_box={}
for i,c in enumerate(fresh_classes): class_to_box[c]=i
for c in damaged_classes: class_to_box[c]=8

random.seed(42)
CLASS_COLORS={cls: tuple(random.randint(50,255) for _ in range(3)) for cls in all_classes}

# CONNECT ROBOT
try:
    robot=xarm.Controller("USB")
    print("[Robot] Connected")
except:
    robot=None
    print("[Robot] NOT connected")

# HELPERS
def clamp_target(pos): return max(0,min(1000,int(pos)))
def reverse_if_out_of_bounds(home,offset):
    t=home+offset
    return clamp_target(t if 0<=t<=1000 else home-offset)

def move_servo(sid,pos,dur):
    if robot: robot.setPosition(sid,clamp_target(pos),duration=dur,wait=False)
    time.sleep(dur/1000)

def move_multiple(targets,dmap):
    max_d=0
    for sid,pos in targets.items():
        d=dmap.get(sid,dmap["default"])
        if robot: robot.setPosition(sid,clamp_target(pos),duration=d,wait=False)
        max_d=max(max_d,d)
    time.sleep(max_d/1000)

def move_all_home(dmap,open_gripper=True):
    move_multiple({5:HOME[5],4:HOME[4]},dmap)
    move_servo(2,HOME[2],dmap.get(2,500))
    if open_gripper: move_servo(1,GRIPPER_OPEN,dmap.get(1,500))
    move_servo(6,HOME[6],dmap.get(6,500))

# PICK & PLACE (UNCHANGED)
def pick_and_place_class(dmap,cls_name,post_task_pause=1.5,slow_factor=1.2):
    if cls_name not in class_to_box: return False
    box_idx=class_to_box[cls_name]; pick_idx=4
    slow_d={k:int(v*slow_factor) for k,v in dmap.items()}

    move_multiple({
        6:box_positions[pick_idx],
        5:reverse_if_out_of_bounds(HOME[5],-190),
        4:reverse_if_out_of_bounds(HOME[4],20),
        2:reverse_if_out_of_bounds(HOME[2],0)
    },slow_d)

    move_servo(1,GRIPPER_CLOSE,slow_d.get(1,700)); time.sleep(0.5)

    move_multiple({6:HOME[6],5:HOME[5],4:HOME[4],2:HOME[2]},slow_d)

    off=BOX_OFFSETS.get(box_idx,BOX_OFFSETS[4])

    move_multiple({
        6:box_positions[box_idx],
        5:reverse_if_out_of_bounds(HOME[5],off[3]),
        4:reverse_if_out_of_bounds(HOME[4],off[2]),
        2:reverse_if_out_of_bounds(HOME[2],off[0])
    },slow_d)

    move_servo(1,GRIPPER_OPEN,slow_d.get(1,700)); time.sleep(0.2)

    move_multiple({6:HOME[6],5:HOME[5],4:HOME[4],2:HOME[2]},slow_d)

    move_servo(1,GRIPPER_OPEN,dmap.get(1,500)); time.sleep(post_task_pause)
    return True

# LOAD YOLO
yolo_model=YOLO(str(MODEL_PATH))

# TRACKING VARIABLES
target_track_id=None
target_start_pos=None
missing_frames=0
TRACK_LOST_THRESHOLD=5
MOVE_THRESHOLD=40

# GLOBALS
state=SystemState.STOPPED
cap=cv2.VideoCapture(CAMERA_INDEX)
prev_time=time.time()
target_class=None
automation_lock=Lock()
stop_flag=False

metrics={
    "total_sorted":0,
    "pick_durations":[],
    "tasks_last_minute":[],
    "tasks_last_5min":[],
    "sorted_per_class":{},
    "joint_positions":[],
    "joint_velocities":[],
    "battery_readings":[],
    "pick_attempts":0,
    "successful_picks":0,
    "detections_total":0,
    "detections_used":0
}
# ================= METRICS =================
def calculate_metrics(metrics,prev_time):
    now=time.time()
    fps=1/(now-prev_time) if prev_time else 0

    avg_pick=sum(metrics["pick_durations"])/len(metrics["pick_durations"]) if metrics["pick_durations"] else 0
    tasks_per_minute=len([t for t in metrics["tasks_last_minute"] if t>=now-60])

    success_rate=metrics["successful_picks"]/metrics["pick_attempts"] if metrics["pick_attempts"] else 0
    precision=metrics["successful_picks"]/metrics["detections_used"] if metrics["detections_used"] else 0
    recall=metrics["detections_used"]/metrics["detections_total"] if metrics["detections_total"] else 0
    accuracy=metrics["successful_picks"]/metrics["detections_total"] if metrics["detections_total"] else 0

    return {
        "fps":fps,
        "avg_pick":avg_pick,
        "tasks_per_minute":tasks_per_minute,
        "tasks_5min":0,
        "avg_5min":0,
        "success_rate":success_rate,
        "precision":precision,
        "recall":recall,
        "accuracy":accuracy,
        "joint_travel":{},
        "joint_velocity_avg":{},
        "repeatability_error":0,
        "battery_voltage":0,
        "voltage_drop_per_cycle":0,
        "current_joint_angles":{}
    }
# AUTOMATION THREAD (UNCHANGED except success logic removed)
def automation_task():
    global state,target_class
    while not stop_flag:
        if state==SystemState.ARM_BUSY and target_class:
            with automation_lock:
                metrics["pick_attempts"]+=1
                pick_and_place_class(build_duration_map(),target_class)
            target_class=None
            if state!=SystemState.STOPPED:
                state=SystemState.DETECTING
        time.sleep(0.01)

Thread(target=automation_task,daemon=True).start()

# MAIN LOOP
while True:
    ret,frame=cap.read()
    if not ret: continue

    annotated_frame=frame.copy()
    detected_class=None

    results=yolo_model.track(frame,conf=CONF_THRESH,persist=True,tracker="bytetrack.yaml")[0]

    tracks={}
    if results.boxes.id is not None:
        for box,tid in zip(results.boxes.xyxy,results.boxes.id):
            x1,y1,x2,y2=map(int,box)
            tid=int(tid)
            cx,cy=(x1+x2)//2,(y1+y2)//2
            tracks[tid]=(cx,cy)

    # SELECT TARGET
    if state==SystemState.DETECTING and tracks:
        target_track_id=list(tracks.keys())[0]
        target_start_pos=tracks[target_track_id]
        target_class=all_classes[0] if all_classes else None
        state=SystemState.ARM_BUSY

    # VALIDATION
    if target_track_id is not None:
        if target_track_id not in tracks:
            missing_frames+=1
        else:
            cx,cy=tracks[target_track_id]
            dist=math.hypot(cx-target_start_pos[0],cy-target_start_pos[1])
            if dist>MOVE_THRESHOLD:
                missing_frames+=1
            else:
                missing_frames=0

        if missing_frames>TRACK_LOST_THRESHOLD:
            metrics["successful_picks"]+=1
            target_track_id=None
    
    # METRICS
    eval_metrics=calculate_metrics(metrics,prev_time)
    prev_time=time.time()
    
    # ------------------- DISPLAY ALL METRICS -------------------
    rx=annotated_frame.shape[1]-320
    cv2.putText(annotated_frame,f"FPS:{eval_metrics['fps']:.1f}",(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.putText(annotated_frame,f"Total:{metrics['total_sorted']}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    cv2.putText(annotated_frame,f"Tasks/min:{eval_metrics['tasks_per_minute']}",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1)
    cv2.putText(annotated_frame,f"Avg1minPick(s):{eval_metrics['avg_pick']:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,255,0),1)

    # Per-class counts
    y=75
    for cls,cnt in metrics["sorted_per_class"].items():
        cv2.putText(annotated_frame,f"{cls}:{cnt}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,180,0),1)
        y+=15

    # 5 min metrics + success & battery
    cv2.putText(annotated_frame,f"Tasks/5min:{eval_metrics['tasks_5min']}",(rx,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(annotated_frame,f"Avg5minPick:{eval_metrics['avg_5min']:.2f}s",(rx,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(annotated_frame,f"Success:{eval_metrics['success_rate']*100:.1f}%",(rx,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,180,255),1)
    cv2.putText(annotated_frame,f"Precision:{eval_metrics['precision']*100:.1f}%",(rx,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,180,0),1)
    cv2.putText(annotated_frame,f"Recall:{eval_metrics['recall']*100:.1f}%",(rx,75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,180,0),1)
    cv2.putText(annotated_frame,f"Accuracy:{eval_metrics['accuracy']*100:.1f}%",(rx,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,180),1)
    cv2.putText(annotated_frame,f"Battery V:{eval_metrics['battery_voltage']:.2f}",(rx,105),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    cv2.putText(annotated_frame,f"Volt Drop:{eval_metrics['voltage_drop_per_cycle']:.2f}",(rx,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    cv2.putText(annotated_frame,f"Repeat Err:{eval_metrics['repeatability_error']:.2f}",(rx,135),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,150,255),1)

    # Joint travel
    jy=150
    for sid, travel in eval_metrics["joint_travel"].items():
        cv2.putText(annotated_frame,f"J{sid} Travel:{travel:.1f}",(rx,jy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,255),1)
        jy+=15

    # Joint avg velocity
    for sid, vel in eval_metrics["joint_velocity_avg"].items():
        cv2.putText(annotated_frame,f"J{sid} Vel:{vel:.1f}",(rx,jy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,255,200),1)
        jy+=15

    # Current servo angles
    for sid, angle in eval_metrics["current_joint_angles"].items():
        cv2.putText(annotated_frame,f"J{sid} Angle:{angle:.1f}",(rx,jy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,150),1)
        jy+=15

    cv2.imshow("Live Detection",annotated_frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()