# ============================================================
# ROBOTIC AUTOMATION WITH CENTER GRID + FULL METRICS DISPLAY
# ============================================================

import cv2, time, yaml, random
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
HOME={1:100,2:500,3:250,4:820,5:640,6:500}
NUM_BOXES=9
LEFT_LIMIT, RIGHT_LIMIT = 900,100
box_positions=[int(LEFT_LIMIT - i*(LEFT_LIMIT-RIGHT_LIMIT)/(NUM_BOXES-1)) for i in range(NUM_BOXES)]
GRIPPER_OPEN, GRIPPER_CLOSE = 150,450
BOX_OFFSETS={0:(50,30,-60,-250),1:(40,30,-100,-280),2:(30,100,-200,-360),3:(20,100,-200,-360),
             4:(30,100,-200,-360),5:(-20,100,-200,-360),6:(-30,100,-200,-360),
             7:(-40,30,-100,-280),8:(-50,30,-60,-250)}
def build_duration_map(general=500, link4=600, link5=700):
    return {1:general,2:general,3:general,4:link4,5:link5,6:general,"default":general}

# LOAD CLASSES
if not YAML_PATH.exists(): cfg={"names":[]}
else: cfg=yaml.safe_load(open(YAML_PATH))
all_classes=cfg.get("names",[])
fresh_classes=[c for c in all_classes if "Fresh" in c]
damaged_classes=[c for c in all_classes if "Damaged" in c]
class_to_box={}
for i,c in enumerate(fresh_classes): class_to_box[c]=i
for c in damaged_classes: class_to_box[c]=8
if len(fresh_classes)>8:
    for c in fresh_classes[8:]: class_to_box[c]=7
random.seed(42)
CLASS_COLORS={cls: tuple(random.randint(50,255) for _ in range(3)) for cls in all_classes}

# CONNECT ROBOT
try: robot=xarm.Controller("USB"); print("[Robot] Connected")
except: robot=None; print("[Robot] NOT connected")

# ROBOT HELPERS
def clamp_target(pos): return max(0,min(1000,int(pos)))
def reverse_if_out_of_bounds(home,offset): t=home+offset; return clamp_target(t if 0<=t<=1000 else home-offset)
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
    move_multiple({5:HOME[5],4:HOME[4],3:HOME[3]},dmap)
    move_servo(2,HOME[2],dmap.get(2,500))
    if open_gripper: move_servo(1,GRIPPER_OPEN,dmap.get(1,500))
    move_servo(6,HOME[6],dmap.get(6,500))

# PICK & PLACE
def pick_and_place_class(dmap,cls_name,post_task_pause=1.5,slow_factor=1.2):
    if cls_name not in class_to_box: return False
    box_idx=class_to_box[cls_name]; pick_idx=4
    slow_d={k:int(v*slow_factor) for k,v in dmap.items()}
    move_multiple({6:box_positions[pick_idx],5:reverse_if_out_of_bounds(HOME[5],-190),
                   4:reverse_if_out_of_bounds(HOME[4],20),3:reverse_if_out_of_bounds(HOME[3],-30),
                   2:reverse_if_out_of_bounds(HOME[2],0)},slow_d)
    move_servo(1,GRIPPER_CLOSE,slow_d.get(1,700)); time.sleep(0.5)
    move_multiple({6:HOME[6],5:HOME[5],4:HOME[4],3:HOME[3],2:HOME[2]},slow_d)
    off=BOX_OFFSETS.get(box_idx,BOX_OFFSETS[4])
    move_multiple({6:box_positions[box_idx],5:reverse_if_out_of_bounds(HOME[5],off[3]),
                   4:reverse_if_out_of_bounds(HOME[4],off[2]),3:reverse_if_out_of_bounds(HOME[3],off[1]),
                   2:reverse_if_out_of_bounds(HOME[2],off[0])},slow_d)
    move_servo(1,GRIPPER_OPEN,slow_d.get(1,700)); time.sleep(0.2)
    move_multiple({6:HOME[6],5:HOME[5],4:HOME[4],3:HOME[3],2:HOME[2]},slow_d)
    move_servo(1,GRIPPER_OPEN,dmap.get(1,500)); time.sleep(post_task_pause)
    return True

# LOAD YOLO
yolo_model=YOLO(str(MODEL_PATH)); print("[YOLO] Model loaded")

# GLOBALS
state=SystemState.STOPPED
cap=cv2.VideoCapture(CAMERA_INDEX)
prev_time=time.time()
target_class=None
automation_lock=Lock()
stop_flag=False
smoothed_boxes={}
SMOOTH_FACTOR=0.8

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

# CENTRAL METRICS FUNCTION
def calculate_metrics(metrics,prev_time):
    now=time.time()
    fps=1/(now-prev_time) if prev_time else 0
    avg_pick=sum(metrics["pick_durations"])/len(metrics["pick_durations"]) if metrics["pick_durations"] else 0
    tasks_per_minute=len([t for t in metrics["tasks_last_minute"] if t>=now-60])
    metrics["tasks_last_5min"]=[(t,d) for t,d in metrics["tasks_last_5min"] if t>=now-300]
    tasks_5min=len(metrics["tasks_last_5min"])
    avg_5min=(sum(d for _,d in metrics["tasks_last_5min"])/tasks_5min) if tasks_5min else 0

    pick_attempts=metrics["pick_attempts"]
    successful_picks=metrics["successful_picks"]
    success_rate=successful_picks/pick_attempts if pick_attempts else 0

    precision=metrics["successful_picks"]/metrics["detections_used"] if metrics["detections_used"] else 0
    recall=metrics["detections_used"]/metrics["detections_total"] if metrics["detections_total"] else 0
    accuracy=metrics["successful_picks"]/metrics["detections_total"] if metrics["detections_total"] else 0

    joint_travel={sid:max([pos[sid] for pos in metrics["joint_positions"]])-min([pos[sid] for pos in metrics["joint_positions"]]) 
                  for sid in range(1,7)} if metrics["joint_positions"] else {sid:0 for sid in range(1,7)}
    joint_velocity_avg={sid:np.mean([v[sid] for v in metrics["joint_velocities"]]) 
                        for sid in range(1,7)} if metrics["joint_velocities"] else {sid:0 for sid in range(1,7)}
    repeatability_error=np.mean([abs(pos[sid]-HOME[sid]) for pos in metrics["joint_positions"] for sid in range(1,7)]) if metrics["joint_positions"] else 0
    battery_voltage=metrics["battery_readings"][-1] if metrics["battery_readings"] else 0
    voltage_drop_per_cycle=metrics["battery_readings"][-2]-metrics["battery_readings"][-1] if len(metrics["battery_readings"])>=2 else 0

    return {
        "fps":fps,
        "avg_pick":avg_pick,
        "tasks_per_minute":tasks_per_minute,
        "tasks_5min":tasks_5min,
        "avg_5min":avg_5min,
        "success_rate":success_rate,
        "precision":precision,
        "recall":recall,
        "accuracy":accuracy,
        "joint_travel":joint_travel,
        "joint_velocity_avg":joint_velocity_avg,
        "repeatability_error":repeatability_error,
        "battery_voltage":battery_voltage,
        "voltage_drop_per_cycle":voltage_drop_per_cycle,
        "current_joint_angles":metrics["joint_positions"][-1] if metrics["joint_positions"] else {sid:0 for sid in range(1,7)}
    }

# AUTOMATION THREAD
def automation_task():
    global state,target_class
    prev_positions=None
    prev_time_kin=time.time()
    while not stop_flag:
        if state==SystemState.ARM_BUSY and target_class:
            with automation_lock:
                metrics["pick_attempts"]+=1
                start=time.time()
                success=pick_and_place_class(build_duration_map(),target_class)
                dur=time.time()-start
                metrics["pick_durations"].append(dur)
                metrics["total_sorted"]+=1
                metrics["sorted_per_class"][target_class]=metrics["sorted_per_class"].get(target_class,0)+1
                metrics["tasks_last_minute"]=[t for t in metrics["tasks_last_minute"] if t>=time.time()-60]
                metrics["tasks_last_minute"].append(time.time())
                metrics["tasks_last_5min"].append((time.time(), dur))
                if success: metrics["successful_picks"]+=1

                if robot:
                    joints={sid:robot.getPosition(sid) for sid in range(1,7)}
                    metrics["joint_positions"].append(joints)
                    if prev_positions is not None:
                        dt=time.time()-prev_time_kin
                        velocities={sid:(joints[sid]-prev_positions[sid])/dt for sid in joints}
                        metrics["joint_velocities"].append(velocities)
                    prev_positions=joints
                    prev_time_kin=time.time()
                    metrics["battery_readings"].append(robot.getBatteryVoltage())
            target_class=None
            if state!=SystemState.STOPPED: state=SystemState.DETECTING
        time.sleep(0.01)
    move_all_home(build_duration_map())

Thread(target=automation_task,daemon=True).start()

# MAIN LOOP
print("[System] Press 'S' to start/stop, 'ESC' to exit")
while True:
    ret,frame=cap.read()
    if not ret: time.sleep(0.05); continue
    annotated_frame=frame.copy()
    detected_class=None
    h,w,_=frame.shape
    center_point=(w//2,h//2)
    cv2.rectangle(annotated_frame,(w//2-50,h//2-50),(w//2+50,h//2+50),(255,255,0),2)

    if state!=SystemState.STOPPED:
        results=yolo_model(frame,conf=CONF_THRESH,imgsz=640)[0]
        best_score=float('inf')
        for box in results.boxes.data.tolist():
            x1,y1,x2,y2,score,cls_idx=box
            cls_idx=int(cls_idx)
            if score<CONF_THRESH or cls_idx>=len(all_classes): continue
            cls_name=all_classes[cls_idx]
            metrics["detections_total"]+=1
            prev_box=smoothed_boxes.get(cls_name,[x1,y1,x2,y2])
            smoothed_box=[SMOOTH_FACTOR*prev_box[i]+(1-SMOOTH_FACTOR)*v for i,v in enumerate([x1,y1,x2,y2])]
            smoothed_boxes[cls_name]=smoothed_box
            x1s,y1s,x2s,y2s=map(int,smoothed_box)
            color=CLASS_COLORS[cls_name]
            cv2.rectangle(annotated_frame,(x1s,y1s),(x2s,y2s),color,2)
            cv2.putText(annotated_frame,cls_name,(x1s,y1s-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
            center_x,center_y=(x1s+x2s)/2,(y1s+y2s)/2
            dist_to_center=((center_x-center_point[0])**2+(center_y-center_point[1])**2)**0.5
            if state==SystemState.DETECTING and dist_to_center<best_score: best_score,detected_class=dist_to_center,cls_name
        if detected_class: state,target_class=SystemState.ARM_BUSY,detected_class
        metrics["detections_used"]+=1 if detected_class else 0

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

    key=cv2.waitKey(1)&0xFF
    if key==27: stop_flag=True; break
    elif key in [ord('s'),ord('S')]:
        with automation_lock:
            if state==SystemState.STOPPED: print("[System] START"); state=SystemState.DETECTING
            else: print("[System] STOP"); target_class=None; state=SystemState.STOPPED

cap.release(); cv2.destroyAllWindows()
move_all_home(build_duration_map())
print("[System] Shutdown complete")
