# 🥬 Vision-Guided Robotic System for Automated Vegetable Sorting

<p align="center">

🎓 Department of Computer Science & Engineering  
**East West University, Dhaka, Bangladesh**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLO-v12-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=flat-square&logo=opencv)
![ByteTrack](https://img.shields.io/badge/ByteTrack-Tracking-success?style=flat-square)
![ArUCo](https://img.shields.io/badge/ArUCo-Calibration-blue?style=flat-square)

</p>

## 📌 Overview

This project presents a **vision-guided robotic system** for **automated vegetable sorting** by integrating **Computer Vision**, **Deep Learning**, and **Robotic Manipulation**.

Unlike conventional pick-and-place systems that rely on predefined object positions, this framework performs **dynamic object picking** within a calibrated workspace. The system detects vegetables using YOLO, estimates object locations from bounding box centers, transforms image coordinates into robot workspace coordinates using **ArUCo marker calibration**, computes robotic arm movement through **Inverse Kinematics (IK)**, and validates successful picks using **ByteTrack-based trajectory tracking** without requiring additional grasping sensors.

The repository also includes a comparative study of baseline YOLO models and Self-Supervised Learning (SSL) approaches (BYOL, DINO, and MAE) for robust feature learning and object detection.

---

## 🎯 Key Features

- Real-time vegetable detection using YOLO
- Dynamic pick-and-place within a calibrated workspace
- ArUCo marker-based workspace calibration
- Image-to-world coordinate transformation
- Inverse Kinematics for robotic arm control
- ByteTrack-based vision-only pick validation
- Baseline YOLO and SSL (BYOL, DINO, MAE) comparison
- Complete research implementation with deployment code

---

## 🏗️ System Pipeline

```text
RGB Camera
      │
      ▼
YOLO Object Detection
      │
      ▼
Bounding Box Center Extraction
      │
      ▼
ArUCo Workspace Calibration
      │
      ▼
Image → Robot Coordinate Transformation
      │
      ▼
Inverse Kinematics
      │
      ▼
Dynamic Pick & Place
      │
      ▼
ByteTrack Trajectory Validation
      │
      ▼
Category-wise Vegetable Sorting
```

---

---

## 🎥 System Demonstration

A complete demonstration of the vision-guided robotic sorting system can be viewed here:

🎬 **Project Demonstration**

https://youtu.be/YOUR_VIDEO_LINK

The demonstration showcases:

- Real-time vegetable detection
- Dynamic workspace calibration using ArUCo markers
- Image-to-world coordinate transformation
- Inverse kinematics-based robotic manipulation
- Dynamic pick-and-place
- ByteTrack-based pick validation
- Automated vegetable sorting



## 💡 Dynamic Picking Strategy

Unlike conventional robotic sorting systems that rely on predefined pick locations, this framework performs **dynamic object picking** within a calibrated workspace.

The complete manipulation process consists of:

1. Detect vegetables using a trained YOLO model.
2. Extract the center point of the detected bounding box.
3. Calibrate the workspace using **ArUCo markers**.
4. Transform image coordinates into robot workspace coordinates.
5. Compute robotic joint angles using **Inverse Kinematics (IK)**.
6. Execute the pick-and-place operation.
7. Verify successful picking using **ByteTrack-based trajectory tracking**.

This enables the robotic arm to pick vegetables from **any position inside the workspace** rather than fixed predefined locations.

---

## 🔍 Sensorless Pick Validation

The robotic arm does not include force, tactile, or proximity sensors to confirm successful grasping.

To address this limitation, the system employs **ByteTrack** as a vision-based validation mechanism. After the robotic arm initiates a grasp, ByteTrack continuously tracks the detected vegetable across consecutive frames. If the object's trajectory follows the expected robotic motion, the pick is considered successful.

This approach provides:

- Vision-only grasp verification
- Reduced hardware complexity
- Low-cost deployment
- Robust trajectory-based pick validation

---

## 📂 Repository Structure

```text
Vision-Guided-Robotic-System
│
├── 📁 Final Code
│   ├── Robotic_Arm_Automation_Code_Final_CAPSTONE_C.py
│   ├── best.pt
│   └── data.yaml
│
├── 📁 Baseline Models
│   ├── YOLOv10
│   ├── YOLOv11
│   └── YOLOv12
│
├── 📁 SSL Architectural Models
│   ├── BYOL
│   ├── DINO
│   └── MAE
│
├── 📁 Trial and Error Codes
│
├── 📄 Research Paper
├── 📊 Poster
└── 📽 Presentation
```

---

## 🧠 Deep Learning Models

This project evaluates both baseline object detection models and Self-Supervised Learning (SSL) enhanced architectures.

| Category | Models |
|----------|--------|
| Baseline Detection | YOLOv10, YOLOv11, YOLOv12 |
| Self-Supervised Learning | BYOL, DINO, MAE |

The SSL models are used for feature learning before fine-tuning the detection model, enabling a comparative analysis of representation learning for robotic vision.

---

## 🥕 Dataset

The dataset consists of annotated images of eight vegetable categories collected from multiple publicly available sources and curated for robotic object detection. All images were manually verified, cleaned, and annotated in **YOLO format** using **Roboflow**.

### Dataset Statistics

| Attribute | Details |
|-----------|---------|
| Total Classes | 8 |
| Annotation Format | YOLO |
| Image Resolution | 640 × 640 |
| Annotation Tool | Roboflow |

### Vegetable Classes

- 🍅 Tomato
- 🥔 Potato
- 🧅 Onion
- 🍋 Lemon
- 🍆 Eggplant
- 🥒 Cucumber
- 🥒 Bitter Gourd
- 🥒 Pointed Gourd

### Dataset Source

The complete annotated dataset is publicly available on Roboflow:

**🔗 Roboflow Dataset:**  
https://universe.roboflow.com/sanjana-kazi-supti-ymhu2/non-seasonal-vegetable-detection-yms0u

Dataset configuration and annotations are provided through the `data.yaml` file.

---

## ⚙️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Deep Learning | PyTorch, Ultralytics YOLO |
| Computer Vision | OpenCV, ArUCo |
| Object Tracking | ByteTrack |
| Robotics | Hiwonder xArm, Inverse Kinematics |
| Dataset | Roboflow |
| Development | Jupyter Notebook, Google Colab |

---

## 👨‍💻 Research Team

This project was developed by the **Vision Intelligence and Robotics Research Team**, Department of Computer Science & Engineering, **East West University**, Dhaka, Bangladesh.

### Team Members

- **Fathhur Rahaman Sams**
- **Sanjana Kazi Supti**
- **Md. Junaeid Ali**
- **Mahfuj Alam Imon**

This repository is an official academic research project conducted under the supervision and authorization of the **Department of Computer Science & Engineering, East West University**.

For academic collaboration or research inquiries, please contact the respective project authors.
