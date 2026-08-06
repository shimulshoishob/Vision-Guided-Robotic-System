# 🥬 Vision-Guided Robotic System for Automated Vegetable Sorting

<p align="center">

🎓 **Department of Computer Science & Engineering**  
**East West University, Dhaka, Bangladesh**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLO-v12-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=flat-square&logo=opencv)
![ByteTrack](https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-success?style=flat-square)
![ArUCo](https://img.shields.io/badge/ArUCo-Workspace%20Calibration-blue?style=flat-square)

</p>

---

## 📌 Overview

This repository presents a **Vision-Guided Robotic System for Automated Vegetable Sorting**, developed as an academic research project at **East West University**.

The proposed framework integrates **Computer Vision**, **Deep Learning**, and **Robotic Manipulation** to automatically detect, localize, and sort vegetables in real time. Unlike conventional robotic sorting systems that depend on predefined object locations, this work introduces a **dynamic vision-guided picking strategy** capable of manipulating objects located anywhere inside a calibrated workspace.

The complete pipeline combines:

- 🟢 YOLO-based real-time vegetable detection
- 🟢 ArUCo marker-based workspace calibration
- 🟢 Image-to-world coordinate transformation
- 🟢 Inverse Kinematics (IK) for robotic arm control
- 🟢 ByteTrack-based trajectory validation for sensorless grasp verification

Additionally, this repository contains a comparative study of baseline YOLO models and Self-Supervised Learning (SSL) approaches (**BYOL, DINO, and MAE**) for improving feature representation and object detection performance.

---

## 🎯 Key Features

- Real-time vegetable detection using Ultralytics YOLO
- Dynamic pick-and-place within a calibrated workspace
- ArUCo marker-based workspace calibration
- Image-to-world coordinate transformation
- Inverse Kinematics for robotic arm manipulation
- ByteTrack-based vision-only pick validation
- Comparative evaluation of YOLOv10, YOLOv11 and YOLOv12
- Self-Supervised Learning using BYOL, DINO and MAE
- Complete research implementation from training to deployment

---

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
ArUCo Marker Detection & Workspace Calibration
      │
      ▼
Image-to-World Coordinate Transformation
      │
      ▼
Inverse Kinematics (IK)
      │
      ▼
Robotic Arm Motion Planning
      │
      ▼
Dynamic Pick-and-Place
      │
      ▼
ByteTrack Trajectory Validation
      │
      ▼
Automated Vegetable Sorting
```

The proposed framework combines computer vision and robotic manipulation into a unified pipeline, enabling autonomous vegetable detection, localization, manipulation, and category-wise sorting within a calibrated workspace.

---

## 🎥 Project Demonstration

A complete demonstration of the vision-guided robotic sorting system is available below.

<div align="center">

[![Watch Demo](https://img.shields.io/badge/▶️-Watch%20Project%20Demo-red?style=for-the-badge)](https://drive.google.com/file/d/1YZWRKNW26gafVP3ZuEG97BOjcmkPyqSK/view?usp=drive_link)

</div>

The demonstration showcases:

- Real-time vegetable detection
- ArUCo marker-based workspace calibration
- Image-to-world coordinate transformation
- Inverse kinematics-based robotic manipulation
- Dynamic pick-and-place
- ByteTrack-based trajectory validation
- Automated vegetable sorting


---

## 💡 Dynamic Picking Strategy

Unlike traditional robotic pick-and-place systems that operate using predefined object coordinates, the proposed framework performs **dynamic object picking** within a calibrated workspace.

The manipulation process follows these steps:

1. Detect vegetables using the trained YOLO model.
2. Extract the center coordinates of each detected bounding box.
3. Establish the robot workspace using **ArUCo marker calibration**.
4. Convert image coordinates into real-world robot coordinates.
5. Compute joint angles using **Inverse Kinematics (IK)**.
6. Execute the robotic pick-and-place operation.
7. Validate the manipulation using **ByteTrack trajectory tracking**.

This enables the robotic arm to accurately grasp vegetables positioned anywhere inside the calibrated workspace without requiring predefined pick locations.

---

## 🔍 Vision-Based Pick Validation

The robotic arm used in this project does not include force, tactile, or proximity sensors for grasp verification.

To overcome this limitation, the framework employs **ByteTrack** as a vision-based validation mechanism. After the robotic arm attempts a grasp, ByteTrack continuously tracks the detected object across consecutive frames. A successful pick is confirmed when the object's trajectory consistently follows the expected robotic arm motion.

### Advantages

- Sensorless grasp verification
- Low-cost deployment
- Reduced hardware complexity
- Robust trajectory-based validation
- Continuous visual feedback during manipulation

---

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
└── README.md
```

The repository contains the complete implementation pipeline, including model development, experimentation, and robotic deployment.

---

## 🤖 Deep Learning Models

This project investigates both baseline object detection models and Self-Supervised Learning (SSL) techniques to improve feature representation for robotic perception.

| Category | Models |
|:---------|:-------|
| **Baseline Detection** | YOLOv10, YOLOv11, YOLOv12 |
| **Self-Supervised Learning** | BYOL, DINO, MAE |

The SSL models are pre-trained to learn robust visual representations before fine-tuning the object detector, enabling a comparative evaluation under the same experimental settings.

---

## 🥕 Dataset

The dataset used in this project was **collected, curated, cleaned, and manually annotated** by the research team as part of an academic research project conducted under the **Department of Computer Science & Engineering, East West University, Dhaka, Bangladesh**.

The dataset consists of annotated images of eight vegetable categories collected from multiple publicly available sources and manually annotated using **Roboflow** in YOLO format. All annotations were reviewed and verified by the research team to ensure consistency and suitability for robotic object detection.

### Dataset Information

| Attribute | Description |
|:----------|:------------|
| Vegetable Categories | 8 |
| Quality Labels | Fresh, Damaged |
| Total Quality States | 2 |
| Annotation Format | YOLO |
| Image Resolution | 640 × 640 |
| Annotation Tool | Roboflow |
| Developed By | Vision Intelligence and Robotics Research Team |
| Affiliation | Department of Computer Science & Engineering, East West University |

### Vegetable Categories

- 🍅 Tomato
- 🥔 Potato
- 🧅 Onion
- 🍋 Lemon
- 🍆 Eggplant
- 🥒 Cucumber
- 🥒 Bitter Gourd
- 🥒 Pointed Gourd

### Dataset Source

The complete annotated dataset is publicly available through **Roboflow Universe**.

🔗 **Dataset Link**

https://universe.roboflow.com/sanjana-kazi-supti-ymhu2/non-seasonal-vegetable-detection-yms0u

The dataset configuration used during training is provided through the `data.yaml` file.

> **Note:** This dataset was developed by the Vision Intelligence and Robotics Research Team as part of an academic research project at East West University. If you use this dataset in your research, please acknowledge the authors and cite the associated publication when available.

---

## ⚙️ Technology Stack

| Category | Technologies |
|:---------|:-------------|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Object Detection | Ultralytics YOLO |
| Computer Vision | OpenCV |
| Workspace Calibration | ArUCo Markers |
| Multi-Object Tracking | ByteTrack |
| Robotics Platform | Hiwonder xArm 1S |
| Motion Planning | Inverse Kinematics |
| Dataset Management | Roboflow |
| Development Environment | Jupyter Notebook, Google Colab |

---

## 📦 Repository Contents

This repository includes the complete implementation of the proposed vision-guided robotic sorting framework.

| Component | Description |
|:----------|:------------|
| 💻 Final Code | Complete robotic arm automation and deployment pipeline |
| 🤖 Trained Model | Pre-trained YOLO detection model (`best.pt`) |
| ⚙️ Configuration | Dataset configuration file (`data.yaml`) |
| 📚 Baseline Models | YOLOv10, YOLOv11, and YOLOv12 training notebooks |
| 🧠 SSL Models | BYOL, DINO, and MAE implementation notebooks |
| 🧪 Experimental Codes | Trial-and-error implementations used during system development |

The repository provides a complete workflow from model training and evaluation to real-time robotic deployment for automated vegetable sorting.

---

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Shams200648/Vision-Guided-Robotic-System.git
cd Vision-Guided-Robotic-System
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` is unavailable, install the required libraries manually according to your development environment.

---

## ▶️ Usage

Run the final robotic automation pipeline:

```bash
python "Final Code/Robotic_Arm_Automation_Code_Final_CAPSTONE_C.py"
```

Before execution, ensure that:

- ✅ The robotic arm is properly connected.
- ✅ The camera is initialized and functioning.
- ✅ ArUCo markers are correctly positioned within the workspace.
- ✅ The trained YOLO model (`best.pt`) is available.
- ✅ The serial communication port is configured correctly.

---

## 👥 Research Team

This project was conducted by the **Vision Intelligence and Robotics Research Team** under the **Department of Computer Science & Engineering, East West University, Dhaka, Bangladesh**.

### Team Members

- **Fathhur Rahaman Sams**
- **Sanjana Kazi Supti**
    Github-https://github.com/sanjana514
- **Md. Junaeid Ali**
- **Mahfuj Alam Imon**

The team was responsible for:

- Dataset collection and curation
- Data annotation and quality verification
- Model development and evaluation
- Vision-guided robotic manipulation
- Experimental validation
- System integration and deployment

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you identify bugs or have ideas for improving the project, please open an **Issue** or submit a **Pull Request**.

---
## 📄 License

This repository is licensed under the **MIT License**.

The repository includes research code, Jupyter notebooks, trained model weights, configuration files, and supporting documentation developed as part of an academic research project.

The **Non-Seasonal Vegetable Detection Dataset** is distributed separately under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

See the [LICENSE](LICENSE) file for the complete terms governing the source code and implementation materials.

⭐ If you found this project useful, please consider giving the repository a star.

</p>

---

## 🙏 Acknowledgements

The authors gratefully acknowledge the support of the **Department of Computer Science & Engineering, East West University**, for providing the academic environment and resources that facilitated this research.

We also acknowledge the open-source communities behind **Ultralytics YOLO**, **OpenCV**, **PyTorch**, **ByteTrack**, **Roboflow**, and **Google Colab**, whose tools and resources contributed significantly to this work.

---

<p align="center">

---


