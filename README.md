<!-- ====== Banner / Title ====== -->

<h1 align="center">🩻 X-Ray Pneumonia Detection</h1>
<h3 align="center">Deep Learning System for Detecting Pneumonia from Chest X-Ray Images</h3>

<p align="center">
Built with <b>PyTorch</b> • Medical Image Classification • Computer Vision
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch"/>
<img src="https://img.shields.io/badge/OpenCV-ComputerVision-green?style=for-the-badge&logo=opencv"/>
<img src="https://img.shields.io/badge/Status-Research_Project-orange?style=for-the-badge"/>
</p>

---

# 🧠 Project Overview

Pneumonia is a serious lung infection that can be detected from chest X-ray scans.
This project builds a **deep learning model using PyTorch** that analyzes X-ray images and predicts whether the patient has pneumonia.

The goal of this project is to demonstrate how **AI can assist medical diagnosis** using computer vision.

---

# 🧠 Problem Statement

Pneumonia is a lung infection that can be detected from chest X-ray images.  
Manual diagnosis can be time-consuming and requires expert radiologists.

This project uses **deep learning with PyTorch** to automatically analyze chest X-ray images and classify them as:

• **Normal**  
• **Pneumonia**

The system learns visual patterns from thousands of X-ray scans to assist in medical diagnosis.
---

# ⚙️ Model Pipeline

```
X-ray Image
     │
     ▼
Image Preprocessing
(resizing, normalization)
     │
     ▼
Deep Learning Model
(PyTorch CNN / EfficientNet)
     │
     ▼
Feature Extraction
     │
     ▼
Binary Classification
     │
     ▼
Prediction
Normal / Pneumonia
```

---

# 🧰 Tech Stack

| Technology | Purpose                 |
| ---------- | ----------------------- |
| Python     | Programming language    |
| PyTorch    | Deep learning framework |
| OpenCV     | Image processing        |
| NumPy      | Numerical computations  |

---

# 📂 Project Structure

```
XRAY_PNEUMONIA_DETECTION
│
├── train.py              # Training script
├── app.py                # Model inference / prediction
├── requirements.txt      # Project dependencies
├── pneumonia_model.pth   # Trained model weights
├── .gitignore
└── README.md
```

---

# 🧪 How the Model Works

1️⃣ Load the X-ray dataset
2️⃣ Preprocess images (resize & normalize)
3️⃣ Train CNN model using **PyTorch**
4️⃣ Extract visual features from lungs
5️⃣ Classify image as:

```
0 → Normal
1 → Pneumonia
```

---

# 🚀 Running the Project

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Train the model

```
python train.py
```

### 3️⃣ Run prediction

```
python app.py
```

---

# 🎯 Key Learning Outcomes

✔ Medical image classification
✔ Deep learning with PyTorch
✔ Image preprocessing pipelines
✔ Building an end-to-end AI project

---

# 👩‍💻 Author

**Ravakutam Mounika**
Robotics & Automation Engineering Student
Passionate about **AI, Robotics, and Intelligent Systems**

GitHub: https://github.com/Mounika0021

---

# ⭐ If you like this project

Give the repository a **star ⭐** to support the work!
