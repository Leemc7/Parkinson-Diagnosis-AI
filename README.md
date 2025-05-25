# Parkinson-Diagnosis-AI

**Multimodal Parkinson Detection using Voice and MRI Data**

## 🧠 Overview

This project implements multiple AI models to detect Parkinson's disease using a **multimodal approach** combining **structured voice data** and **MRI brain images**. The project leverages deep learning (PyTorch) and advanced gradient boosting methods (CatBoost, LightGBM, XGBoost) for high-accuracy classification.

## 🗂️ Project Structure

```
Parkinson-Diagnosis-AI/
├── data/                      # Voice features & image folders
├── models/                    # Saved trained models
├── notebooks/                 # Jupyter notebooks for training & evaluation
├── src/                       # Python scripts for model training & evaluation
│   ├── EfficientNetB0_Model_GPU_Graph_Final.py
│   ├── ResNet50_Model_GPU_Graph_Final.py
│   ├── ImageDataGenerator_Model_GPU_Graph_Final.py
│   ├── CatBoost Model.py
│   ├── XGBoost Model.py
│   └── LightGBM  Model.py
├── requirements.txt
└── README.md
```

## 📊 Dataset

- **MRI Images:** Brain scans preprocessed to size `128x128`.
- **Voice Data:** Structured features like Jitter, Shimmer, NHR, PPE from `parkinsons.data` dataset.
- **Sample distribution:**  
  - 8,415 healthy  
  - 22,856 Parkinson’s

## 🔧 Installation

```bash
git clone https://github.com/Leemc7/Parkinson-Diagnosis-AI.git
cd Parkinson-Diagnosis-AI
pip install -r requirements.txt
```

## 🚀 How to Run

### 📁 Voice-based models:

```bash
python "src/XGBoost Model.py"
python "src/LightGBM  Model.py"
python "src/CatBoost Model.py"
```

### 🖼️ MRI-based models (requires GPU):

```bash
python "src/EfficientNetB0_Model_GPU_Graph_Final.py"
python "src/ResNet50_Model_GPU_Graph_Final.py"
python "src/ImageDataGenerator_Model_GPU_Graph_Final.py"
```

Each script trains and evaluates the model, and saves:
- Accuracy metrics
- F1-scores
- Confusion matrix images
- Trained model weights (`.pth`)

## 🧪 Models and Performance

| Model            | Input Type     | Accuracy (%) |
|------------------|----------------|--------------|
| EfficientNetB0   | MRI            | 88.55%       |
| SimpleCNN        | MRI            | 82.79%       |
| ResNet50         | MRI            | 79.08%       |
| XGBoost          | Voice Features | 94.87%       |
| LightGBM         | Voice Features | 92.31%       |
| CatBoost         | Voice Features | 94.87%       |

## 📈 Architecture Summary

- **Image Models:** CNN-based deep learning using PyTorch and EfficientNet.
- **Voice Models:** Structured data classification using boosting algorithms.
- **Ensemble potential:** Combination of predictions can further enhance diagnostic accuracy.

## 📌 Limitations & Future Work

- MRI data size is limited – potential to improve via augmentation or access to 3D scans.
- Voice features are static – future versions could use time-series or raw audio with RNNs.
- Integrating clinical metadata (age, gender, UPDRS) could further improve accuracy.

## 📄 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## 👤 Author

Developed by Leemc7 | 2025  
Course: Introduction to Artificial Intelligence – Parkinson’s Disease Detection Project
