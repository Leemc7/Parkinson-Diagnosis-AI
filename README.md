# Parkinson-Diagnosis-AI
Multimodal Parkinson Detection using Voice and MRI

# Parkinson's Disease Prediction Using AI

## Overview
This project develops an AI model to predict Parkinson's disease using MRI images and structured voice data. The model utilizes deep learning and gradient boosting techniques for accurate diagnosis.

## Models Used
- SimpleCNN – Custom CNN for MRI classification.
- EfficientNetB0 _– Pretrained EfficientNet for feature extraction.
- ResNet50 – Fine-tuned ResNet50 for image-based classification.
- CatBoost – Gradient boosting on structured voice data.
- LightGBM – Lightweight gradient boosting model.
- XGBoost – Optimized gradient boosting model.


## Results
| Model | Accuracy |
|--------|----------|
| EfficientNetB0 | 88.55% |
| SimpleCNN| 82.79% |
| ResNet50 | 79.08% |
| CatBoost | 94.87% |
| LightGBM | 92.31% |
| XGBoost   | 94.87% |


