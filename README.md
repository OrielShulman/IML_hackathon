
# Cancer Detection Mission - Hackathon 2022

## Overview
This project was developed as part of Hackathon 2022 to detect attributes of breast cancer using machine learning models. Our mission focused on improving medical diagnoses by predicting metastasis sites and tumor sizes based on patient visit characteristics. This project has the potential to aid doctors in making more informed decisions and reduce the need for expensive, time-consuming tests.

## Challenge
The challenge was divided into two main parts, both focused on breast cancer attributes:
1. **Predicting Metastases Sites**: A multi-label classification task to predict metastases sites based on patient visits.
2. **Predicting Tumor Size**: A regression task to predict the tumor size (in millimeters) given patient visit data.

## Dataset
The dataset provided for this challenge contains **65,798 records** across training and testing sets, with each record representing a patient's visit. There are 34 features describing patient demographics, medical history, tumor markers, and treatment details.

The dataset is split into:
- `train.feats.csv` (49,351 records) for training
- `test.feats.csv` (16,447 records) for testing

Each record contains the following features (high-level):
- Patient information (age, diagnosis date, hospital, etc.)
- Tumor details (tumor size, tumor markers like HER2, etc.)
- Surgery details (dates, types, etc.)
- Metastasis markers (M, N, T markers)
  
## Tasks

### 1. Predicting Metastases Sites
We trained a multi-label classifier to predict metastasis sites for each patient based on the provided features. We used various machine learning techniques and evaluated performance using **Micro** and **Macro average F1 scores**.

### 2. Predicting Tumor Size
For the second task, we implemented a regression model to predict the tumor size in millimeters. The model was evaluated using **Mean Squared Error (MSE)**.

### Bonus Task: Unsupervised Data Analysis
As an extra, we explored the dataset using **unsupervised learning techniques** (e.g., clustering, PCA) to find interesting patterns and insights.
