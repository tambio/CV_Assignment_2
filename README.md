# Computer Vision A2: Level-Up  A1 Topic with SOTA CV

## Project Overview
This project is the continuation of Assignment 1 (Household Object Recognition). The goal of A2 is to upgrade the baseline model to a State-of-the-Art (SOTA) architecture, add visual explainability, and deploy an interactive web application.

The model classifies images into 3 categories:
*    **Mug**
*    **Pen**
*    **Funko Pop**

## The Upgrade (A1 vs A2)
*   **A1 Baseline:** MobileNetV2. Achieved 100% accuracy, but had higher latency (6.77 ms/image).
*   **A2 SOTA:** **EfficientNet-B0**. Achieved 100% accuracy with **faster inference** (5.01 ms/image).
*   **Explainability:** Integrated **Grad-CAM** to visualize where the neural network is focusing its attention.
*   **App:** Built an interactive UI using **Streamlit**.

## Results Summary
| Model | Accuracy | F1-Score | Latency (ms/image) |
| :--- | :--- | :--- | :--- |
| A1 Baseline (MobileNetV2) | 100% | 1.0000 | 6.77 ms |
| **A2 SOTA (EfficientNet-B0)** | **100%** | **1.0000** | **5.01 ms** |
| A2 Ablation (No Augmentation) | 100% | 1.0000 | 4.07 ms |

## Project Structure
*   `app.py` — The Streamlit web application (Grad-CAM + Latency).
*   `src/train_sota.py` — Training script for the EfficientNet-B0 model.
*   `src/train_ablation.py` — Training script for the Ablation study (without data augmentation).
*   `src/evaluate_sota.py` — Evaluation script comparing A1 baseline, A2 SOTA, and Ablation models.
*   `outputs/` — Saved metrics, plots, and training curves.

## Run Locally

### 1. Install Requirements
Make sure you have Python installed, then run:
pip install -r requirements.txt

### 2. Run the Web Application
Start the Streamlit server:
**streamlit run app.py**

### 3. Evaluate the Models
To reproduce the metrics table, run:
python src/evaluate_sota.py

## Demo Video
[Click here to watch the 3-minute demo video](https://drive.google.com/file/d/1m3eSxyVjVsiEqprrlYgWnXvl_iHlRxor/view?usp=drive_link)

##  Author
### **Amir Orman**
### Astana IT University — CV