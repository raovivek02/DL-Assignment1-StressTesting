# Stress-Testing of Convolutional Neural Networks (DL-2026)

## Project Overview
This repository contains the code and experimental results for **Assignment 1** of the Deep Learning . The objective of this project is to analyze the robustness of a custom Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

Rather than focusing solely on accuracy, this project investigates:
1.  **Baseline Performance:** Training a simple CNN from scratch.
2.  **Failure Analysis:** Identifying "confident failures" (high-confidence misclassifications).
3.  **Explainability:** Using Grad-CAM to visualize model attention and diagnose "Background Bias".
4.  **Robustness Improvement:** Implementing Data Augmentation (Random Flip & Crop) to mitigate bias and improve generalization.

## Dataset
* **Dataset:** CIFAR-10
* **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
* **Input Size:** 32x32 color images.
* **Source:** `torchvision.datasets.CIFAR10` (Official train-test splits used).

## Methodology & Results

### 1. Baseline Model
* **Architecture:** Custom 3-block CNN.
* **Training:** 15 Epochs, Adam Optimizer, CrossEntropyLoss.
* **Accuracy:** ~75%
* **Findings:** The model exhibited overfitting and "Background Bias" (e.g., classifying green backgrounds as "Horse").

### 2. Failure Analysis & Grad-CAM
* We extracted high-confidence errors (e.g., Frog -> Plane).
* **Grad-CAM visualization** confirmed the model was focusing on background pixels (sky, grass) rather than the object features.

### 3. Improvement (Data Augmentation)
* **Technique:** Random Horizontal Flip (p=0.5) and Random Crop (padding=4).
* **Result:** Accuracy improved to ~80% (+5%). The model learned to generalize better by ignoring positional/background artifacts.

## How to Run the Code

The entire project is self-contained in a single Jupyter Notebook. The easiest way to run the code and reproduce the results is via Google Colab.

### Option 1: Google Colab (Recommended)
1.  Download the file `CNN_Stress_Test_CIFAR10.ipynb` from this repository.
2.  Upload the file to [Google Colab](https://colab.research.google.com/).
3.  **Important:** Change the Runtime type to **GPU** (`Runtime` > `Change runtime type` > Select `T4 GPU`).
4.  Run all cells (`Runtime` > `Run all`).
5.  The notebook will automatically download the dataset, train the models, and generate the plots used in the report.

### Option 2: Local Execution
If running locally, ensure you have Python installed with the following libraries:
* `torch`
* `torchvision`
* `matplotlib`
* `numpy`
* `opencv-python` (cv2)

Run the notebook using Jupyter Lab or Jupyter Notebook:
```bash
jupyter notebook CNN_Stress_Test_CIFAR10.ipynb
```
### Reproducibility

Seed: A fixed random seed (42) is set at the beginning of the notebook to ensure that training results, data splits, and failure case selection are deterministic and reproducible.

### Group Members

VIVEK YADAV


ATHARv


KRUPAL

    
