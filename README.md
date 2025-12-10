# 🤖 MNIST Digit Classifier (Convolutional Neural Network)

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using PyTorch. The model was successfully trained to achieve high classification accuracy, significantly exceeding the target requirement.

## 🚀 Project Goals & Achievements

| Requirement | Status | Result |
| :--- | :--- | :--- |
| **Test Accuracy Target** | ✅ Achieved | Final Test Accuracy: **~98.7%** |
| Model Implementation | ✅ Complete | Built and trained a 2-layer CNN. |
| Tooling | ✅ Complete | Utilized DataLoader, `autograd`, and **SGD optimizer**. |
| Evaluation & Validation | ✅ Complete | Generated **Training Curves** and **Confusion Matrices**. |
| Demonstration | ✅ Complete | Demonstrated model inference on random test samples. |

## 🛠️ Technology Stack

* **Deep Learning Framework:** PyTorch
* **Data Handling:** Torchvision
* **Optimization:** Stochastic Gradient Descent (SGD)
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Model Architecture

The core of the project is the `Net` class, a Convolutional Neural Network architecture designed for image classification.

| Layer Type | Parameters / Operation | Output Shape (after operation) |
| :--- | :--- | :--- |
| **Input** | 28x28 Grayscale Image | 1x28x28 |
| **Conv1** | `nn.Conv2d(1, 32, kernel_size=3)` | 32x26x26 |
| **Pool1** | `nn.MaxPool2d(2)` | 32x13x13 |
| **Conv2** | `nn.Conv2d(32, 64, kernel_size=3)` | 64x11x11 |
| **Pool2** | `nn.MaxPool2d(2)` | 64x5x5 |
| **Flatten** | `torch.flatten` | **1600** features |
| **FC1** | `nn.Linear(1600, 128)` + ReLU + Dropout | 128 features |
| **FC2 (Output)** | `nn.Linear(128, 10)` | 10 classes |

##  Getting Started

### Prerequisites
Ensure you have Python 3.7+ installed and the necessary libraries:
pip install torch torchvision numpy matplotlib scikit-learn seaborn pandas
### Execution
Open the project notebook (MNIST_Digit_Classifier.ipynb or equivalent Python script).

Run the code cells sequentially, starting with Data Loading, Model Definition, and the Training/Testing loop.

The final cells will generate the required evaluation plots:

Test Accuracy vs. Epoch: Visualizing learning progress.

Confusion Matrix: Showing which digits the model successfully or incorrectly classified.

Sample Inference: Displaying random test images with their predicted labels.

## Results
The model achieved its peak performance around Epoch 7-10. The Confusion Matrix confirms robust performance across all 10 digits, with minimal errors concentrated on visually similar digits (e.g., 4 vs. 9, 3 vs. 5).
