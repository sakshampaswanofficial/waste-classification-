# ♻️ Smart Waste Classifier: Real-Time Hazardous Material Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## 📌 Overview
The **Smart Waste Classifier** is a real-time computer vision application designed to identify and categorize waste directly from a live webcam feed. Built with scalability and edge-deployment in mind, the model not only predicts the specific type of material but also categorizes it into actionable broad categories (**Hazardous** vs. **Non-Hazardous**).

This dual-layered classification approach is highly applicable for smart-bin hardware integrations, recycling plant automation, and sustainability-focused ventures.

## ✨ Key Features
* **Real-Time Webcam Inference:** Uses OpenCV to capture and process live video feeds with minimal latency.
* **Targeting System (ROI):** Features a "Region of Interest" bounding box to isolate the target waste item from messy background environments, drastically improving real-world accuracy.
* **Dual-Layered Outputs:** * Identifies 12 specific sub-categories (paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, battery, trash).
  * Automatically maps specific materials to parent safety categories for immediate visual feedback.
* **Hardware Accelerated:** Compatible with standard CPU inference, Linux WSL2 (CUDA), and Native Windows via the TensorFlow-DirectML plugin.

## 🧠 Model Architecture & Training
This project utilizes **Transfer Learning** to achieve high accuracy with a lightweight footprint.
* **Base Model:** MobileNetV2 (Pre-trained on ImageNet). Chosen for its high efficiency and low computational cost, making it ideal for real-time edge computing.
* **Fine-Tuning:** The model's top layers were unfreezed and trained specifically on the 12-class Garbage Classification dataset using categorical crossentropy.
* **Custom Baselines:** The repository also contains the architecture for a custom CNN built from scratch to benchmark against the MobileNetV2 feature extractor.

## 🗄️ Dataset
Trained using the [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification). 

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/sakshampaswanofficial/waste-classification-.git]
cd smart-waste-classifier
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn
```

*Note for Windows Users:* If you are running native Windows and want to utilize your GPU without installing a full WSL2 Linux subsystem, install the DirectML plugin:
```bash
pip install tensorflow-directml-plugin
```

### 3. Add the Pre-Trained Model
Ensure your trained `waste_classifier_12classes.keras` file is downloaded and placed in the root directory of the project. *(Note: If the file is larger than 100MB, use Git LFS or host it externally and link it here).*

## 💻 Usage

Run the live inference script to open your webcam and start classifying:

```bash
python live_classifier.py
```
* **Instructions:** Hold the waste item directly inside the blue targeting square on your screen.
* **Exit:** Press `q` while the video window is active to safely close the camera and application.

## 🔮 Future Roadmap
- [ ] **Data Augmentation Upgrade:** Implement dynamic lighting and rotation augmentation to handle diverse webcam setups.
- [ ] **Object Detection Integration:** Upgrade from Image Classification (MobileNetV2) to Object Detection (YOLOv8) to classify multiple pieces of waste scattered in a single frame.
- [ ] **Heavyweight Model Benchmark:** Train an EfficientNetB0 version to compare accuracy trade-offs against MobileNetV2.

## 👨‍💻 Author
**Saksham Paswan**

Feel free to reach out for collaborations, startup discussions, or contributions to this project!
```
