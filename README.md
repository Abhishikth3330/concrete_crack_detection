# 🧱 Concrete Crack Detection using Deep Learning

A deep learning-based web application to detect cracks in concrete structures using image classification techniques.

---

## 📌 Overview

This project aims to automatically detect cracks in concrete surfaces using Convolutional Neural Networks (CNNs). We built, trained, evaluated, and deployed a deep learning model that classifies images as either:

- ✅ **No Crack**
- ❌ **Crack Detected**

The model is deployed using **Gradio** as a user-friendly web interface.

---

## 🧠 Methodology

### 1. **Data Collection**
- Source: `Concrete Crack Images for Classification` from Kaggle or other datasets.
- Structure:
data/raw/

├── Positive/ (cracked images)

└── Negative/ (non-cracked images)


### 2. **Preprocessing**
- Resizing all images to `150x150`.
- Normalizing pixel values to the range `[0, 1]`.

### 3. **Model Building**
- Architecture:
- Convolutional Layers
- MaxPooling
- Flatten + Dense
- Sigmoid Output Layer (binary classification)
- Loss Function: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`

### 4. **Training**
- Trained using `ImageDataGenerator` with 20% validation split.
- Early stopping and `ModelCheckpoint` to save the best model (`final_model.h5`).

### 5. **Evaluation**
- Evaluated on validation data.
- Accuracy and loss plotted.
- Final model accuracy: **~98–99%**

### 6. **Deployment**
- Used **Gradio** to build a web interface.
- Supports uploading an image from your device.
- Predicts and returns one of two outcomes.

---

## 🚀 How to Run Locally

### 1. Clone the Repo


- git clone https://github.com/your-username/concrete-crack-detection.git


### 2. Install Dependencies

Make sure you have Python ≥ 3.8 installed.

Use the following command to install all required libraries:

- pip install -r requirements.txt

### 3. Run the App
Once dependencies are installed, launch the Gradio web app using:
- python app/app.py

After a few seconds, a local web URL (like http://127.0.0.1:7860) will open in your browser. You can upload any image to see the prediction results.


### 🧠 Model Details
- Input Size: 150x150 RGB images

- Model: Convolutional Neural Network (CNN)

- Output: Binary classification (Crack or No Crack)

- Final Layer: Sigmoid activation

- Loss: Binary Crossentropy

- Optimizer: Adam



### 📦 Dependencies
- TensorFlow

- NumPy

- Pillow (PIL)

- Gradio

All included in requirements.txt.


### 🧪 Testing
You can test the model using images from the samples/ folder, or upload your own cracked or non-cracked concrete images.



### 🔧 Troubleshooting
❗ If you get an error like "No module named X", make sure to run pip install -r requirements.txt.

📷 Images must be in formats like JPG or PNG.

🧠 If you get a model-related error, ensure that final_model.h5 is correctly located in the models/ folder.

