# Driver_drowsiness_and_drunk_detection
An AI combined  with IOT devices that detect drowsiness, drunk and alert you
## dataset
- download mrleyedataset 
- download haarcascade_frontalface_default.xml dataset
- download haarcascade_eye.xml dataset
## AI Pre-trained model
- ### Drowsiness_ai_model -
This project implements a deep learning model to classify whether an eye is open or closed using TensorFlow and MobileNet. The model is trained on an eye image dataset and can be used for applications like drowsiness detection, eye-tracking, and biometric authentication.

📌  Features
Pretrained MobileNet model for efficient feature extraction.
Binary classification (open vs. closed eyes).
Dataset preprocessing including normalization and resizing.
Model training and validation with an 80-20 train-test split.
Pickle-based dataset saving for future use.

🛠 Installation
Ensure you have Python installed (>=3.6) and run the following:
pip install tensorflow numpy opencv-python matplotlib scikit-learn

📂 Dataset
Prepare a dataset with two folders inside a main directory:

dataset

    ├── close_eye/  # Images of closed eyes
    |
    ├── open_eye/   # Images of open eyes

Ensure that all images are in .png, .jpg, or .jpeg format.

🎯 Results
After training for 10 epochs, the model achieves binary classification accuracy for detecting open and closed eyes. Performance can be improved by:
Using data augmentation
Training with more images
Fine-tuning MobileNet layers
