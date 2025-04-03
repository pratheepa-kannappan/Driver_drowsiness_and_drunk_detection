# Driver_drowsiness_and_drunk_detection
An AI combined  with IOT devices that detect drowsiness, drunk and alert you
## dataset
- download mrleyedataset 
- download haarcascade_frontalface_default.xml dataset
- download haarcascade_eye.xml dataset
## AI Pre-trained model
- ### Drowsiness_ai_model -
This project implements a deep learning model to classify whether an eye is open or closed using TensorFlow and MobileNet. The model is trained on an eye image dataset and can be used for applications like drowsiness detection, eye-tracking, and biometric authentication.

📌 Features

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

    ├── close_eye  # Images of closed eyes
    |
    ├── open_eye   # Images of open eyes

Ensure that all images are in .png, .jpg, or .jpeg format.

🎯 Results

After training for 10 epochs, the model achieves binary classification accuracy for detecting open and closed eyes. Performance can be improved by:
Using data augmentation
Training with more images
Fine-tuning MobileNet layers

- ### Drunk_ai_model -

Overview
This project implements a Random Forest Classifier to determine alcohol intoxication levels based on sensor readings. The model is trained to classify whether a person is sober (0) or drunk (1).

How It Works
Data Collection: A sample dataset is created with sensor values (e.g., alcohol concentration levels).

📌 Training: A Random Forest Classifier with 100 decision trees is trained to learn the patterns.
    Model Saving: The trained model is saved using joblib for later use.

🛠 Evaluation:
- A confusion matrix is plotted to visualize classification accuracy.
- The accuracy score and classification report show the model’s performance.
- An ROC curve is generated to evaluate how well the model differentiates between sober and drunk states.

🎯 Results & Improvements
Provides an accuracy score and a visual representation of performance.
Can be improved by collecting more real-world data, tuning hyperparameters, or using different machine learning models.
