<img width="788" height="412" alt="image" src="https://github.com/user-attachments/assets/3bc1db80-4f0e-4d7b-8f6e-1ad4b6e16de6" />🌸 Flower Classification using Neural Networks
Overview

This project is a Digital Image Processing assignment focused on classifying flowers using a 4-layer Neural Network in Python. The dataset consists of multiple flower categories, and the model is trained to recognize and classify images accurately.

The project demonstrates the use of image preprocessing, neural network design, and evaluation metrics for a simple computer vision task.

🖥️ Project Details

Domain: Digital Image Processing / Computer Vision

Dataset: Flowers (multiple classes)

Tech Stack:

Python

TensorFlow / Keras

NumPy

ImageDataGenerator for preprocessing

Neural Network Architecture:

Input layer: Flatten image data

Hidden Layer 1: Dense, 128 neurons, ReLU activation

Hidden Layer 2: Dense, 64 neurons, ReLU activation

Hidden Layer 3: Dense, 32 neurons, ReLU activation

Output Layer: Dense with softmax, number of neurons = number of classes

Image Size: 64x64 pixels

Training Strategy:

Train/Validation split: 80/20

Batch size: 32

Loss: Categorical Crossentropy

Optimizer: Adam

Epochs: 20


🔹 Features

Multi-class flower classification

4-layer fully connected Neural Network

Image preprocessing using ImageDataGenerator

Model evaluation on unseen test data

Model saved for future predictions

📊 Results

Accuracy on validation set: High (depends on dataset)

Test Accuracy: Variable based on dataset size and quality

Model can classify flowers from unseen images after training


⚙️ How to Run

Install required packages:

pip install tensorflow numpy


Ensure dataset is in the correct folder structure.

Open imageclassfication.ipynb in Jupyter Notebook and run all cells to train and evaluate the model.

Trained model is saved as flower_classification_model.h5 for future use.

📌 Notes

Make sure each flower class has enough images for proper training.

Adjust target_size, batch_size, and epochs according to your dataset and computing resources.

The project is suitable for beginners in Computer Vision and Deep Learning.
