# Image-Classification

# CIFAR-10 Image Classification using Convolutional Neural Networks

This project involves building and deploying a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset. The project is divided into two main parts: training the model and classifying images using the trained model.

## Project Structure
.
├── train_model.py # Script for training the CNN model
├── classify_images.py # Script for classifying images using the trained model
├── images # Directory containing sample images for classification
│ ├── airplane.jpg
│ ├── cat.jpg
│ ├── dog.jpg
│ └── ship.jpg
├── cifar10_cnn_model.h5 # Saved trained model (generated after running train_model.py)

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

Install the required packages using pip:
pip install tensorflow numpy matplotlib


## Dataset
The CIFAR-10 dataset is automatically downloaded when running the training script. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Training the Model
To train the CNN model, run the train_model.py script:
python train_model.py

This script will:
Load and preprocess the CIFAR-10 dataset.
Define and compile a CNN model.
Train the model for 10 epochs.
Evaluate the model on the test dataset.
Save the trained model to cifar10_cnn_model.h5.


## Classifying Images
To classify new images using the trained model, run the classify_images.py script:
python classify_images.py

This script will:
Load the saved model (cifar10_cnn_model.h5).
Define a function to preprocess and classify images.
Classify and display the specified images from the images directory.

## Model Architecture
Input Layer: 32x32x3 images
Convolutional Layers: Three Conv2D layers with ReLU activation
Pooling Layers: Two MaxPooling2D layers
Dense Layers: Two Dense layers with ReLU activation
Output Layer: Dense layer with 10 units (one for each class) and softmax activation


## Performance
The model achieved a test accuracy of 70% on the CIFAR-10 dataset after 10 epochs of training.
