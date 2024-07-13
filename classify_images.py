import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model = load_model('cifar10_cnn_model.h5')

# Function to preprocess and classify an image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return img, predicted_class

# List of image paths to classify
image_paths = ['images/airplane.jpg', 'images/cat.jpg', 'images/dog.jpg', 'images/ship.jpg']

# Classify and display each image
plt.figure(figsize=(10, 10))
for i, img_path in enumerate(image_paths):
    img, predicted_class = classify_image(img_path)
    plt.subplot(1, len(image_paths), i + 1)
    plt.imshow(img)
    plt.title(predicted_class)
    plt.axis('off')
plt.show()
