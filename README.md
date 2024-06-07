# Image Classifier using Convolutional Neural Network (CNN)

This project is an image classifier built using a Convolutional Neural Network (CNN) with TensorFlow and Keras. It is trained on the CIFAR-10 dataset and can classify images into one of ten classes: Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Loading and Using the Pre-trained Model](#loading-and-using-the-pre-trained-model)
7. [Usage](#usage)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/alexanderjoossens/image-classifier.git
    cd image-classifier
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have the following libraries installed:
    - OpenCV (`cv2`)
    - NumPy
    - Matplotlib
    - TensorFlow

## Dataset

The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

## Model Architecture

The CNN model is constructed using the following layers:

- 3 Convolutional Layers with ReLU activation
- 2 MaxPooling Layers
- 1 Flatten Layer
- 1 Dense Layer with 64 units and ReLU activation
- 1 Output Dense Layer with 10 units and softmax activation

## Training the Model

To train the model, uncomment the relevant sections in the script and run the following:

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')
```

## Evaluating the Model

After training, the model is evaluated using the testing dataset:

```python
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
```

## Loading and Using the Pre-trained Model

The pre-trained model can be loaded using:

```python
model = models.load_model('image_classifier.keras')
```

## Usage

To use the pre-trained model for classification, you can run the following script. It reads an image, processes it, and predicts the class:

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier.keras')

img = cv.imread("images/paard.jpg")  # Replace with your image path
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)

print(f'Prediction is {class_names[index]}')
```

Replace `"images/paard.jpg"` with the path to the image you want to classify. The script will output the predicted class for the given image.

## Conclusion

This project demonstrates how to build, train, and use a CNN for image classification using the CIFAR-10 dataset. The pre-trained model can classify images into ten different categories, making it a valuable tool for various industrial applications. By following the instructions provided, you can replicate the results and further extend the model's capabilities.
