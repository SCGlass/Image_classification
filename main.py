import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# importing the training and testing from dataset in tensorflow. Saving them as tuples. It will return them as arrays as pixels 
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# we now need to scale the pixels from value of 0 to 255 to 0 to 1, Will normalize it. 
training_images, testing_images = training_images  / 255, testing_images / 255

# assigning labels as the info is numbers, these are the things the neural network can/will  identify
class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse","Ship", "Truck" ]

# now we will create a class which will display 16 images from the dataset
for i in range(16):
    plt.subplot(4,4, i +1) # 4x4 grid and each i + 1 will iterate over the images
    plt.xticks([]) # takes away coordinate system
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]]) # This will assign the labels depending on the number and the position in the list

plt.show()
# loading the saved model
model = models.load_model("image_classifier.model")







