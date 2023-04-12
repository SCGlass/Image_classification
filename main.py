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

training_images = training_images[:20000] # using up to 20000
training_labels = training_labels[:20000] # using up to 20000
testing_images = testing_images[:4000] # reducing the testing as well
testing_labels = testing_labels [:4000]

# loading the saved model
model = models.load_model("image_classifier.model")

#loading in the picture
img = cv.imread("deer.jpg")
# convert the color scheme
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# shows the image
plt.imshow(img, cmap=plt.cm.binary)
# doing the prediction
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction) # gives us the neuron with highest activation
print(f"Prediction is {class_names[index]}")

plt.show()
