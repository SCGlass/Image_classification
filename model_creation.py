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

# reducing the amount of images to put into the training model, this saves time, but accuracy may be lower
training_images = training_images[:20000] # using up to 20000
training_labels = training_labels[:20000] # using up to 20000
testing_images = testing_images[:4000] # reducing the testing as well
testing_labels = testing_labels [:4000]

# starting to build the neural network
model = models.Sequential()
# adding input layer as converlutional layer, 32 neurons and 3,3 as conver layer
# input shape is 32 X 32 pixels and 3 color grades
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3))) # rectofying liner unit, TODO check it out
model.add(layers.MaxPooling2D((2,2))) # TODO check out what this does
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
# now will flatten the inputs
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu")) # TODO check out
model.add(layers.Dense(10, activation="softmax")) # this is the last layer which scales to a probability percentage

# compiling the model #TODO check out what the parameters do
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# fitting model to the training data.
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels)) #epochs is how many times the model is going to see the data again

# evaluating the model with testing
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# saving the model so it is trained only once
model.save("image_classifier.model")




