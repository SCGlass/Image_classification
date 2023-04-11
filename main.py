import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# importing the training and testing from dataset in tensorflow. Saving them as tuples
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()