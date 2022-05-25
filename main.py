
import os
import cv2
import pydotplus as pydot
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, concatenate,Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

def load_dataset():
    path = 'dataset_reza'
    img_height, img_width = 100, 100
    target_size = (img_height, img_width)
    batch_size = 32
    labels = []
    list = os.listdir(path)
    for i in list:
        list1 = os.listdir('dataset_reza/' + i)
        labels.append(i)
    print(labels)
    classes = len(labels)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   load_dataset()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
