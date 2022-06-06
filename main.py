
import os
import cv2
import pydotplus as pydot
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.client.session
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, concatenate,Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from IPython.display import display
import matplotlib.pyplot as plt
from IPython.display import display
import matplotlib.pyplot as plt
from tensorflow.keras import initializers

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

    ######
    train_datagen = ImageDataGenerator(validation_split=0.2)  # ,brightness_range= [0.7, 1.3],)
    # preprocessing_function=preprocessing_function)

    train_gen = train_datagen.flow_from_directory(path, target_size=target_size, shuffle=True, class_mode='categorical',
                                                  batch_size=32,
                                                  subset='training', color_mode='grayscale')
    val_gen = train_datagen.flow_from_directory(path, target_size=target_size, shuffle=True, class_mode='categorical',
                                                batch_size=32,
                                                subset='validation', color_mode='grayscale')
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    # layer_med = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    print("INIT ", initializer)

    model_med = Sequential()
    # model_med.add(Conv2D(16,kernel_size=2,strides=1, activation='relu', padding='same',
    #                      input_shape=(img_height, img_width, 1),kernel_initializer=initializers.RandomNormal(
    #         mean=0.,stddev=0.01))
    #               )
    model_med.add(Conv2D(16,kernel_size=2,strides=1, activation='relu', padding='same',
                         input_shape=(img_height, img_width, 1),kernel_initializer=initializers.RandomUniform(minval=-0.05,maxval=0.05))
                  )
    # layer_med = tf.keras.layers.Dense(3, kernel_initializer='zeros')
    model_med.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model_med.add(Conv2D(32, kernel_size=2, strides=1, activation='relu', padding='same'))
    model_med.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model_med.add(Conv2D(64, kernel_size=2, strides=1, activation='relu', padding='same'))
    model_med.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model_med.add(Conv2D(128, kernel_size=2, strides=1, activation='relu', padding='same'))
    model_med.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model_med.add(Conv2D(256, kernel_size=2, strides=1, activation='relu', padding='same'))
    model_med.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model_med.add(Flatten())

    model_med.add(Dense(128, activation='sigmoid'))
    model_med.add(Dense(classes, activation='softmax'))

    model_med.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=(['accuracy']))

    # model.layers[0].set_weights()
    # print(model.layers[0].get_weights()[0].shape)
    # print(model.layers[0].get_weights())

    model_med.summary()

    plot_model(model_med, show_shapes=True, to_file="model_med.png")

    # plot_model(model_light, show_shapes=True, to_file="model_light.png")

    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.reduce_sum(tf.random.normal([1000, 1000])))
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # configproto = tf.compat.v1.ConfigProto()
    # configproto.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=configproto)
    # tf.compat.v1.keras.backend.set_session(sess)

    ########
    ## SET WEIGHT
    # weight, bias = model_med.layers[0].get_weights()
    #
    # weight = np.full(
    #     shape=weight.shape,
    #     fill_value=1)
    #
    # bias = np.full(
    #     shape=bias.shape,
    #     fill_value=1)
    # model_med.layers[0].set_weights([weight, bias])



    #before
    print("BEFORE LEARNING")
    print(model_med.layers[0].get_weights()[0].shape)
    print(model_med.layers[0].get_weights())


    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    # history_light = model_light.fit(train_gen, validation_data=val_gen, epochs=10)
    with tf.device('/cpu:0'):
        model = model_med.fit(train_gen, validation_data=val_gen, epochs=10)

    ########
    # model_med.layers[0].set_weights()
    print("AFTER LEARNING")
    print(model_med.layers[0].get_weights()[0].shape)
    print(model_med.layers[0].get_weights())



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   load_dataset()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
