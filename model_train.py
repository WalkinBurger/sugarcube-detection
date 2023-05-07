import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


INPUT_DIR = 'train_imgs'
IMG_SIZE = 128


for class_i,classes in enumerate(os.listdir(INPUT_DIR)):
    class_path = os.path.join(INPUT_DIR, classes)
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path,img)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
        cv2.imwrite(img_path, new_array)

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,      
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)
traingen = datagen.flow_from_directory(
    directory=INPUT_DIR,
    target_size=((IMG_SIZE,IMG_SIZE)),
    batch_size=16,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=True,
)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    Dropout(0.05),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Dropout(0.05),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    Dropout(0.05),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(units=256, activation='relu'),
    Dropout(0.05),
    Dense(units=1, activation='sigmoid')
])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(traingen, batch_size=16, epochs=128, validation_data=traingen, validation_split=0.4)

model.save('model/sugarcube_detection.h5')
model.summary()