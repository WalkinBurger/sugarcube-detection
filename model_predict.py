import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

model = load_model('model/sugarcube_detection.h5')

INPUT_DIR = 'predict_test_imgs/test1.png'
IMG_SIZE = 128

img_array = cv2.imread(INPUT_DIR, cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
new_array = np.reshape(new_array, (-1,128,128,1))

prediction = model.predict(new_array)
print(prediction)
if prediction[0][0] >= 0.5:
    print('pass')
else:
    print('fail')