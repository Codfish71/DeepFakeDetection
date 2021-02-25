import tensorflow as tf
from tensorflow.keras.preprocessing import image
# Helper libraries
import numpy as np
import matplotlib.pyplot as pl
import h5py
from os import path
import sys

if getattr(sys, 'frozen', False):
    MODEL_DIRECTORY = path.join(path.dirname(sys.executable), '.')
else:
    MODEL_DIRECTORY = path.dirname('.')
    
    
mymodel= path.join(MODEL_DIRECTORY, 'custom_model.h5')

model = tf.keras.models.load_model(mymodel)
    
img = image.load_img(path.join(MODEL_DIRECTORY, 'fake4.jpg'),target_size=(224,224))

img_arr = image.img_to_array(img)

test_img = np.expand_dims(img_arr,axis=0)

result = model.predict(test_img)

print(result)