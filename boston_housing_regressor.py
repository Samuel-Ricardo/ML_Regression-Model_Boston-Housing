# -*- coding: utf-8 -*-
"""Boston Housing Regressor.ipynb

# Setup and Install Deps

https://www.python.org/

https://www.tensorflow.org/?hl=pt-br

https://keras.io/

https://www.anaconda.com/
"""

!pip install -q -U tensorflow
!pip install -q -U keras
!pip install -q -U numpy
!pip install -q -U pandas
!pip install -q -U tensorflow-addons
!pip install -q -U keras-utils

#--use-deprecated=legacy-resolver

#!pip install -q -U tensorflow==2.15
#!pip install -q -U keras==2.3.1
#!pip install -q -U numpy pandas==2.0.3 tensorflow-addons keras-utils

import os
import keras
import numpy as np
import cv2
import PIL
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
#import keras.layers.convolutional as conv
from keras.layers import Conv2D as conv
from math import sqrt
from PIL import Image
from tensorflow import keras
from numpy import mean
from keras.models import Sequential, load_model
from tensorflow.keras import regularizers, layers, Model
from keras.preprocessing import image
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
#from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
from keras import backend as B
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Model
from keras.layers import Dense, Add, Conv1D, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, ZeroPadding2D, MaxPooling2D, Activation, Input, UpSampling2D, AveragePooling2D, Reshape, InputLayer, SeparableConv2D
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.exposure import equalize_adapthist
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, rank, median
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.metrics import *
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support, f1_score

"""# Linear Regression | Boston Housing

https://keras.io/api/datasets/boston_housing/

http://lib.stat.cmu.edu/datasets/boston

Boston house price data from Harrison, D. and Rubinfeld, D.L. for the article 'Hedonic Prices and the Demand for Clean Air', J. Environ. Economics and Management, vol.5, 81-102, 1978.

- Per capita crime rate by city.
- Proportion to residential land zoned in lots over 25,000 square feet.
- Proportion in relation to hectares of non-retail land by city
 - What is the relationship between residential land and commercial land?
- Proportion in relation to land allocated with rivers
- Concentration of nitric oxides measured on the ground (parts per 10 million)
 - Amount of trash
- Average number of rooms per dwelling in the same subdivision
 - Quite common to have more rooms left to rent to guests
 - These extra rooms are usually planned / estimated during construction
- Proportion of occupied units (subdivision) and built before 1940
 - Get a sense of how much that area evolved at a specific time
- Weighted distances to five Boston job centers
 - Distance from the subdivision to the centers
- Accessibility index to radial highways
- Average value property tax rate (scaled by 10,000)
- Ratio of students per teacher by city
- Proportion of black people by city (per thousand inhabitants)
- Lowest status percentage of the population
- Average home value (scaled by 1,000)

The goal is to use linear regression to find the average value of owner-occupied homes at $1,000.
"""
#Import Libs, modules and Packages

import tensorflow as tf # to create the model, the architeture, of neural network
import numpy as np # Visualize Data

from keras.models import Sequential # I Will use this tool to create my model, Sequencial Layers Model
from keras.layers import Dense # This is the Layer Type used

import matplotlib.pyplot as plt # Graphic View

print(f'TensorFlow Lib Version: {tf.__version__}')

# Load Data Base | # https://keras.io/api/datasets/boston_housing/ # |

data = tf.keras.datasets.boston_housing

# Separate Test Data and Train Data

(x_train, y_train), (x_test, y_test) = data.load_data()

# Data Type

print(type(x_train))

# Train Data Format

print(x_train.shape)

# 404 Data Lines
# 13 Different Categories

# Test Data Format

print(x_test.shape)

# 102 Samples
# 13 Different Categories

# Show First Train Data Sample

print(x_train[0])

# Show First Test Data Sample

print(x_test[0])

# Show First Test Data Sample Answer

print(y_test[0])

# Normalization Through Mean and Standard Deviation

media = x_train.mean(axis = 0)
desvio = x_train.std( axis = 0 )

x_train = (x_train - media) / desvio
x_test = (x_test - media) / desvio

# DOCS:
# | https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# | https://numpy.org/doc/stable/reference/generated/numpy.std.html

# MY MODEL:

model = Sequential([
    Dense(
      units = 64,
      activation = 'relu',
      input_shape = [13]
    ),
    Dense(
      units = 64,
      activation = "relu"
    ),
    Dense( units = 1 )
])

# DOCS:
# | https://keras.io/api/models/sequential/#sequential-class
# | https://keras.io/api/layers/core_layers/dense/
# | https://keras.io/api/layers/activations/

model.summary()

tf.keras.utils.plot_model(model,
           to_file = 'model.png',
           show_shapes = True,
           show_layer_names = False)

# OPTIMIZE MODEL ALGORITHM
# RUN WITH MODEL AND OPTIMIZE IN EACH CYCLE GENERATION

model.compile(
    optimizer = "adam",
    loss = "mse", # Mean Squared Error
    metrics = ["mae"] # Mean Absolute Error
    )

# DOCS:
# | https://keras.io/api/optimizers/
# | https://keras.io/api/losses/
# | https://keras.io/api/metrics/

# Commented out IPython magic to ensure Python compatibility.
# %time

# TEST MODEL WITH TRAIN DATA

history = model.fit(
    x_train,
    y_train,
    epochs = 100,           # EPOCHS / CYCLES / ROUNDS OF TRAIN
    validation_split = 0.2  # AUTO METRICS OBSERVE 80% OF TRAIN DATA AND 20% OF TEST DATA TO AVALUETA THE QUALITY OF MODEL
)

# HISTORIC

print(history.history.keys())
print(history.history)

# GRAPH VIEW

plt.rcParams['figure.figsize'] = (12.0, 6.0)  # WIDTH x HEIGHT

plt.plot(history.history['loss'])      # DATA
plt.plot(history.history['val_loss'])  # DATA

plt.legend(['Loss Rate', 'Loss Rate (Validation)'], loc = 'upper right', fontsize = 'x-large')
plt.xlabel('Processing Epochs', fontsize = 16)
plt.ylabel('Value', fontsize=16)
plt.title('Average Loss Rate', fontsize=18)

plt.show()

# GRAPH VIEW

plt.rcParams['figure.figsize'] = (12.0, 6.0)  # WIDTH x HEIGHT

plt.plot(history.history['mae']) # DATA

plt.legend(['Absolute Loss'], loc = 'upper right', fontsize = 'x-large')
plt.xlabel('Processing Epochs', fontsize = 16)
plt.ylabel('Value', fontsize=16)
plt.title('Average Loss Rate', fontsize=18)

plt.show()

# USE MODEL IN PRODUCTION
# SIMULATE THAT A CUSTOMER HAS PASSED IN THE DATA OF A HOUSE AND EXPECTS TO RECEIVE THE CORRECT PRICE
# THE LAST CALCULATIONS WERE A TEST TO CALCULATE THE ACCURATE MODEL
# THIS IS A REAL SCENARIO SIMULATION

x_new = x_test[:10]            # INSERT DATA | HOUSE DATA
y_pred = model.predict(x_new)  # CALCULATE BOSTON HOUSE PRICE

# RESULT | 1.0 -> $1000,00

print(y_pred[0])

# SAVE MY MODEL

model.save('/content/drive/MyDrive/Colab Notebooks/regressor.h5')
model.save_weights('/content/drive/MyDrive/Colab Notebooks/regressor_weights.h5')

# RECREATE MY MODEL TO USE AGAIN IN OTHER PLACE

model = Sequential()
model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/regressor.h5')
model.load_weights('/content/drive/MyDrive/Colab Notebooks/regressor_weights.h5')

#new_sample = '/content/drive/MyDrive/file.ext'
#test_result = model.predict(new_sample)

#print(test_result)
