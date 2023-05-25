# - Course Name: Neural Networks and Deep Learning
# - This related to mini-project#1. This project is about implementation of CNN Neural Networks
# - Supervisor: Dr. Kalhor

# ----------------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools


# ============================================= INITIALIZATION =========================================================
BATCH_SIZE = 32
NUMBER_CLASS = 10
EPOCH = 60
SaveDirectory = os.path.join(os.getcwd(), 'saved_models')
ModelName = 'CNN_Trained_Model'

# ********************************************** SPLITTING THE DATA ****************************************************

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# **********************************************************************************************************************
# Converting vector to binary matrix

y_train = tensorflow.keras.utils.to_categorical(y_train, NUMBER_CLASS)
y_test = tensorflow.keras.utils.to_categorical(y_test, NUMBER_CLASS)


# ======================================================================================================================
# Building model...
# Structure of model...
MODEL = Sequential()

MODEL.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL.add(Activation('relu'))

MODEL.add(Conv2D(32, (3, 3)))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL.add(Conv2D(64, (3, 3)))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D(pool_size=(2, 2)))

MODEL.add(Flatten())
MODEL.add(Dense(512))
MODEL.add(Activation('relu'))
MODEL.add(Dense(NUMBER_CLASS))
MODEL.add(Activation('softmax'))

# **********************************************************************************************************************

# Initiating an Optimizer (RMSProp)...
OPTIMIZER = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)


# Training the model using defined RMSProp optimizer...
MODEL.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ============================================= ...FITTING THE MODEL... ================================================

results = MODEL.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    shuffle=True)
# =============================================== SAVING MODEL =========================================================

# Here, we are saving the model and weights
if not os.path.isdir(SaveDirectory):
    os.makedirs(SaveDirectory)
PATH = os.path.join(SaveDirectory, ModelName)
MODEL.save(PATH)
print('The trained model saved in this directory: %s ' % PATH)


# ================================================= EVALUATION =========================================================

# Here we are evaluating the trained model
EVAL = MODEL.evaluate(x_test, y_test, verbose=1)

# Plotting line graph of the results...
plt.figure()

plt.plot(results.epoch, np.array(results.history['val_acc']), label='Test Accuracy')
plt.plot(results.epoch, np.array(results.history['acc']), label='Train Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ************************************
plt.figure()

plt.plot(results.epoch, np.array(results.history['val_loss']), label='Test Loss')
plt.plot(results.epoch, np.array(results.history['loss']), label='Train Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Printing the results of evaluation...
print('The evaluated loss:', EVAL[0])
print('The evaluated accuracy:', EVAL[1])


