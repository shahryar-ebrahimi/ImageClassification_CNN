
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
EPOCH = 70
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
MODEL_WD = Sequential()

MODEL_WD.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(Conv2D(32, (3, 3)))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(MaxPooling2D(pool_size=(2, 2)))
MODEL_WD.add(Dropout(0.25))

MODEL_WD.add(Conv2D(32, (3, 3), padding='same'))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(Conv2D(32, (3, 3)))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(MaxPooling2D(pool_size=(2, 2)))
MODEL_WD.add(Dropout(0.25))


MODEL_WD.add(Conv2D(64, (3, 3), padding='same'))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(Conv2D(64, (3, 3)))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(MaxPooling2D(pool_size=(2, 2)))
MODEL_WD.add(Dropout(0.25))


MODEL_WD.add(Flatten())
MODEL_WD.add(Dense(512))
MODEL_WD.add(Activation('relu'))
MODEL_WD.add(Dense(NUMBER_CLASS))
MODEL_WD.add(Activation('softmax'))

# **************************************************
MODEL_WOD = Sequential()

MODEL_WOD.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(Conv2D(32, (3, 3)))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_WOD.add(Conv2D(32, (3, 3), padding='same'))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(Conv2D(32, (3, 3)))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(MaxPooling2D(pool_size=(2, 2)))


MODEL_WOD.add(Conv2D(64, (3, 3), padding='same'))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(Conv2D(64, (3, 3)))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(MaxPooling2D(pool_size=(2, 2)))


MODEL_WOD.add(Flatten())
MODEL_WOD.add(Dense(512))
MODEL_WOD.add(Activation('relu'))
MODEL_WOD.add(Dense(NUMBER_CLASS))
MODEL_WOD.add(Activation('softmax'))

# **********************************************************************************************************************

# Initiating an Optimizer (RMSProp)...
OPT_WD = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
OPT_WOD = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)


# Training the model using defined RMSProp optimizer...
MODEL_WD.compile(loss='categorical_crossentropy', optimizer=OPT_WD, metrics=['accuracy'])
MODEL_WOD.compile(loss='categorical_crossentropy', optimizer=OPT_WOD, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ============================================= ...FITTING THE MODEL... ================================================

results_WD = MODEL_WD.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    shuffle=True)

results_WOD = MODEL_WOD.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    shuffle=True)


# ================================================= EVALUATION =========================================================

# Here we are evaluating the trained model
EVAL1 = MODEL_WD.evaluate(x_test, y_test, verbose=1)
EVAL2 = MODEL_WOD.evaluate(x_test, y_test, verbose=1)

# Plotting line graph of the results...
plt.figure()

plt.plot(results_WD.epoch, np.array(results_WD.history['val_acc']), label='with Dropout')
plt.plot(results_WOD.epoch, np.array(results_WOD.history['val_acc']), label='without Dropout')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy for with/without Dropout')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_WD.epoch, np.array(results_WD.history['acc']), label='with Dropout')
plt.plot(results_WOD.epoch, np.array(results_WOD.history['acc']), label='without Dropout')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Accuracy for with/without Dropout')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_WD.epoch, np.array(results_WD.history['val_loss']), label='with Dropout')
plt.plot(results_WOD.epoch, np.array(results_WOD.history['val_loss']), label='without Dropout')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss for with/without Dropout')
plt.show()

# **************************************
plt.figure()

plt.plot(results_WD.epoch, np.array(results_WD.history['loss']), label='with Dropout')
plt.plot(results_WOD.epoch, np.array(results_WOD.history['loss']), label='without Dropout')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss for with/without Dropout')
plt.show()

# Printing the results of evaluation...
print('The evaluated loss for with Dropout:', EVAL1[0])
print('The evaluated accuracy for with Dropout:', EVAL1[1])

print('The evaluated loss for without Dropout:', EVAL2[0])
print('The evaluated accuracy for without Dropout:', EVAL2[1])


