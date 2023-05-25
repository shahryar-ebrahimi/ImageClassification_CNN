
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
EPOCH = 10

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
# Optimizer: Gradient Descent

MODEL_GD = Sequential()

MODEL_GD.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_GD.add(Activation('elu'))

MODEL_GD.add(Conv2D(32, (3, 3)))
MODEL_GD.add(Activation('elu'))
MODEL_GD.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_GD.add(Conv2D(64, (3, 3)))
MODEL_GD.add(Activation('elu'))
MODEL_GD.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_GD.add(Flatten())
MODEL_GD.add(Dense(512))
MODEL_GD.add(Activation('elu'))
MODEL_GD.add(Dense(NUMBER_CLASS))
MODEL_GD.add(Activation('softmax'))

# =============================================
# Optimizer: Adam

MODEL_ADAM = Sequential()

MODEL_ADAM.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_ADAM.add(Activation('elu'))

MODEL_ADAM.add(Conv2D(32, (3, 3)))
MODEL_ADAM.add(Activation('elu'))
MODEL_ADAM.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_ADAM.add(Conv2D(64, (3, 3)))
MODEL_ADAM.add(Activation('elu'))
MODEL_ADAM.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_ADAM.add(Flatten())
MODEL_ADAM.add(Dense(512))
MODEL_ADAM.add(Activation('elu'))
MODEL_ADAM.add(Dense(NUMBER_CLASS))
MODEL_ADAM.add(Activation('softmax'))


# **********************************************************************************************************************

# Initiating Optimizers...
OPT_GD = tensorflow.keras.optimizers.SGD(lr=0.0001, decay=1e-6)
OPT_ADAM = tensorflow.keras.optimizers.Adam(lr=0.0001, decay=1e-6)


# Training the models using defined RMSProp optimizer...
MODEL_GD.compile(loss='categorical_crossentropy', optimizer=OPT_GD, metrics=['accuracy'])
MODEL_ADAM.compile(loss='categorical_crossentropy', optimizer=OPT_ADAM, metrics=['accuracy'])


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ============================================= ...FITTING THE MODEL... ================================================

results_GD = MODEL_GD.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                            shuffle=True)

results_ADAM = MODEL_ADAM.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                             shuffle=True)
# ================================================= EVALUATION =========================================================

# Here we are evaluating the trained model
EVAL1 = MODEL_GD.evaluate(x_test, y_test, verbose=1)
EVAL2 = MODEL_ADAM.evaluate(x_test, y_test, verbose=1)

# Plotting line graph of the results...
plt.figure()

plt.plot(results_GD.epoch, np.array(results_GD.history['val_acc']), label='Gradient Descent')
plt.plot(results_ADAM.epoch, np.array(results_ADAM.history['val_acc']), label='Adam')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy for 2 Different Optimizers')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_GD.epoch, np.array(results_GD.history['acc']), label='Gradient Descent')
plt.plot(results_ADAM.epoch, np.array(results_ADAM.history['acc']), label='Adam')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Accuracy for 2 Different Optimizers')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_GD.epoch, np.array(results_GD.history['val_loss']), label='Gradient Descent')
plt.plot(results_ADAM.epoch, np.array(results_ADAM.history['val_loss']), label='Adam')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss for 2 Different Optimizers')
plt.show()

# **************************************
plt.figure()

plt.plot(results_GD.epoch, np.array(results_GD.history['loss']), label='Gradient Descent')
plt.plot(results_ADAM.epoch, np.array(results_ADAM.history['loss']), label='Adam')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss for 2 Different Optimizers')
plt.show()

# ======================================================================================================================
# Printing the results of evaluation...
print('The evaluated loss for GD:', EVAL1[0])
print('The evaluated accuracy for GD:', EVAL1[1])

print('The evaluated loss for Adam:', EVAL2[0])
print('The evaluated accuracy for Adam:', EVAL2[1])