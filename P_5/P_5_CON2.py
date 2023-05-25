
from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
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
MODEL_C1 = Sequential()

MODEL_C1.add(Conv2D(32, (7, 7), padding='same', input_shape=x_train.shape[1:]))
MODEL_C1.add(Activation('relu'))

MODEL_C1.add(Conv2D(32, (7, 7)))
MODEL_C1.add(Activation('relu'))
MODEL_C1.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_C1.add(Conv2D(64, (7, 7)))
MODEL_C1.add(Activation('relu'))
MODEL_C1.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_C1.add(Flatten())
MODEL_C1.add(Dense(512))
MODEL_C1.add(Activation('relu'))
MODEL_C1.add(Dense(NUMBER_CLASS))
MODEL_C1.add(Activation('softmax'))

# *********************************************

MODEL_C2 = Sequential()

MODEL_C2.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(Conv2D(32, (3, 3)))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_C2.add(Conv2D(32, (3, 3), padding='same'))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(Conv2D(32, (3, 3)))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(MaxPooling2D(pool_size=(2, 2)))


MODEL_C2.add(Conv2D(64, (3, 3), padding='same'))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(Conv2D(64, (3, 3)))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(MaxPooling2D(pool_size=(2, 2)))


MODEL_C2.add(Flatten())
MODEL_C2.add(Dense(512))
MODEL_C2.add(Activation('relu'))
MODEL_C2.add(Dense(NUMBER_CLASS))
MODEL_C2.add(Activation('softmax'))


# **********************************************************************************************************************

# Initiating an Optimizer (RMSProp)...
OPT_C1 = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
OPT_C2 = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)


# Training the model using defined RMSProp optimizer...
MODEL_C1.compile(loss='categorical_crossentropy', optimizer=OPT_C1, metrics=['accuracy'])
MODEL_C2.compile(loss='categorical_crossentropy', optimizer=OPT_C2, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ============================================= ...FITTING THE MODEL... ================================================

results_C1 = MODEL_C1.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    shuffle=True)

results_C2 = MODEL_C2.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    shuffle=True)


# ================================================= EVALUATION =========================================================

# Here we are evaluating the trained model
EVAL1 = MODEL_C1.evaluate(x_test, y_test, verbose=1)
EVAL2 = MODEL_C2.evaluate(x_test, y_test, verbose=1)

# Plotting line graph of the results...
plt.figure()

plt.plot(results_C1.epoch, np.array(results_C1.history['val_acc']), label='with one Conv')
plt.plot(results_C2.epoch, np.array(results_C2.history['val_acc']), label='with two Conv')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy for 2 types of model')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_C1.epoch, np.array(results_C1.history['acc']), label='with one Conv')
plt.plot(results_C2.epoch, np.array(results_C2.history['acc']), label='with two Conv')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Accuracy for 2 types of model')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_C1.epoch, np.array(results_C1.history['val_loss']), label='with one Conv')
plt.plot(results_C2.epoch, np.array(results_C2.history['val_loss']), label='with two Conv')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss for 2 types of model')
plt.show()

# **************************************
plt.figure()

plt.plot(results_C1.epoch, np.array(results_C1.history['loss']), label='with one Conv')
plt.plot(results_C2.epoch, np.array(results_C2.history['loss']), label='with two Conv')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss for 2 types of model')
plt.show()

# Printing the results of evaluation...
print('The evaluated loss for one Conv:', EVAL1[0])
print('The evaluated accuracy for one Conv:', EVAL1[1])

print('The evaluated loss for two Conv:', EVAL2[0])
print('The evaluated accuracy for two Conv:', EVAL2[1])


