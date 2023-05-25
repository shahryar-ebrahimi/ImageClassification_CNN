
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
# Activation Function : Relu

MODEL_RELU = Sequential()

MODEL_RELU.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_RELU.add(Activation('relu'))

MODEL_RELU.add(Conv2D(32, (3, 3)))
MODEL_RELU.add(Activation('relu'))
MODEL_RELU.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_RELU.add(Conv2D(64, (3, 3)))
MODEL_RELU.add(Activation('relu'))
MODEL_RELU.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_RELU.add(Flatten())
MODEL_RELU.add(Dense(512))
MODEL_RELU.add(Activation('relu'))
MODEL_RELU.add(Dense(NUMBER_CLASS))
MODEL_RELU.add(Activation('softmax'))

# =============================================
# Activation Function : Sigmoid

MODEL_SIG = Sequential()

MODEL_SIG.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_SIG.add(Activation('sigmoid'))

MODEL_SIG.add(Conv2D(32, (3, 3)))
MODEL_SIG.add(Activation('sigmoid'))
MODEL_SIG.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_SIG.add(Conv2D(64, (3, 3)))
MODEL_SIG.add(Activation('sigmoid'))
MODEL_SIG.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_SIG.add(Flatten())
MODEL_SIG.add(Dense(512))
MODEL_SIG.add(Activation('sigmoid'))
MODEL_SIG.add(Dense(NUMBER_CLASS))
MODEL_SIG.add(Activation('softmax'))

# =============================================
# Activation Function : ELU

MODEL_ELU = Sequential()

MODEL_ELU.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
MODEL_ELU.add(Activation('elu'))

MODEL_ELU.add(Conv2D(32, (3, 3)))
MODEL_ELU.add(Activation('elu'))
MODEL_ELU.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_ELU.add(Conv2D(64, (3, 3)))
MODEL_ELU.add(Activation('elu'))
MODEL_ELU.add(MaxPooling2D(pool_size=(2, 2)))

MODEL_ELU.add(Flatten())
MODEL_ELU.add(Dense(512))
MODEL_ELU.add(Activation('elu'))
MODEL_ELU.add(Dense(NUMBER_CLASS))
MODEL_ELU.add(Activation('softmax'))


# **********************************************************************************************************************

# Initiating an Optimizer (RMSProp)...
OPT_RELU = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
OPT_SIG = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
OPT_ELU = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)


# Training the models using defined RMSProp optimizer...
MODEL_RELU.compile(loss='categorical_crossentropy', optimizer=OPT_RELU, metrics=['accuracy'])
MODEL_SIG.compile(loss='categorical_crossentropy', optimizer=OPT_SIG, metrics=['accuracy'])
MODEL_ELU.compile(loss='categorical_crossentropy', optimizer=OPT_ELU, metrics=['accuracy'])


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# ============================================= ...FITTING THE MODEL... ================================================

results_RELU = MODEL_RELU.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                              shuffle=True)

results_SIG = MODEL_SIG.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                            shuffle=True)

results_ELU = MODEL_ELU.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                            shuffle=True)

# ================================================= EVALUATION =========================================================

# Here we are evaluating the trained model
EVAL1 = MODEL_RELU.evaluate(x_test, y_test, verbose=1)
EVAL2 = MODEL_SIG.evaluate(x_test, y_test, verbose=1)
EVAL3 = MODEL_ELU.evaluate(x_test, y_test, verbose=1)

# Plotting line graph of the results...
plt.figure()

plt.plot(results_RELU.epoch, np.array(results_RELU.history['val_acc']), label='Relu')
plt.plot(results_SIG.epoch, np.array(results_SIG.history['val_acc']), label='Sigmoid')
plt.plot(results_ELU.epoch, np.array(results_ELU.history['val_acc']), label='ELU')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy for 3 Different Activation Functions')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_RELU.epoch, np.array(results_RELU.history['acc']), label='Relu')
plt.plot(results_SIG.epoch, np.array(results_SIG.history['acc']), label='Sigmoid')
plt.plot(results_ELU.epoch, np.array(results_ELU.history['acc']), label='ELU')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train Accuracy for 3 Different Activation Functions')
plt.show()

# ***************************************
plt.figure()

plt.plot(results_RELU.epoch, np.array(results_RELU.history['val_loss']), label='Relu')
plt.plot(results_SIG.epoch, np.array(results_SIG.history['val_loss']), label='Sigmoid')
plt.plot(results_ELU.epoch, np.array(results_ELU.history['val_loss']), label='ELU')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss for 3 Different Activation Functions')
plt.show()

# **************************************
plt.figure()

plt.plot(results_RELU.epoch, np.array(results_RELU.history['loss']), label='Relu')
plt.plot(results_SIG.epoch, np.array(results_SIG.history['loss']), label='Sigmoid')
plt.plot(results_ELU.epoch, np.array(results_ELU.history['loss']), label='ELU')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss for 3 Different Activation Functions')
plt.show()

# ======================================================================================================================
# Printing the results of evaluation...
print('The evaluated loss:', EVAL1[0])
print('The evaluated accuracy:', EVAL1[1])

print('The evaluated loss:', EVAL2[0])
print('The evaluated accuracy:', EVAL2[1])

print('The evaluated loss:', EVAL3[0])
print('The evaluated accuracy:', EVAL3[1])


