from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
import random

# load dataset
(x_train, y_train), (x_test, y_test) = load_data()

# # Explore the dataset
# # summarize loaded dataset
# print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
# print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
# # plot first few images
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# plt.show()

# fix the random seed
random.seed(1)

# reshape data to have a single channel
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# determine the shape of the input images
in_shape = x_train.shape[1:]

# determine the number of classes
n_classes = len(unique(y_train))
print(in_shape, n_classes)

# normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# define model
model = Sequential()

# # Convolution layer with 32 3 by 3 filters, the activation is relu
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=in_shape))

# # Max pooling layer with 2 by 2 pooling window.
# model.add(MaxPool2D(pool_size=(2, 2)))

# # Flatten layer
model.add(Flatten())

# # First hidden layer with 100 hidden nodes
model.add(Dense(units=100, activation='sigmoid'))

# # The output layer with 10 classes output.
# # Use the softmax activation function for classification
model.add(Dense(units=n_classes, activation='softmax'))

# define loss function and optimizer
# set the optimizer to 'sgd', then you may switch to 'adam'.
# use cross entropy as the loss for multi-class classification
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

# evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy on the test set: %.3f' % acc)


# make a prediction
# image = x_train[0]
# yhat = model.predict(asarray([image]))
# print('Predicted: class=%d' % argmax(yhat))