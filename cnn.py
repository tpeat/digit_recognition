#@author Tristan
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([i])
  plt.yticks([i])
fig
#plt.show()

#record model performance on validation dataset during training
#history = model.fit(..., validation_data=(valX, valY))

# load train and test dataset
def load_dataset():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  #need to reshape the data from a 3-dim to a 4-dim array
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  input_shape = (28, 28, 1)

  #there are 10 classes representing unique ints
  #we can make thema binary vector with 1 for index of the class value and 0 for all other values
  y_test = to_categorical(y_test)
  y_train = to_categorical(y_train)

#Prepare pixel data
#reshake grayscale pixel values between 0 and 1
def prep_pixels(train, test):
  #convert from ints to floats
  train_nrom = train.astype('float32')
  test_norm = test.astype('flaot32')

  #nromalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0

  #return normalized images
  return train_norm, test_norm


#define model
#two aspects: feature extraction front end using conv and pooling layers
#and a classifying backend to predict
from keras.models import Sequential
def define_model():
  from keras.models import Sequential
  from keras.layers import Conv2D
  model = Sequential()
  model.add(Conv2d(32,(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=(28,28,1)))
  model.add(MaxPooling2D((2,2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  mdel.add(Dense(10, activation='softmax'))

  #compile model
  opt = SGD(lr=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
  return model
