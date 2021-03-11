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
  from keras.utils import to_categorical
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  #need to reshape the data from a 3-dim to a 4-dim array
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  input_shape = (28, 28, 1)

  #there are 10 classes representing unique ints
  #we can make thema binary vector with 1 for index of the class value and 0 for all other values
  y_test = to_categorical(y_test)
  y_train = to_categorical(y_train)
  print(y_train)
  
  return x_train, y_train, x_test, y_test

#Prepare pixel data
#reshake grayscale pixel values between 0 and 1
def prep_pixels(train, test):
  #convert from ints to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')

  #nromalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  
  plt.imshow(train_norm[0],cmap='gray', interpolation='none')
  print(train_norm[0])
  #return normalized images
  return train_norm, test_norm


#define model
#two aspects: feature extraction front end using conv and pooling layers
#and a classifying backend to predict
from keras.models import Sequential
def define_model():
  from keras.models import Sequential
  from keras.layers import Conv2D
  from keras.layers import MaxPooling2D
  from keras.layers import Dense
  from keras.layers import Flatten
  from keras.optimizers import SGD
  model = Sequential()
  model.add(Conv2D(32,(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=(28,28,1)))
  model.add(MaxPooling2D((2,2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))

  #compile model
  opt = SGD(lr=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
  return model

#evaluate a modl using k-fold cross-validaiton
def evaluate_model(dataX, dataY, n_folds=5):
  from sklearn.model_selection import KFold
  scores, histories = list(), list()
  #prepare cross validation
  kfold = KFold(n_folds, shuffle=True,random_state=1)
  #enumerate splits
  for train_ix, test_ix in kfold.split(dataX):
    #define model
    model = define_model()
    #select number of rows for train adn test
    trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
    #fit model
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX,testY),verbose=0)
    #evaludate model
    print(history)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('>%.3f'%(acc *100.0))
    #store scores
    scores.append(acc)
    histories.append(history)
  return scores, histories

#plot diagnostic learning curves
def summarize_diagnostics(histories):
  for i in range(len(histories)):
    #plot loss
    plt.subsplot(2,1,1)
    
    plt.title('Cross Entropy Loss')
    plt.plot(histories[i].history['loss'], color='blue', label='train')

  plt.show()
  
#run the test ahrness for evaluating a model

if __name__ == '__main__':
  #load dataset
  trainX, trainY, testX, testY = load_dataset()
  #prepare pixel data
  trainX, testX = prep_pixels(trainX,testX)
#  #evaluate model
#  scores, histories = evaluate_model(trainX, trainY)
#  #learning curves
#  summarize_diagnostics(histories)
