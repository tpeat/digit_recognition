#@author Tristan

#get dataset
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.laod_data()

#need to reshape the data from a 3-dim to a 4-dim array
x_train = x_train.reshae(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
