import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print ("MNIST datset shape:")
print('x_train: ' + str(x_train.shape))
print('y_train: ' + str(y_train.shape))
print('x_test: ' + str(x_test.shape))
print('y_test: ' + str(y_test.shape))


#euclidean distance
def Euclidean_distance(row1, row2):
  distance = 0
  for i in range(len(row1) -1):
    distance +=(row1[i] - row2[i])**2
  return sqrt(distance)

def Get_Neighbors(train, test_row, num):
  distance = list() 
  data = []
  for i in train:
    dist = Euclidean_distance(test_row, i)
    distance.append(i)
    distance = np.array(distance)
    data = np.array(data)

  #fixing the index in ascending order
  index_dist = distance.argsort()

  #arraning data according to index
  data = data[index_dist]

  #slicing k value from number of data
  neighbors = data[:num]
  
  return neighbors
  
def predict_classification(train, test_row, num):
  Neighbors = Get_Neighbors(train, test_row, num)
  Classes = []
  for i in Neighbors:
    Classes.append(i[-1])
  prediction = max(Classes, key= Classes,count)
  return prediction

def accuracy(y_true, y_pred):
  n_correct = 0
  for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
      n_correct += 1
  acc = n_correct/len(y_true)
  return acc
