import sys
import funkcje
import dane
from warstwa import warstwa
import numpy as np
from siec import siec

def preprocessing(data):
  centered = funkcje.centering(data)
  normalized = funkcje.normalize(centered)
  standarized = funkcje.standarize(normalized)
  return standarized

def activation(data):
  data_ReLU = list(map(funkcje.ReLU, data))
  data_elu = list(map(funkcje.elu, data))
  data_tanh = list(map(funkcje.tanh, data))
  data_sigmoid = list(map(funkcje.sigmoid, data))

weights = np.array([[1,2],[3,4]])
input_data = np.array([[1],[2],[3]])
weights2 = np.array([[1,2,3],[4,5,6]])
initial_loss = np.array([[1],[2]])

dataset = dane.dane(sys.argv[2])
preprocessed_height = np.array([preprocessing(dataset[:,1])])
preprocessed_weight = np.array([preprocessing(dataset[:,2])])

siec = siec(1, funkcje.ReLU, 1, 10, funkcje.tanh, 1, funkcje.ReLU)
output = siec.output(preprocessed_height.T)
print('input lenght: ',siec.inputdata_lenght)
print('input neurons quantity: ',siec.inputdata_neurons)
print('output from last layer: ',output)
siec.init_from_json(sys.argv[1])
print('New network init from json:')
print(siec.init_json_activation)
print(siec.init_json_neurons)
