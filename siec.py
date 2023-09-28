import numpy as np
from warstwa import warstwa
import json
class siec:
  def __init__(self, input_neurons, input_activation, hidden_quantity, hidden_neurons, hidden_activation, output_neurons, output_activation):
    self.warstwy = []
    self.warstwy.append(warstwa(input_neurons, input_activation))
    for i in range(hidden_quantity):
      self.warstwy.append(warstwa(hidden_neurons, hidden_activation))
    self.warstwy.append(warstwa(output_neurons, output_activation))

  def output(self, input_data):
    self.inputdata_lenght,self.inputdata_neurons = input_data.shape
    self.output = []
    self.output.append(self.warstwy[0].output(input_data))
    for i in range(len(self.warstwy)-1):
      self.output.append(self.warstwy[i+1].output(self.output[i]))
    return self.output[len(self.warstwy)-1]
    
  def init_from_json(self, json_file):
    f = open(json_file)
    data = json.load(f)
    self.init_json_activation = []
    self.init_json_neurons = []
    for i in data["warstwy"]:
      self.init_json_activation.append(i["funkcja_aktywacji"])
      self.init_json_neurons.append(i["liczba_neuronów"])
    f.close()
    self.warstwy = []
    for i in range(len(self.init_json_activation)):
      self.warstwy.append(warstwa(self.init_json_neurons[i], self.init_json_activation[i]))
