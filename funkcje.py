import numpy as np

def centering(z):
  return z - np.average(z)
  
def normalize(z):
  return (z-min(z))/(max(z)-min(z))
  
def standarize(z):
  return (z-np.average(z))/np.std(z)
  
def ReLU(z):
  return max(0,z)
  
def sigmoid(z):
  e=2.71
  sigmoid_computed = 1/(1+pow(e, -z))
  return sigmoid_computed
  
def tanh(z):
  tanh_computed = np.tanh(z)
  return tanh_computed
  
def elu(z):
  e=2.71
  alpha = 1
  if (z>=0):
    sigmoid_computed = z
  if (z<0):
    sigmoid_computed = alpha*pow(e, z) - 1
  return sigmoid_computed
