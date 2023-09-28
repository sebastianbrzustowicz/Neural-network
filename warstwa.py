import numpy as np
import funkcje
class warstwa:
  def __init__(self, neuron_num, activation): #neuron_num = output_shape
    self.neuron_num = neuron_num
    self.activation = activation
    self.parameters_init = 1

  def output(self, input_data):
    if self.parameters_init == 1:
      self.input_data = input_data
      self.input_shape = max(len(input_data),len(input_data.T)) #reshaping input data
      input_data = np.reshape(input_data,(self.input_shape,1))
      self.weights = np.random.uniform(low=-0.3, high=0.3, size=(self.neuron_num,self.input_shape))
      self.biases = np.random.uniform(low=-0.3, high=0.3, size=(self.neuron_num,1))
      self.parameters_init = 0
    
    self.output_value = np.empty(shape=[self.neuron_num, 1])
    #self.L2 = np.empty(shape=[self.neuron_num, 1])
    self.q = np.empty(shape=[self.neuron_num, self.input_shape])
    self.q = np.matmul(self.weights,input_data)+self.biases
    for neuron in range(self.neuron_num):
      self.output_value[neuron] = self.activation(self.q[neuron])
    return self.output_value
    
  def gradient(self, loss, learning_rate):
    self.loss = np.array(loss)
    self.weights_back = np.empty(shape=[self.neuron_num,self.input_shape])
    self.x_back = np.empty(shape=[self.input_shape,1])
    self.q_back = np.empty(shape=[self.neuron_num, self.input_shape])
    self.gradient_activation = np.empty(shape=[self.neuron_num,1])
    
    for neuron in range(self.neuron_num):
      if self.activation == funkcje.ReLU:
        if self.output_value[neuron]>0:
          self.gradient_activation[neuron] = 1
        else:
          self.gradient_activation[neuron] = 0
      if self.activation == funkcje.tanh:
        self.gradient_activation[neuron] = 1 - pow(np.tanh(self.loss[neuron]),2)
      if self.activation == funkcje.sigmoid:
        e=2.71
        sigmoid_computed = 1/(1+pow(e, -self.loss[neuron]))
        self.gradient_activation[neuron] = sigmoid_computed*(1-sigmoid_computed)
      if self.activation == funkcje.elu:
        if self.output_value[neuron]>0:
          self.gradient_activation[neuron] = 1
        else:
          e=2.71
          alpha=1
          self.gradient_activation[neuron] = alpha*pow(e, self.loss[neuron])
       
    self.weights_back = np.matmul(np.multiply(self.loss,self.gradient_activation),self.input_data.T)
    self.x_back = np.matmul(self.weights.T,np.multiply(self.loss,self.gradient_activation))
    self.biases_back = np.multiply(self.loss,self.gradient_activation)
    
    #weights update
    self.learning_rate = learning_rate
    self.weights = self.weights - self.learning_rate*self.weights_back
    self.biases = self.biases - self.learning_rate*self.biases_back
    
    #self.weights_back = 2*np.matmul(self.q,self.input_data.T)
    #self.x_back = 2*np.matmul(self.weights.T,self.q)
    #self.q_back = 2*self.q
    return self.weights_back
