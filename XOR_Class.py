import numpy as np
from numpy import random, dot, array

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])



labels = np.array([
    [0],
    [1],
    [1],
    [0]])



# This is the activation function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))



# This is the derivative of the activation function. 
def dsigmoid_dx(a):
    return a * (1 - a)



class Layer(object):
    
    # This function initializes all of the values that will be used repeatedly in the class.
    def __init__(self, inputs, outputs, learning_rate = 0.1):
        
        self.weights = np.random.uniform(size = (outputs, inputs))
        self.bias = np.zeros(outputs)
        self.learning_rate = learning_rate
        
        

    # The forward function takes the weighted sum of the weights and inputs at each neuron. Then,
    # the weighted sum is passed through the Sigmoid activation function.
    def forward(self, inputs):
        self.last_inputs = inputs
        w_sum_bias = (np.dot(self.weights, inputs) + self.bias)
        activation = sigmoid(w_sum_bias)
        self.last_activation = activation
        return activation
    
    
    
    # The backward function takes the derivative of the output from the forward function. Then, it
    # updates the weights and biases. 
    def backward(self, grad_out):
        grad = grad_out * dsigmoid_dx(self.last_activation)
        self.bias = self.bias - self.learning_rate * grad
        self.weights = self.weights - self.learning_rate * np.array(np.matrix(grad).T @ np.matrix(self.last_inputs))
        grad = np.matrix(grad) @ self.weights
        grad = np.array(grad)[0, :]
        return grad



# This is setting up the two layers of neurons. 
first_neuron_layer = Layer(2, 2, learning_rate = 0.1)
second_neuron_layer = Layer(2, 1, learning_rate = 0.1)



# This loop is the training loop
for i in range(40000):
    
    k = random.randint(len(inputs))
    x = inputs[k]
    desired_response = labels[k]
    
    
    
    # This block is running through forward and backpropagation once
    hidden_layer = first_neuron_layer.forward(x)
    output_layer = second_neuron_layer.forward(hidden_layer)
    loss = (desired_response - output_layer) ** 2
    dloss_doutput_layer = - 2 * (desired_response - output_layer)
    back_output = second_neuron_layer.backward(dloss_doutput_layer)
    back_hidden = first_neuron_layer.backward(back_output)
    
    
    
# This is printing off the results
print(" Network output ")
for i in range(len(inputs)):
    x = inputs[i]
    response = labels[i]
    final_hidden = first_neuron_layer.forward(x)
    final_output = second_neuron_layer.forward(final_hidden)
    print("{} -> {} = {}".format(x, response, final_output))


