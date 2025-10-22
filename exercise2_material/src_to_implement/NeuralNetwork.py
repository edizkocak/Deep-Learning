import numpy as np
from copy import deepcopy

class NeuralNetwork:
	def __init__(self, optimizer, weights_initializer: float, bias_initializer: float):
		self.optimizer = optimizer
		self.weights_initializer = weights_initializer
		self.bias_initializer = bias_initializer
		
		self.loss = []
		self.layers = []
		self.data_layer = None
		self.loss_layer = None
		self.label = None
		
	def forward(self):
		input_tensor, label_tensor = self.data_layer.next()
		output = input_tensor
		for layer in self.layers:
			output = layer.forward(output)
			
		loss = self.loss_layer.forward(output, label_tensor)
		self.label = label_tensor
		return loss	
		
	def backward(self):
		error_tensor = self.loss_layer.backward(self.label)
		for layer in self.layers[::-1]:
			error_tensor = layer.backward(error_tensor)
		
	def append_layer(self, layer):
		if layer.trainable:
			layer.optimizer = deepcopy(self.optimizer)
			layer.initialize(self.weights_initializer, self.bias_initializer)
			
		self.layers.append(layer)
		
	def train(self, iterations):
		for _ in range(iterations):
			pred = self.forward()
			self.loss.append(pred)
			self.backward()
		
	def test(self, input_tensor):
		pred = input_tensor
		for layer in self.layers:
			pred = layer.forward(pred)
		return pred

