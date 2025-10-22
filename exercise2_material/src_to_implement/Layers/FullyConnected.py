import numpy as np
from . import Base

class FullyConnected(Base.BaseLayer):
	def __init__(self, input_size, output_size):
		super(FullyConnected, self).__init__()
		self.trainable = True
		
		self.input_size = input_size
		self.output_size = output_size
		
		self._optimizer = None
		self._gradient_weights = None
		
		self.weights_initializer = None
		self.bias_initializer = None
		
		self.old_input = None
		
		self.weights = np.random.rand(input_size + 1, output_size)
		
	@property
	def gradient_weights(self):
		return self._gradient_weights
	
	@gradient_weights.setter	
	def gradient_weights(self, gradients):
		self._gradient_weights = gradients
		
	@property	
	def optimizer(self):
		return self._optimizer
		
	@optimizer.setter	
	def optimizer(self, optimizer):
		self._optimizer = optimizer
		
	def initialize(self, weights_initializer, bias_initializer):
		new_weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
		new_bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
		self.weights = np.vstack((new_weights, new_bias))
		
	def forward(self, input_tensor):	
		batch_size = len(input_tensor)
		
		expand_ones = np.ones(batch_size)
		expand_ones = np.expand_dims(expand_ones, axis=1)
		expanded_input = np.concatenate((input_tensor, expand_ones), axis=1)
		
		self.old_input = np.copy(expanded_input)
		out_tensor = np.dot(expanded_input, self.weights)
		
		return out_tensor
	
	def backward(self, error_tensor):
		prev_error = np.dot(error_tensor, self.weights.T)
		gradient_resp_w = np.dot(self.old_input.T, error_tensor)
		self.gradient_weights = gradient_resp_w	
		
		if(self.optimizer):
			self.weights = self.optimizer.calculate_update(self.weights, gradient_resp_w)
		
		return prev_error[:, :-1]
		
		
		
		

