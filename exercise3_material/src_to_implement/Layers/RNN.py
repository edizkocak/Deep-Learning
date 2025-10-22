import numpy as np
from . import Base

class RNN(Base.BaseLayer):
	def __init__(self,input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self._memorize = False
		self._gradient_weights = None

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.hidden_state = np.zeros(hidden_size)

	def forward(self,input_tensor):
		pass

	def backward(self,error_tensor):
		pass

	@property
	def memorize(self):
		return self._memorize	

	@property
	def gradient_weights(self):
		return self._gradient_weights

	@gradient_weights.setter
	def gradient_weights(self, weights):
		self._gradient_weights = weights


	@property
	def optimizer(self):
		return self._optimizer

	@property
	def optimizer(self, optimizer):
		self._optimizer = optimizer
