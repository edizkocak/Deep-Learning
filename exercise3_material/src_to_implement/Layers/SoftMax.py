from . import Base
import numpy as np

class SoftMax(Base.BaseLayer):
	def __init__(self):
		super().__init__()
		self.old_output = None
		
	def forward(self, input_tensor):
		shifted_input = input_tensor - np.max(input_tensor)
		exponented_input = np.exp(shifted_input)
		
		output_tensor = exponented_input / np.sum(exponented_input, axis=1, keepdims=True)
		self.old_output = output_tensor
		return output_tensor
		
	def backward(self, error_tensor):
		return self.old_output * (error_tensor - np.sum(error_tensor * self.old_output, axis=1, keepdims=True) )
		
		
