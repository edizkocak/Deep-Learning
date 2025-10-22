#internal imports
from . import Base
import numpy as np

class ReLU(Base.BaseLayer):
	def __init__(self):
		super(ReLU, self).__init__()
		self.old_input = None
		
	def forward(self, input_tensor):
		self.old_input = input_tensor
		new_input = np.maximum(0, input_tensor)
		return new_input 
		
	def backward(self, error_tensor):
		return np.where(self.old_input > 0, error_tensor, 0)
