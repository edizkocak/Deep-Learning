import numpy as np
from . import Base

class BatchNormalization(Base.BaseLayer):
	def __init__(channels: int):
		super().__init__()
		self.channels = channels
	def forward(self,input_tensor):
		pass
	def backward(self,error_tensor):
		pass

	def initialize(self):
		pass
