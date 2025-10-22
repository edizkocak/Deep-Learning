import numpy as np

class Constant:
	def __init__(self, constant_value=0.1):
		self.constant_value = constant_value
		self.fan_in = None
		self.fan_out = None
	
	def initialize(self, weights_shape, fan_in, fan_out):
		self.fan_in = fan_in
		self.fan_out = fan_out
		
		weights = np.full(weights_shape, self.constant_value)
		return weights
	
class UniformRandom:
	def __init__(self):
		pass
	
	def initialize(self, weights_shape, fan_in, fan_out):
		self.fan_in = fan_in
		self.fan_out = fan_out
		
		weights = np.random.rand(*weights_shape)
		return weights
	
class Xavier:
	def __init__(self):
		pass
	
	def initialize(self, weights_shape, fan_in, fan_out):
		self.fan_in = fan_in
		self.fan_out = fan_out
		
		value = np.sqrt(2.0 / (fan_out + fan_in) )
		weights = np.random.normal(0, value, weights_shape)
		return weights
		
class He:
	def __init__(self):
		pass
	
	def initialize(self, weights_shape, fan_in, fan_out):
		self.fan_in = fan_in
		self.fan_out = fan_out
		
		value = np.sqrt(2.0 / fan_in)
		weights = np.random.normal(0, value, weights_shape)
		return weights
	

