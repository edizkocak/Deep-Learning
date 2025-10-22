import numpy as np

class Sgd:
	def __init__(self, learning_rate: float):
		self.learning_rate = learning_rate
		
	
	def calculate_update(self, weight_tensor, gradient_tensor):
		new_weights = weight_tensor - self.learning_rate * gradient_tensor
		return new_weights
		
class SgdWithMomentum:
	def __init__(self, learning_rate: float, momentum_rate: float):
		self.learning_rate = learning_rate
		self.momentum_rate = momentum_rate
		
		self.prev_v = 0.0
		
	def calculate_update(self, weight_tensor, gradient_tensor):
		v = self.learning_rate * gradient_tensor + self.momentum_rate * self.prev_v
		self.prev_v = v
		new_weights = weight_tensor - v
		
		return new_weights
		
		
class Adam:
	def __init__(self, learning_rate: float, mu: float, rho: float):
		self.learning_rate = learning_rate
		self.mu = mu
		self.rho = rho
		
		self.prev_v = 0.0
		self.prev_r = 0.0
		
		self.k = 1
		
		self.eps = np.finfo(float).eps
		
	def calculate_update(self, weight_tensor, gradient_tensor):
		v = self.mu * self.prev_v + (1.0 - self.mu) * gradient_tensor
		r = self.rho * self.prev_r + (1.0 - self.rho) * (gradient_tensor * gradient_tensor)
		
		self.prev_v = v
		self.prev_r = r
		
		bias_v = v / (1.0 - self.mu ** self.k)
		bias_r = r / (1.0 - self.rho ** self.k)
		
		weights = weight_tensor - self.learning_rate * (bias_v / (np.sqrt(bias_r) + self.eps) )
		
		self.k += 1
		
		return weights
		
	
	

	
