import numpy as np

class CrossEntropyLoss:
	def __init__(self):
		self.eps = np.finfo(float).eps
		self.old_prediction = None
		
	def forward(self, prediction_tensor, label_tensor):
		self.old_prediction = prediction_tensor + self.eps
		loss = -np.sum(np.log(self.old_prediction) * label_tensor)
		return loss
		
	def backward(self, label_tensor):
		gradient =  -1.0 * (label_tensor / (self.old_prediction) )
		return gradient
		
