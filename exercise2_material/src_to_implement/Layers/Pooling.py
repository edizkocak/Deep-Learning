import numpy as np
from . import Base

class Pooling(Base.BaseLayer):
	def __init__(self, stride_shape, pooling_shape):
		super().__init__()
		self.stride_shape = stride_shape
		self.pooling_shape = pooling_shape
		
		# stores all max_values and their x,y coordinates in form (val, (x, y))
		self.max_val_coords = []
		
		self.old_input = None
		
	def forward(self, input_tensor):
		#height, width = input_tensor.shape[:2]
		#pooled_height = height // 
		
		self.old_input = input_tensor
		
		output = []
		for i in range(input_tensor.shape[0]):	# loop over batch dimension
			channels_pooled = []
			for j in range(input_tensor.shape[1]): # loop over channel dimension
				pooled_channel = []
				for m in range(input_tensor.shape[2] - self.pooling_shape[0] + 1):
					pooled_row = []
					for n in range(input_tensor.shape[3] - self.pooling_shape[1] + 1):
						kernel = input_tensor[i, j, m : m+self.pooling_shape[0], n : n + self.pooling_shape[1]]
						max_val = np.max(kernel)
						
						kernel_coords = np.argwhere(kernel == max_val)
						global_coords = kernel_coords + (m, n)
						self.max_val_coords.append((max_val, global_coords))
						
						pooled_row.append(max_val)
					pooled_channel.append(np.array(pooled_row))
				channels_pooled.append(np.array(pooled_channel))
			output.append(np.array(channels_pooled))
			
		output = np.array(output)
		return output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
					
					
	def backward(self, error_tensor):
		input_tensor = self.old_input
		
		# get old coordinate positions from forward pass
		val_coords = self.max_val_coords
		
		output = np.zeros(input_tensor.shape)
		
		for i in range(error_tensor.shape[0]): # loop over batch dimension
			for j in range(error_tensor.shape[1]): #loop over channel dimension
				for m in range(error_tensor.shape[2]):
					for n in range(error_tensor.shape[3]):
						y = m * self.stride_shape[0]
						x = n * self.stride_shape[1]
						
						kernel = input_tensor[i, j, y : y + self.pooling_shape[0], x : x + self.pooling_shape[1]]
						max_val = np.max(kernel)
						
						error_pooled = (kernel == max_val) * error_tensor[i, j, m, n]
						# print(error_pooled.shape)
						output[i, j, y : y + self.pooling_shape[0], x : x + self.pooling_shape[1]] += error_pooled

		output = np.array(output)
		
		return output
		
						
						
						
