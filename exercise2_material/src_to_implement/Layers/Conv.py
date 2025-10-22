import numpy as np
from scipy.signal import correlate
from scipy.signal import correlate2d
from scipy.signal import convolve

import copy

from . import Base


class Conv(Base.BaseLayer):
	def __init__(self, stride_shape, convolution_shape, num_kernels: int):
		self.stride_shape = stride_shape
		self.convolution_shape = convolution_shape
		self.num_kernels = num_kernels
		
		self.trainable = True
		
		# initialize properties
		self._gradient_weights = None
		self._gradient_bias = None
		
		self._optimizer = None
		self._optimizer_weights = None
		self._optimizer_bias = None
		
		self.weights = None
		self.bias = None
		
		if len(convolution_shape) == 2:
			self.one_dim = True
			self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1])
			self.bias = np.random.rand(num_kernels)
			
		else:
			self.one_dim = False
			self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
			self.bias = np.random.rand(num_kernels)
			
		self.old_input = None
		self.input = None
		self.input_tensor_shape = None
			
		
	@property
	def gradient_weights(self):
		return self._gradient_weights
		
	#@gradient_weights.setter
	#def gradient_weights(self, value):
#		self._gradient_weights = value
		
	@property
	def gradient_bias(self):
		return self._gradient_bias
		
	#@gradient_bias.setter
	#def gradient_bias(self, value):
	#self._gradient_bias = value
		
	@property
	def optimizer(self):
		return self._optimizer_opt
		
	@optimizer.setter
	def optimizer(self, optimizer):
		self._optimizer_opt = optimizer
		self._optimizer_weights = copy.deepcopy(optimizer)
		self._optimizer_bias = copy.deepcopy(optimizer)
		
		
	def initialize(self, weights_initializer, bias_initializer):
		input_channels = self.convolution_shape[0]
		if self.one_dim:
			fan_in = input_channels
			fan_out = 1
			weights_shape = self.weights.shape
			self.weights = weights_initializer(weights_shape, fan_in, fan_out)
			self.bias = bias_initializer.initialize(self.num_kernels, fan_in, fan_out)
			return
		
		kernel_height = self.convolution_shape[1]
		kernel_width = self.convolution_shape[2]
		fan_in = input_channels * kernel_height * kernel_width
		fan_out = kernel_width * kernel_height * self.num_kernels
		
		self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
		self.bias = bias_initializer.initialize(self.num_kernels, fan_in, fan_out)
		
	def forward(self, input_tensor):
		kernel = self.weights
		result = None
		
		self.old_input = input_tensor
		self.input = input_tensor
		self.input_tensor_shape = input_tensor.shape
		
		if self.one_dim:
			result = np.zeros((input_tensor.shape[0], self.num_kernels, input_tensor.shape[2]))
			pad_width = kernel.shape[2] // 2  # For odd-sized kernels
			
			# Adjust padding for even-sized kernels
			if kernel.shape[-1] % 2 == 0:
				pad_width_left = max(0, kernel.shape[-1] // 2)
				pad_width_right = max(0, kernel.shape[-1] // 2 - 1)
			else:
				pad_width_left = max(0, kernel.shape[-1] // 2)  # For odd-sized kernels
				pad_width_right = max(0, kernel.shape[-1] // 2)  # For odd-sized kernels
				
			pad_width = [(0, 0), (pad_width_left, pad_width_right)]
			
			for i in range(input_tensor.shape[0]): # Loop through batch dimension
				for j in range(kernel.shape[0]): # Loop through output channels of kernel
					# Pad only along the second dimension of the input tensor
					padded = np.pad(input_tensor[i], pad_width, mode='constant', constant_values=0)
					result[i, :, :] = correlate(padded, kernel[j], mode='valid')			
					#result[i, :, :] += self.bias[j]

			output = result[:, : , ::self.stride_shape[0]]		
			output += self.bias.reshape((1, self.num_kernels, 1))
			
			return output
			
		else:
			result = np.zeros((input_tensor.shape[0], kernel.shape[0], input_tensor.shape[-2], input_tensor.shape[-1]))  # Initialize output array		
			# Adjust padding for even-sized kernels
			if kernel.shape[-2] % 2 == 0:
				m_pad_width_left = max(0, kernel.shape[-2] // 2)
				m_pad_width_right = max(0, kernel.shape[-2] // 2 - 1)
			else:
				m_pad_width_left = max(0, kernel.shape[-2] // 2)
				m_pad_width_right = max(0, kernel.shape[-2] // 2)
				
			if kernel.shape[-1] % 2 == 0:
				n_pad_width_left = max(0, kernel.shape[-1] // 2)
				n_pad_width_right = max(0, kernel.shape[-1] // 2 - 1)
			else:
				n_pad_width_left = max(0, kernel.shape[-1] // 2) 
				n_pad_width_right = max(0, kernel.shape[-1] // 2)
				
			pad_width = [(0,0), (m_pad_width_left, m_pad_width_right), (n_pad_width_left, n_pad_width_right)]
				
			for i in range(input_tensor.shape[0]):  # Loop through batch dimension
				for j in range(kernel.shape[0]):  # Loop through output channels of kernel
					padded = np.pad(input_tensor[i], pad_width, mode='constant', constant_values=0)
						
					# Perform correlation for each channel pair
					result[i, j, :, :] = correlate(padded, kernel[j], mode='valid')
					#result[i, j, :, :] += self.bias[j]
		
		output = result[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
		output += self.bias.reshape((1, self.num_kernels, 1, 1))
		
		#self.old_output = output
		return output
		
		
	def backward(self, error_tensor):
		#Set matrix sizes
		input_tensor = self.old_input
		if(self.one_dim):
			input_tensor = input_tensor[:, :, :, np.newaxis]
			self.weights = self.weights[:, :, :, np.newaxis]

		next_layer_error_tensor = np.zeros(self.input_tensor_shape)
		self._gradient_weights = np.zeros_like(self.weights)
		self._gradient_bias = np.zeros(self.num_kernels)

        # compute gradient with resect to next layer
		if self.one_dim:

			self._gradient_bias = np.sum(error_tensor, axis=1)
			padded_error = np.zeros((self.input_tensor_shape[0],self.num_kernels, input_tensor.shape[2]))
			padded_error = padded_error[:, :, :, np.newaxis]
			next_layer_error_tensor = next_layer_error_tensor[:, :, :, np.newaxis]


			for i in range(input_tensor.shape[0]):
				for j in range(self.num_kernels):
					for k in range(error_tensor.shape[2]):
							padded_error[i,j,k * self.stride_shape[0]] = error_tensor[i,j,k]
					#gradient
					for m in range(self.input_tensor_shape[1]):
						next_layer_error_tensor[i,m] += convolve(padded_error[i,j],self.weights[j,m],mode="same")
		else:
			self._gradient_bias = np.sum(error_tensor, axis=(0,2,3))
			padded_error = np.zeros((self.input_tensor_shape[0],self.num_kernels, self.input_tensor_shape[2], input_tensor.shape[3]))

			for i in range(input_tensor.shape[0]):
				for j in range(self.num_kernels):
					for k in range(error_tensor.shape[2]):
							for l in range(error_tensor.shape[3]):
								padded_error[i,j,k * self.stride_shape[0], l * self.stride_shape[1]] = error_tensor[i,j,k,l]
					#gradient
					for m in range(self.input_tensor_shape[1]):
						next_layer_error_tensor[i,m] += convolve(padded_error[i,j],self.weights[j,m],mode="same")


        # adjust padding for 1D case
		y_pad = None
		conv_last_dim = None
		if(self.one_dim):
			y_pad = 0
			conv_last_dim = 1
		else:
			y_pad = self.convolution_shape[2] // 2
			conv_last_dim = self.convolution_shape[2]

		x_pad = self.convolution_shape[1] // 2

		
		
		unpadded = np.zeros((*input_tensor.shape[:2], input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                   input_tensor.shape[3] + conv_last_dim - 1))

        # unpad the input tensor
		for i in range(input_tensor.shape[0]):
			for j in range(input_tensor.shape[1]):
				for k in range(unpadded.shape[2]):
					for w in range(unpadded.shape[3]):
						if (k > x_pad - 1) and (k < input_tensor.shape[2] + x_pad):
							if (w > y_pad - 1) and (w < input_tensor.shape[3] + y_pad):
								unpadded[i, j, k, w] = input_tensor[i, j, k - x_pad, w - y_pad]


        # compute gradients with respect to weights
		for i in range(input_tensor.shape[0]):
			for k in range(self.num_kernels):
				for c in range(input_tensor.shape[1]):
					# convolution of the error tensor with the padded input tensor
					if(len(unpadded[i, c, :].shape) != len(padded_error[i, k, :].shape)):
						print("unpadded = ", unpadded[i, c, :].shape)
						print("padded = ", padded_error[i, k, :].shape)
						padded_error = padded_error[:, :, :, np.newaxis]
						print("padded new = ", padded_error[i, k, :].shape)


					self._gradient_weights[k, c, :] += correlate(unpadded[i, c, :], padded_error[i, k, :], 'valid')  # valid padding
					
		# update weights and bias
		if self._optimizer_weights:
			self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
		if self._optimizer_bias:
			self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
		
		# special case for 1D case
		if(self.one_dim):
			one_dim_output = next_layer_error_tensor.squeeze(axis = 3)
			next_layer_error_tensor = one_dim_output

		return next_layer_error_tensor
