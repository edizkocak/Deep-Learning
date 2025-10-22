import numpy as np
import matplotlib.pyplot as plt

class Checker:
	def __init__(self, resolution: int, tile_size: int):
		assert resolution % (2 * tile_size) == 0
		assert tile_size > 1
		
		self.resolution = resolution
		self.tile_size = tile_size
		self.output = None
	
	def draw(self):
		left1 = np.tile(np.zeros((1,), dtype=int), self.tile_size)
		right1 = np.tile(np.ones((1,), dtype=int), self.tile_size)
		
		first_row = np.concatenate((left1, right1), axis=0)
		second_row = first_row[::-1]
		
		repeat = self.resolution // self.tile_size // 2
		first_row = np.tile(first_row, (self.tile_size, repeat))
		second_row = np.tile(second_row, (self.tile_size, repeat))
		
		two_rows = np.concatenate((first_row, second_row), axis=0) 	
		self.output = np.tile(two_rows, (self.resolution // (self.tile_size * 2), 1))
		
		return np.copy(self.output)
		
	def show(self):
		plt.imshow(self.output, cmap = 'grey')
		plt.show()
			


class Circle:
	def __init__(self, resolution: int, radius: int, position: (int, int)):
		assert resolution > 2
		assert radius > 0
		
		self.res_half = resolution / 2
		
		assert position[0] >= 0 and position[1] >= 0 and position[0] <= self.res_half and position[0] >= -self.res_half and position[1] <= self.res_half and position[1] >= -self.res_half
		
		self.resolution = resolution
		self.radius = radius
		self.position = position
		self.output = None
		
	def draw(self):
		x = np.linspace(-self.res_half, self.res_half, self.resolution)
		y = np.linspace(-self.res_half, self.res_half, self.resolution)
		
		X, Y = np.meshgrid(x,y)
		
		X = X - self.position[0] + self.res_half
		Y = Y - self.position[1] + self.res_half
		self.output = X**2 + Y**2 <= self.radius**2
		self.output = self.output.astype(int)
		
		return np.copy(self.output)

	def show(self):
		plt.imshow(self.output, cmap = 'binary', extent=[-self.res_half, self.res_half, -self.res_half, self.res_half])
		plt.colorbar()
		plt.show()
		
class Spectrum:
	def __init__(self, resolution: int):
		assert resolution > 2
		self.resolution = resolution
		self.res_half = resolution / 2
		self.output = None
		
	def draw(self):
		x = np.linspace(-self.res_half, self.res_half, self.resolution)
		y = np.linspace(-self.res_half, self.res_half, self.resolution)
		
		X, Y = np.meshgrid(x,y)
		
		R = (X + self.res_half) / self.resolution 
		G = (Y + self.res_half) / self.resolution
		B = (X + self.res_half)
		B = B[:, ::-1]
		B = B / self.resolution
		
		R = np.expand_dims(R, axis=2)
		G = np.expand_dims(G, axis=2)
		B = np.expand_dims(B, axis=2)
		
		img = np.concatenate( (R, G, B), axis=2)
		self.output = img
		return np.copy(self.output)
		
	def show(self):
		plt.imshow(self.output)
		plt.show()
		
		
		
		

		
		
		
	
		
