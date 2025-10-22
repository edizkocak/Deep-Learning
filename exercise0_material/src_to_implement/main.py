import pattern
import generator
#import NumpyTests.TestCheckers




if __name__ == "__main__":
	#board = pattern.Checker(100, 25)
	#board.draw()
	#board.show()
	
	#circle = pattern.Circle(512, 20, (50,50))
	#circle.draw()
	#circle.show()
	
	#spectrum = pattern.Spectrum(512)
	#spectrum.draw()
	#spectrum.show()
	
	
	file_path = "./exercise_data/"
	label_path = "./Labels.json"
	batch_size = 9
	image_size = [512, 512, 3]
	
	generator = generator.ImageGenerator(file_path, label_path, batch_size, image_size)
	generator.show()
	
	
