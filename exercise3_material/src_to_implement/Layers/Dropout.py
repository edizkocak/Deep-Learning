from . import Base
import numpy as np

class Dropout(Base.BaseLayer):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        activation = np.where(np.random.random(input_tensor.shape) <= 1.0 - self.probability, 0, input_tensor)

        if self.testing_phase:
            activation *= 1.0 - self.probability
        else:
            activation *= (1.0 / self.probability)

        return activation

    def backward(self, error_tensor):
        pass

