from . import Base
import numpy as np

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

        self.activated_input = None

    def forward(self, input_tensor):
        self.activated_input = 1.0 / (1.0 + np.exp(-input_tensor)) 
        return self.activated_input

    def backward(self, error_tensor):
        return self.activated_input * (1.0 - self.activated_input) * error_tensor
