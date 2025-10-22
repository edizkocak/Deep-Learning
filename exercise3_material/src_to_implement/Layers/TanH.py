from . import Base
import numpy as np

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

        self.activated_input=None

    def forward(self, input_tensor):
        self.activated_input = np.tanh(input_tensor)
        return self.activated_input

    def backward(self, error_tensor):
        return error_tensor * (1.0  - self.activated_input * self.activated_input)
