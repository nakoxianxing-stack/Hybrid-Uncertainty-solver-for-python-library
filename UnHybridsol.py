#5032_ir


import numpy as np
from enum import Enum, auto
from typing import List, Callable, Union, Optional
import time

class PropagationMethod(Enum):
    LINEAR = auto()
    MONTE_CARLO = auto()
    AUTO = auto()

class SDEType(Enum):
    ITO  = auto()
    STRATONOVICH = auto()

#class





class ComputatonContext:
    def __init__(self, method: PropagationMethod = PropagationMethod.LINEAR, mc_samples: int = 1000):
        self.method = method
        self.mc_samples = mc_samples
        self.seed = 42
        self._rng = np.random.default_rng(self.seed)

    def set_method(self,method:PropagationMethod):
        self.method = method

    def get_rng(self):
        return self._rng
    
