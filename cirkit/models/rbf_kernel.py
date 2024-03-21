from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

import gpytorch
from gpytorch.constraints import Positive

from cirkit.models.tensorized_circuit import TensorizedPC

class RBFCircuitKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True
    circuit: TensorizedPC

    # We will register the parameter when initializing the kernel
    def __init__(
        self, 
        circuit: TensorizedPC, 
        batch_shape: Optional[torch.Size] = torch.Size([]),
        **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)

        
        self.circuit = circuit
        
        print ("All circuit parameters shape: ")
        for param in circuit.parameters(): 
            print (param.shape)


    # this is the kernel function
    def forward(self, x1, x2, **params):
        return self.circuit(x1.unsqueeze(-1), x2.unsqueeze(-1)).squeeze(-1)