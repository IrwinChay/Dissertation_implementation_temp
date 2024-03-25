from typing import Callable, Union
from typing import Literal, cast

import torch
from torch import Tensor

from cirkit.layers.input import InputLayer
from cirkit.utils.type_aliases import ReparamFactory
from cirkit.reparams.leaf import ReparamExp, ReparamSoftmax
from cirkit.utils.log_trick import log_func_exp

# TODO: rework interface and docstring, the const value should be properly shaped


class RBFKernelLayer(InputLayer):
    """The constant layer, i.e., an input layer that returns constant values."""
    
    params: Tensor
    
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[0] = 0,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamExp,
    ) -> None:
        """Init class.

        Args:
            num_vars (int): The number of variables of the circuit.
            num_channels (int, optional): The number of channels of each variable. Defaults to 1.
            num_replicas (int, optional): The number of replicas for each variable. Defaults to 1.
            num_input_units (Literal[1], optional): The number of input units, must be 1. \
                Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            num_folds (Literal[0], optional): The number of folds. Should not be provided and will \
                be calculated as num_vars*num_replicas. Defaults to 0.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamEFNormal.
        """
        super().__init__(
            num_vars=num_vars,
            num_channels=num_channels,
            num_replicas=num_replicas,
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
            num_suff_stats=2 * num_channels,
        )
        
        self.params = reparam(
            (self.num_vars, self.num_output_units), dim=-1
        )
        reparam_weights = ReparamSoftmax
        self.params_weight = reparam_weights(
            (self.num_vars, self.num_output_units, self.num_output_units), dim=-2
        )
        
        self.reset_parameters()
        # shape (self.num_vars, self.num_output_units, self.num_replicas, self.num_suff_stats)

    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        for param in self.parameters():
            torch.nn.init.uniform_(param, 0.01, 0.99)

    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of constants functions is not implemented")

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x1 (Tensor): The first input to this layer, shape (*B, D, C).
            x2 (Tensor): The second input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        x1_ = x1.squeeze(-1).unsqueeze(-2)
        # (B, 1, D)
        x2_ = x2.squeeze(-1).unsqueeze(-3)
        # (1, B, D)
        exp_term = (x1_ - x2_).unsqueeze(-1).div(self.params() + 1e-6)
        # (B1, B2, D, K)
        activation = exp_term.pow_(2).div_(-2)

        def _forward_linear(x: Tensor) -> Tensor:
            return torch.einsum('...di,dio->...do', x, self.params_weight())
        
        activation = log_func_exp(
            activation, func=_forward_linear, dim=-1, keepdim=True)
        # (B1, B2, D, K)
        return activation.unsqueeze(-1)
        # (B1, B2, D, K, 1)
        
