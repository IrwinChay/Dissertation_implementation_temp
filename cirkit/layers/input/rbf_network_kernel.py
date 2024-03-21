
from typing import Callable, Union
from typing import Literal, cast

import torch
from torch import Tensor

from cirkit.layers.input import InputLayer
from cirkit.utils.type_aliases import ReparamFactory
from cirkit.reparams.leaf import ReparamExp, ReparamIdentity

# TODO: rework interface and docstring, the const value should be properly shaped


class RBFNetworkKernelLayer(InputLayer):
    """The constant layer, i.e., an input layer that returns constant values."""
    
    params_sigma: Tensor
    params_mu: Tensor
    
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
        reparam_sigma: ReparamFactory = ReparamExp,
        reparam_mu: ReparamFactory = ReparamIdentity,
        reparam_weights: ReparamFactory = ReparamIdentity,
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
            reparam=reparam_sigma,
            num_suff_stats=2 * num_channels,
        )
        
        # size = (self.num_vars, self.num_output_units)
        # self.params = torch.nn.Parameter(torch.zeros(size))
        
        self.params_sigma = reparam_sigma(
            (self.num_vars, self.num_output_units), dim=-1
        )
        self.params_mu = reparam_mu(
            (self.num_vars, self.num_output_units), dim=-1
        )
        self.params_weight = reparam_weights(
            (self.num_vars, self.num_output_units, self.num_output_units), dim=-1
        )
        self.reset_parameters()
        # shape (self.num_vars, self.num_output_units, self.num_replicas, self.num_suff_stats)

    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        torch.nn.init.uniform_(
                self.params_mu.param,
                a=-1.0,
                b=1.0)
        torch.nn.init.normal_(self.params_sigma.param, mean=0.0, std=0.1)
        torch.nn.init.uniform_(
                self.params_weight.param,
                a=-0.2,
                b=0.2)

    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of constants functions is not implemented")

    def forward(self, x) -> Tensor:
        """Run forward pass.

        Args:
            x1 (Tensor): The first input to this layer, shape (*B, D, C).
            x2 (Tensor): The second input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        # Input has size B x Fin
        batch_size = x.size(0)

        # Compute difference from centers
        # c has size B x num_kernels x Fin
        # (B, D, K)
        c = self.params_mu().expand(batch_size, self.num_vars, self.num_output_units)

        diff = x.unsqueeze(-1) - c

        # (B, D, K)
        eps_r = diff * self.params_sigma().unsqueeze(0)

        rbf = torch.exp((-eps_r.pow(2)))
        
        # (B, D, K)
        rbf = torch.einsum('bdi,dio->bdo', rbf, self.params_weight())
        
        return rbf.unsqueeze(-1)
        # (B, D, K, 1)
