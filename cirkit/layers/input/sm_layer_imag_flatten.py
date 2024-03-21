from typing import Callable, Union
from typing import Literal, cast

import torch
from torch import Tensor

import math

from cirkit.layers.input import InputLayer
from cirkit.utils.type_aliases import ReparamFactory
from cirkit.reparams.leaf import ReparamExp, ReparamIdentity

# TODO: rework interface and docstring, the const value should be properly shaped


class SMKernelImagFlattenLayerParams(torch.nn.Module):
    """The constant layer, i.e., an input layer that returns constant values."""
    
    params_sigma: ReparamFactory
    params_mu: ReparamFactory
    
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_output_units: int,
        reparam_sigma: ReparamFactory = ReparamExp,
        reparam_mu: ReparamFactory = ReparamIdentity,
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
        super().__init__()
        
        self.num_output_units = num_output_units
        self.num_vars = num_vars
        
        self.params_sigma = reparam_sigma(
            (self.num_output_units, 1, self.num_vars), dim=-1
        )
        self.params_mu = reparam_mu(
            (self.num_output_units, 1, self.num_vars), dim=-1
        )
        self.reset_parameters()
        # shape (self.num_vars, self.num_output_units, self.num_replicas, self.num_suff_stats)

    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.2, 0.2)

    
class SMKernelPosImagLayer(InputLayer):
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
        params_module: SMKernelImagFlattenLayerParams
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
        )
        
        self.params = params_module
        
        # parameters are shared in the parent class
        
    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        
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
        # (B, D, 1) -> (1, B, D)
        x1_ = x1.squeeze(-1).unsqueeze(-3)
        x2_ = x2.squeeze(-1).unsqueeze(-3)

        # (1, B, D) * (K, 1, D) -> (K, B, D)
        x1_exp = x1_ * self.params.params_sigma()
        x2_exp = x2_ * self.params.params_sigma()
        x1_cos = x1_ * self.params.params_mu()
        x2_cos = x2_ * self.params.params_mu()

        # Create grids
        # (K, B, 1, D)
        x1_exp_ = x1_exp.unsqueeze(-2)
        # (K, 1, B, D)
        x2_exp_ = x2_exp.unsqueeze(-3)

        x1_cos_ = x1_cos.unsqueeze(-2)
        x2_cos_ = x2_cos.unsqueeze(-3)

        # Compute the exponential and cosine terms
        # (K, B, B, D)
        real_exp_term = (x1_exp_ - x2_exp_).pow_(2).mul_(-2 * math.pi**2)
        imag_cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        
        # inf to 0
        real_exp_term[torch.isinf(real_exp_term)] = 0
        imag_cos_term[torch.isinf(imag_cos_term)] = 0
        
        # (K, B, B, D)
        fin_term = torch.complex(real_exp_term, imag_cos_term.float())
        
        # (B, B, D, K, 1)
        return fin_term.permute(1, 2, 3, 0).unsqueeze(-1)



class SMKernelNegImagLayer(InputLayer):
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
        params_module: SMKernelImagLayerParams
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
        )
        
        self.params = params_module
        
        # parameters are shared in the parent class
        
    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        
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
        # (B, D, 1) -> (1, B, D)
        x1_ = x1.squeeze(-1).unsqueeze(-3)
        x2_ = x2.squeeze(-1).unsqueeze(-3)

        # (1, B, D) * (K, 1, D) -> (K, B, D)
        x1_exp = x1_ * self.params.params_sigma()
        x2_exp = x2_ * self.params.params_sigma()
        x1_cos = x1_ * self.params.params_mu()
        x2_cos = x2_ * self.params.params_mu()

        # Create grids
        # (K, B, 1, D)
        x1_exp_ = x1_exp.unsqueeze(-2)
        # (K, 1, B, D)
        x2_exp_ = x2_exp.unsqueeze(-3)

        x1_cos_ = x1_cos.unsqueeze(-2)
        x2_cos_ = x2_cos.unsqueeze(-3)

        # Compute the exponential and cosine terms
        # (K, B, B, D)
        real_exp_term = (x1_exp_ - x2_exp_).pow_(2).mul_(-2 * math.pi**2)
        imag_cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        # Imag term is negative!!!
        imag_cos_term = -imag_cos_term
        
        # inf to 0
        real_exp_term[torch.isinf(real_exp_term)] = 0
        imag_cos_term[torch.isinf(imag_cos_term)] = 0
        
        # (K, B, B, D)
        fin_term = torch.complex(real_exp_term, imag_cos_term.float())
        
        # (B, B, D, K, 1)
        return fin_term.permute(1, 2, 3, 0).unsqueeze(-1)
        
