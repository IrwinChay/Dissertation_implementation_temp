#!/usr/bin/env python3

import logging
import math
from typing import Optional, Tuple, Union

import torch

from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel

logger = logging.getLogger()


class EnsembleRBFKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Spectral Mixture Kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

    It was proposed in `Gaussian Process Kernels for Pattern Discovery and Extrapolation`_.

    .. note::
        Unlike other kernels,

            * ard_num_dims **must equal** the number of dimensions of the data.
            * This kernel should not be combined with a :class:`gpytorch.kernels.ScaleKernel`.

    :param int num_mixtures: The number of components in the mixture.
    :param int ard_num_dims: Set this to match the dimensionality of the input.
        It should be `d` if x1 is a `... x n x d` matrix. (Default: `1`.)
    :param batch_shape: Set this if the data is batch of input data. It should
        be `b_1 x ... x b_j` if x1 is a `b_1 x ... x b_j x n x d` tensor. (Default: `torch.Size([])`.)
    :type batch_shape: torch.Size, optional
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the dimensions. (Default: `None`.)
    :type active_dims: float, optional
    :param eps: The minimum value that the lengthscale can take (prevents divide by zero errors). (Default: `1e-6`.)
    :type eps: float, optional

    :param mixture_scales_prior: A prior to set on the mixture_scales parameter
    :type mixture_scales_prior: ~gpytorch.priors.Prior, optional
    :param mixture_scales_constraint: A constraint to set on the mixture_scales parameter
    :type mixture_scales_constraint: ~gpytorch.constraints.Interval, optional
    :param mixture_means_prior: A prior to set on the mixture_means parameter
    :type mixture_means_prior: ~gpytorch.priors.Prior, optional
    :param mixture_means_constraint: A constraint to set on the mixture_means parameter
    :type mixture_means_constraint: ~gpytorch.constraints.Interval, optional
    :param mixture_weights_prior: A prior to set on the mixture_weights parameter
    :type mixture_weights_prior: ~gpytorch.priors.Prior, optional
    :param mixture_weights_constraint: A constraint to set on the mixture_weights parameter
    :type mixture_weights_constraint: ~gpytorch.constraints.Interval, optional

    :ivar torch.Tensor mixture_scales: The lengthscale parameter. Given
        `k` mixture components, and `... x n x d` data, this will be of size `... x k x 1 x d`.
    :ivar torch.Tensor mixture_means: The mixture mean parameters (`... x k x 1 x d`).
    :ivar torch.Tensor mixture_weights: The mixture weight parameters (`... x k`).

    Example:

        >>> # Non-batch
        >>> x = torch.randn(10, 5)
        >>> covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5)
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> # Batch
        >>> batch_x = torch.randn(2, 10, 5)
        >>> covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, batch_size=2, ard_num_dims=5)
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)

    .. _Gaussian Process Kernels for Pattern Discovery and Extrapolation:
        https://arxiv.org/pdf/1302.4245.pdf
    """

    is_stationary = (
        True  # kernel is stationary even though it does not have a lengthscale
    )

    def __init__(
        self,
        num_mixtures: Optional[int] = None,
        ard_num_dims: Optional[int] = 1,
        batch_shape: Optional[torch.Size] = torch.Size([]),
        mixture_scales_prior: Optional[Prior] = None,
        mixture_scales_constraint: Optional[Interval] = None,
        mixture_weights_prior: Optional[Prior] = None,
        mixture_weights_constraint: Optional[Interval] = None,
        **kwargs,
    ):

        print("ensemble of RBF kernel used")

        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")

        # This kernel does not use the default lengthscale
        super().__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)
        self.num_mixtures = num_mixtures

        if mixture_scales_constraint is None:
            mixture_scales_constraint = Positive()

        # if mixture_means_constraint is None:
        #     mixture_means_constraint = Positive()

        if mixture_weights_constraint is None:
            mixture_weights_constraint = Positive()

        self.register_parameter(
            name="raw_mixture_weights",
            parameter=torch.nn.Parameter(
                torch.zeros(*self.batch_shape, self.num_mixtures)
            ),
        )
        # (D, K)
        ms_shape = torch.Size([*self.batch_shape, self.ard_num_dims, self.num_mixtures])
        self.register_parameter(
            name="raw_mixture_scales",
            parameter=torch.nn.Parameter(torch.zeros(ms_shape)),
        )

        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)

        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.2, 0.2)

    @property
    def mixture_scales(self):
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)

    @mixture_scales.setter
    def mixture_scales(self, value: Union[torch.Tensor, float]):
        self._set_mixture_scales(value)

    def _set_mixture_scales(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(
            raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(
                value
            )
        )

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @mixture_weights.setter
    def mixture_weights(self, value: Union[torch.Tensor, float]):
        self._set_mixture_weights(value)

    def _set_mixture_weights(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(
            raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(
                value
            )
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(
                    self.ard_num_dims, num_dims
                )
            )

        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (... x n x d x k) for k mixtures
        # (N, 1, D)
        x1_ = x1.unsqueeze(-2)
        # (1, N, D)
        x2_ = x2.unsqueeze(-3)
        # (N, N, D, K)
        exp_term = (x1_ - x2_).unsqueeze(-1).div(self.mixture_scales + 1e-6)
        exp_term = exp_term.pow_(2).div_(-2)

        # (N, N, K)
        exp_term = torch.sum(exp_term, dim=-2).exp_()

        # (N, N)
        res = torch.einsum("k,ijk->ij", self.mixture_weights, exp_term)
        return res
