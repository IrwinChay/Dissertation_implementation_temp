import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel, SpectralMixtureKernel

from cirkit.layers.input.sm_layer_imag import SMKernelPosImagLayer, SMKernelNegImagLayer, SMKernelImagLayerParams

from cirkit.models.sm_kernel import SMCircuitKernel

from sklearn import cluster


def initial_values(train_dataset, feature_extractor, n_inducing_points):
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)
    print("f_X_samples", f_X_samples.shape)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )

    return initial_inducing_points

def initialize_from_data(params_module: SMKernelImagLayerParams,
                         train_x: torch.Tensor, 
                         train_y: torch.Tensor, 
                         **kwargs):
    """
    Initialize mixture components based on batch statistics of the data. You should use
    this initialization routine if your observations are not evenly spaced.

    :param torch.Tensor train_x: Training inputs
    :param torch.Tensor train_y: Training outputs
    """

    with torch.no_grad():
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            train_x = torch.tensor(train_x)
            train_y = torch.tensor(train_y)
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)

        # Compute maximum distance between points in each dimension
        train_x_sort = train_x.sort(dim=-2)[0]
        max_dist = train_x_sort[..., -1, :] - train_x_sort[..., 0, :]

        # Compute the minimum distance between points in each dimension
        dists = train_x_sort[..., 1:, :] - train_x_sort[..., :-1, :]
        # We don't want the minimum distance to be zero, so fill zero values with some large number
        dists = torch.where(dists.eq(0.0), torch.tensor(1.0e10, dtype=train_x.dtype, device=train_x.device), dists)
        sorted_dists = dists.sort(dim=-2)[0]
        min_dist = sorted_dists[..., 0, :]

        # Reshape min_dist and max_dist to match the shape of parameters
        # First add a singleton data dimension (-2) and a dimension for the mixture components (-3)
        min_dist = min_dist.unsqueeze_(-2).unsqueeze_(-3)
        max_dist = max_dist.unsqueeze_(-2).unsqueeze_(-3)
        # Compress any dimensions in min_dist/max_dist that correspond to singletons in the SM parameters
        dim = -3
        while -dim <= min_dist.dim():
            if -dim > params_module.params_sigma.param.dim():
                min_dist = min_dist.min(dim=dim)[0]
                max_dist = max_dist.max(dim=dim)[0]
            elif params_module.params_sigma.param.size(dim) == 1:
                min_dist = min_dist.min(dim=dim, keepdim=True)[0]
                max_dist = max_dist.max(dim=dim, keepdim=True)[0]
                dim -= 1
            else:
                dim -= 1

        # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
        actual_init_sigma = torch.randn_like(params_module.params_sigma.param).mul_(max_dist).abs_().reciprocal_()
        params_module.params_sigma.param = torch.nn.Parameter(torch.log(actual_init_sigma))
        # Draw means from Unif(0, 0.5 / minimum distance between two points)
        actual_init_means = torch.rand_like(params_module.params_mu.param).mul_(0.5).div(min_dist)
        params_module.params_mu.param = torch.nn.Parameter(actual_init_means)
        # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
        # self.mixture_weights = train_y.std().div(self.num_mixtures)


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    # print("f_X_sample.view(1000, -1)", type(f_X_sample))
    # kmeans.fit(f_X_sample.reshape(1000, -1))
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points

class CircuitSMGP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        num_features,
        initial_inducing_points,
        circuit,
        use_default_sm = False, 
        default_sm_kwargs = None
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape, mean_init_std=1e-4
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution, learn_inducing_locations=True,  # jitter_val=1e-4,
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs, jitter_val=1e-4
            )

        super().__init__(variational_strategy)
        
        if use_default_sm:
            self.kernel = SpectralMixtureKernel(**default_sm_kwargs)
            # num_mixtures, ard_num_dims, batch_shape, 

        else:
        
            self.circuit = circuit

            kwargs = {
                "circuit": circuit, 
                "batch_shape": batch_shape,
            }
            self.kernel = SMCircuitKernel(**kwargs)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(self.kernel, batch_shape=batch_shape) 

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)
    
    def get_covar(self, x):
        # mean = self.mean_module(x)
        covar = self.covar_module(x)

        return covar

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param

