import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

# from gpytorch.kernels import SpectralMixtureKernel
from due.sm_kernel import SMKernel


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
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()
    print("initial_lengthscale", initial_lengthscale)

    return initial_lengthscale.cpu()


class GP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        num_features,
        initial_inducing_points,
        kernel="RBF",
        initial_lengthscale = None,
        sm_kernel_mixtures = 1, 
        sm_kernel_x_train = None, 
        sm_kernel_y_train = None, 
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )

        super().__init__(variational_strategy)

        kwargs = {
            "batch_shape": batch_shape,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
            kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)
        elif kernel == "HBF":
            kernel = RBFKernel(batch_shape=batch_shape, ard_num_dims=num_features)
            kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)
        elif kernel == "SM":
            
            sm_kernel_kwargs = {**kwargs, 
                                "num_mixtures": sm_kernel_mixtures, 
                                "ard_num_dims" : num_features,
                                }
            # num_mixtures, ard_num_dims, batch_shape
            
            kernel = SMKernel(**sm_kernel_kwargs) # SMKernel SpectralMixtureKernel
            # kernel.initialize_from_data(sm_kernel_x_train, sm_kernel_y_train)
        else:
            raise ValueError("Specified kernel not known.")
        
        self.kernel = kernel

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


class DKL(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does some magic
        on the forward method which is not compatible with a feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.gp = gp

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp(features)
