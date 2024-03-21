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
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel

from cirkit.models.rbf_kernel import RBFCircuitKernel

from sklearn import cluster


def initial_values(train_dataset, feature_extractor, n_inducing_points):
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    # idx = torch.randperm(len(train_dataset)).chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)
    print("f_X_samples shape", f_X_samples.shape)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale

def _pdist_per_dim(input_tensor):
    # Expand the input tensor to form all pairs for differences calculation
    t1 = input_tensor.unsqueeze(1)  # Shape: (1000, 1, 8)
    t2 = input_tensor.unsqueeze(0)  # Shape: (1, 1000, 8)

    # Compute pairwise differences for each dimension (broadcasting)
    diffs = torch.abs(t1 - t2)  # Shape: (1000, 1000, 8)

    # Mask to extract the upper triangular part without the diagonal
    mask = torch.triu(torch.ones(input_tensor.shape[0], input_tensor.shape[0]), diagonal=1).bool()

    # Apply mask and reshape to get the final shape (499500, 8)
    result = diffs[mask].reshape(-1, input_tensor.shape[1])

    return result


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    # print("f_X_sample.view(1000, -1)", type(f_X_sample))
    # kmeans.fit(f_X_sample.reshape(1000, -1))
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points

def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()
    print("initial_lengthscale", initial_lengthscale)

    return initial_lengthscale.cpu()


class CircuitGP(ApproximateGP):
    def __init__(
        self,
        num_outputs,
        num_features,
        initial_lengthscale,
        initial_inducing_points,
        circuit,
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
        
        self.circuit = circuit

        kwargs = {
            "circuit": circuit, 
            "batch_shape": batch_shape,
        }
        self.kernel = RBFCircuitKernel(**kwargs)
        
        ################## Initialize Lengthscale ##################
        # lengthscales = self.kernel.circuit.input_layer.params.param
        
        # self.kernel.circuit.input_layer.params.param = torch.nn.Parameter(
        #     torch.log(initial_lengthscale * torch.ones_like(lengthscales) ))
        

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

