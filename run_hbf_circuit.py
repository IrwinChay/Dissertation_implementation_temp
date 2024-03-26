import random
import time
import numpy as np
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from uci_datasets import Dataset

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RootMeanSquaredError

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.region_graph.fully_factorized import FullyFactorized
from cirkit.models.gp import CircuitGP, initial_values
from cirkit.layers.sum_product import CPLayer
# from cirkit.layers.input.rbf_kernel_flatten import RBFKernelFlattenLayer
from cirkit.layers.input.rbf_kernel import RBFKernelLayer
from cirkit.reparams.leaf import ReparamExp, ReparamLogSoftmax, ReparamSoftmax
from cirkit.models.tensorized_circuit import TensorizedPC

class IdentityMapping(nn.Module):
    def __init__(self):
        super(IdentityMapping, self).__init__()
    
    def forward(self, x):
        return x
    
step_counter = 0

################## Hyper Param ##################

dataset = "kin40k"
lr = 1e-3
batch_size = 32
epochs = 50
n_inducing_points_s = [5, 10, 50, 100, 500]
kernel = "HBF" # not used

num_vars = 8
num_mixtures = 32
depth = 3
num_repetitions = 10
# region_graph = FullyFactorized(num_vars=num_vars)
region_graph = RandomBinaryTree(num_vars=num_vars, depth=depth, num_repetitions=num_repetitions)
efamily_cls = RBFKernelLayer   # Flatten
layer_cls = CPLayer
reparam = ReparamSoftmax


def main():

    all_search_test_rmse_means = []
    all_search_test_rmse_stds = []
    all_search_training_time_means = []
    all_search_training_time_stds = []

    for n_inducing_points in n_inducing_points_s:

        print("n_inducing_points: ", n_inducing_points)
    
        all_test_rmses = []
        all_training_time = []
        all_peak_memory = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        print("Device: ", device)
        # The device to use, e.g., "cpu", "cuda", "cuda:1"
        
        for run in range(1, 4):  # Running the model 3 times
            
            dataset_split = random.randint(0, 9)
            efamily_kwargs = {}
    
            layer_kwargs = {'rank': 1}
    
            ################## Random ##################
            
            print()
            print()
            print(f"Run {run}")
            print()
            print()
    
            # Update seed settings for each run
            seed = run * 10  # Example: 10, 20, 30 for the three runs
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.manual_seed_all(seed) 
                
            
            ################## Circuit ##################
    
            pc = TensorizedPC.from_region_graph(
                region_graph,
                num_inner_units=num_mixtures,
                num_input_units=num_mixtures,
                efamily_cls=efamily_cls,
                efamily_kwargs=efamily_kwargs,
                layer_cls=layer_cls,
                layer_kwargs=layer_kwargs,
                num_classes=1,
                reparam=reparam # ReparamLogSoftmax #  ReparamSoftmax
            )
            pc.to(device)
            print(pc)
    
            print("Circuit parameters: ")
            for param in pc.parameters(): 
                print (param.shape)
                
            total_params = sum(p.numel() for p in pc.parameters() if p.requires_grad)
            print(f"Total number of parameters: {total_params}")
            
            print()
            print()
            
            
            ################## Dataset ##################
            
            data = Dataset(dataset)
            x_train, y_train, x_test, y_test = data.get_split(split=dataset_split) 
    
            val_split_point = x_train.shape[0] - x_test.shape[0]
    
            x_train_real = x_train[:val_split_point] 
            y_train_real = y_train[:val_split_point]
            y_train_real = y_train_real.squeeze()
            x_val = x_train[val_split_point:]
            y_val = y_train[val_split_point:]
            y_val = y_val.squeeze()
            y_test = y_test.squeeze()
    
            # Normalize dataset
            mean = x_train_real.mean(axis=0)
            std = x_train_real.std(axis=0)
    
            x_train_real_normalized = (x_train_real - mean) / std
            x_val_normalized = (x_val - mean) / std
            x_test_normalized = (x_test - mean) / std
    
            input_dim = x_train_real_normalized.shape[1]
            num_outputs = 1
            feature_extractor = IdentityMapping()
    
            print("Training dataset size: ", x_train_real_normalized.shape[0])
            print("Val dataset size: ", x_val_normalized.shape[0])
            print("Test dataset size: ", x_test_normalized.shape[0])
            print("Input dimension: ", input_dim)
            print()
            print()
            
            ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train_real_normalized).float(), torch.from_numpy(y_train_real).float())
            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True) # suffle 
    
            ds_val = torch.utils.data.TensorDataset(torch.from_numpy(x_val_normalized).float(), torch.from_numpy(y_val).float())
            dl_val = torch.utils.data.DataLoader(ds_val, batch_size=32, shuffle=False)
    
            ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test_normalized).float(), torch.from_numpy(y_test).float())
            dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False)
            
            ################## Training Definitions ##################
    
            initial_inducing_points, initial_lengthscale = initial_values(
                    ds_train, feature_extractor, n_inducing_points
            )
    
            gp_model = CircuitGP(
                num_outputs=num_outputs,
                num_features=input_dim,          # CHANGE features / input_dim
                initial_lengthscale=initial_lengthscale,
                initial_inducing_points=initial_inducing_points,
                circuit=pc
                # kernel=kernel,
            )
 
            likelihood = GaussianLikelihood()
            elbo_fn = VariationalELBO(likelihood, gp_model, num_data=len(ds_train))
            loss_fn = lambda x, y: -elbo_fn(x, y)
            
            if torch.cuda.is_available():
                gp_model = gp_model.cuda()
                likelihood = likelihood.cuda()
    
            parameters = [
                {"params": gp_model.parameters(), "lr": lr},
            ]
            parameters.append({"params": likelihood.parameters(), "lr": lr})
                
            optimizer = torch.optim.Adam(parameters)
            pbar = ProgressBar()
            
    
            def step(engine, batch):
                
                global step_counter
                
                step_counter += 1
                
                gp_model.train()
                likelihood.train()
                
                optimizer.zero_grad()
                
                x, y = batch
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    
                y_pred = gp_model(x) # get y    
                loss = loss_fn(y_pred, y) # loss
                
                
                if torch.isnan(loss).any():
                    print(f"Step {step_counter}: NaN detected in loss.")
                    print("loss", loss)
                    print("y_pred", y_pred)
                
                if torch.isnan(loss).any():
                    print("NaN detected in loss, saving model and stopping.")
                    # Save model weights before termination
                    torch.save(gp_model.state_dict(), 'model_weights_before_nan.pt')
                    engine.terminate()
                    return
                
                loss.backward()
                optimizer.step()
                
                return loss.item()
    
    
            def eval_step(engine, batch):
                gp_model.eval() # set to eval
                likelihood.eval()
                
                x, y = batch
                if torch.cuda.is_available():
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
    
                y_pred = gp_model(x)   
                return y_pred, y
    
                
            trainer = Engine(step)
            evaluator = Engine(eval_step)
    
            metric = Average()
            metric.attach(trainer, "loss")
            pbar.attach(trainer)
    
            metric = Loss(lambda y_pred, y: - likelihood.expected_log_prob(y, y_pred).mean())
    
            metric.attach(evaluator, "loss")
    
            @trainer.on(Events.EPOCH_COMPLETED(every=5)
            def log_results(trainer):
    
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
                evaluator.run(dl_val) # val dataset
                print(f"Results - Epoch: {trainer.state.epoch} - "
                    f"Val Loss: {evaluator.state.metrics['loss']:.2f} - "
                    f"Train Loss: {trainer.state.metrics['loss']:.2f}")
                
            print("Total model params: ")
            for index, param in enumerate(gp_model.parameters()): 
                # if (index==2):
                print(param.shape)
                
            ################## Run Training ##################
            print()
            print()
            
            # Before starting the training, record the start time and initial memory usage
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)  # Reset peak memory stats at the start of each run
            
            trainer.run(dl_train, max_epochs=epochs)
            print()
            print()
            
            # After training, calculate and print the total time and memory used
            total_time = time.time() - start_time
            all_training_time.append(total_time)
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert bytes to MB
                all_peak_memory.append(peak_memory)
                print(f"Run {run} - Total training time: {total_time:.2f} seconds, Peak GPU memory used: {peak_memory:.2f} MB")
            else:
                print(f"Run {run} - Total training time: {total_time:.2f} seconds")
    
            # Optional: Clear some memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear unused memory
                
            print()
            print()
            
    
    
            
            ################## Testing ##################
            
            # Assuming you have a function to compute RMSE, or you're using Ignite's RMSE metric
    
            def eval_step(engine, batch):
                gp_model.eval()  # Ensure model is in evaluation mode
                likelihood.eval()
                
                x, y = batch
                if torch.cuda.is_available():
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
    
                # Assuming your model outputs a distribution, e.g., MultivariateNormal
                with torch.no_grad():  # Disable gradient computation for evaluation
                    distribution = gp_model(x)
                    y_pred = distribution.mean  # Use the mean of the distribution as the prediction
    
                return y_pred, y
    
            # Update the evaluator engine
            evaluator = Engine(eval_step)
    
            # Attach the RMSE metric to the evaluator
            rmse = RootMeanSquaredError()
            rmse.attach(evaluator, "RMSE")
    
            # After training, run the evaluator on the test dataset to compute the RMSE
            evaluator.run(dl_test)
    
            # Retrieve and display the RMSE
            test_rmse = evaluator.state.metrics['RMSE']
            all_test_rmses.append(test_rmse)
            print(f"Test RMSE: {test_rmse:.2f}")
            print()
            print()
    
            del x_train, y_train, x_val, y_val, x_test, y_test
            del x_train_real, y_train_real, x_train_real_normalized, x_val_normalized, x_test_normalized
            del ds_train, ds_val, ds_test, dl_train, dl_val, dl_test
            del pc, gp_model
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        
        # Test RMSE
        mean_test_rmse = statistics.mean(all_test_rmses)
        std_test_rmse = statistics.stdev(all_test_rmses)
        all_search_test_rmse_means.append(mean_test_rmse)
        all_search_test_rmse_stds.append(std_test_rmse)
        print("Mean test RMSE:", mean_test_rmse)
        print("STD test RMSE:", std_test_rmse)
        print()
        print()
        
        # Training Time
        mean_training_time = statistics.mean(all_training_time)
        std_training_time = statistics.stdev(all_training_time)
        all_search_training_time_means.append(mean_training_time)
        all_search_training_time_stds.append(std_training_time)
        print("Mean training time:", mean_training_time)
        print("STD training time:", std_training_time)
        
        print()
        print()
        
        # Training Mem
        if all_peak_memory:
            mean_peak_mem = statistics.mean(all_peak_memory)
            std_peak_mem = statistics.stdev(all_peak_memory)
            print("Mean peak memory:", mean_peak_mem)
            print("STD peak memory:", std_peak_mem)

    print()
    print("all_search_test_rmse_means", all_search_test_rmse_means)
    print()
    print("all_search_test_rmse_stds", all_search_test_rmse_stds)
    print()
    print("all_search_training_time_means", all_search_training_time_means)
    print()
    print("all_search_training_time_stds", all_search_training_time_stds)
        
    
    print()
    print("Terminated Successfully.")

if __name__ == "__main__":
    main()
