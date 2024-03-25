import random
import time
import numpy as np
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from uci_datasets import Dataset

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RootMeanSquaredError

from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.region_graph.fully_factorized import FullyFactorized
from cirkit.models.gp import CircuitGP, initial_values
from cirkit.layers.sum_product.cp_w_bias import CPLayerWithBias
from cirkit.layers.input.rbf_network_kernel import RBFNetworkKernelLayer
from cirkit.reparams.leaf import ReparamExp, ReparamLogSoftmax, ReparamSoftmax, ReparamIdentity
from cirkit.models.tensorized_circuit import TensorizedPC

class IdentityMapping(nn.Module):
    def __init__(self):
        super(IdentityMapping, self).__init__()
    
    def forward(self, x):
        return x
    
step_counter = 0

################## Hyper Param ##################

dataset = "protein"
lr = 1e-3
batch_size = 32
epochs = 50
# n_inducing_points = 50
# kernel = "HBF" # not used

num_vars = 9
num_mixtures_s = [16, 32, 64, 96, 128]
depth = 3
num_repetitions = 10
# region_graph = FullyFactorized(num_vars=num_vars)
region_graph = RandomBinaryTree(num_vars=num_vars, depth=depth, num_repetitions=num_repetitions)
efamily_cls = RBFNetworkKernelLayer   # Flatten
layer_cls = CPLayerWithBias
reparam = ReparamIdentity


def main():
    
    all_search_test_rmse_means = []
    all_search_test_rmse_stds = []
    all_search_training_time_means = []
    all_search_training_time_stds = []
    
    for num_mixtures in num_mixtures_s:
    
        print("num_mixtures", num_mixtures)
    
        all_test_rmses = []
        all_training_time = []
        all_peak_memory = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        print("Device: ", device)
        # The device to use, e.g., "cpu", "cuda", "cuda:1"
        
        for run in range(1, 6):  # Running the model 5 times
            
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
            # seed = 42
            seed = run * 10  # Example: 10, 20, 30 for the three runs
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed) 
                
            
            ################## Circuit ##################

            pc_rbfn = TensorizedPC.from_region_graph(
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
            pc_rbfn.to(device)
            print(pc_rbfn)

            print("Circuit parameters: ")
            for param in pc_rbfn.parameters(): 
                print (param.shape)
                
            total_params = sum(p.numel() for p in pc_rbfn.parameters() if p.requires_grad)
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
            # num_outputs = 1
            # feature_extractor = IdentityMapping()

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
            
            ################## Training ##################


            # Loss function
            criterion = nn.MSELoss()

            # Optimizer
            optimizer = optim.Adam(pc_rbfn.parameters(), lr=lr)
            
            
            print()
            print()
            
            # Before starting the training, record the start time and initial memory usage
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)  # Reset peak memory stats at the start of each run


            # Training and Validation Loop
            for epoch in range(epochs):
                # Training phase
                pc_rbfn.train()
                train_loss = 0.0
                
                for inputs, targets in dl_train:
                    optimizer.zero_grad()
                    outputs = pc_rbfn(inputs.to(device))
                    outputs = outputs.squeeze(1)  # Ensure outputs match the target's shape
                    
                    # if(torch.isnan(outputs).any() == False):
                    #     print("no NAN")

                    loss = criterion(outputs, targets.to(device))
                    if(torch.isnan(loss).any()):
                        print("loss", loss)
                        
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    
                train_loss /= len(dl_train.dataset)

                # Validation phas
                # if (True):
                if (epoch % 10 == 0):
                    pc_rbfn.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in dl_val:
                            outputs = pc_rbfn(inputs.to(device))
                            outputs = outputs.squeeze(1)  # Ensure outputs match the target's shape
                            loss = criterion(outputs, targets.to(device))
                            val_loss += loss.item() * inputs.size(0)
                    val_loss /= len(dl_val.dataset)

                    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
                
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
            
            # Testing
            test_loss = 0.0
            pc_rbfn.eval()  # Ensure model is in evaluation mode
            with torch.no_grad():  # No gradients needed
                for inputs, targets in dl_test:
                    outputs = pc_rbfn(inputs.to(device)).squeeze(1)
                    loss = criterion(outputs, targets.to(device))
                    test_loss += loss.item() * inputs.size(0)
            test_loss /= len(dl_test.dataset)
            rmse = np.sqrt(test_loss)  # Calculate RMSE
            print(f"Test RMSE: {rmse:.4f}")
            all_test_rmses.append(rmse)
            print()
            print()
            
            del x_train, y_train, x_val, y_val, x_test, y_test
            del x_train_real, y_train_real, x_train_real_normalized, x_val_normalized, x_test_normalized
            del ds_train, ds_val, ds_test, dl_train, dl_val, dl_test
            del pc_rbfn
            
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
