import random
import time
import numpy as np
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("..")
from hbf_layer import HBFLayer
from uci_datasets import Dataset

    
# euclidean norm
def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)


# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()
    
step_counter = 0

################## Hyper Param ##################

dataset = "protein"
lr = 1e-3
batch_size = 32 
epochs = 50
# n_inducing_points = 50
# kernel = "HBF" # not used

num_vars = 9
# num_mixtures = 10 # 1004 3868 15172
num_mixtures_s = [953, 3766, 14969, 33610, 59688]



def main():
    
    all_search_test_rmse_means = []
    all_search_test_rmse_stds = []
    all_search_training_time_means = []
    all_search_training_time_stds = []
    
    for num_mixtures in num_mixtures_s:
    
        all_test_rmses = []
        all_training_time = []
        all_peak_memory = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        print("Device: ", device)
        # The device to use, e.g., "cpu", "cuda", "cuda:1"
        
        for run in range(1, 6):  # Running the model 5 times
            
            dataset_split = random.randint(0, 9)    ############################

            ################## Random ##################
            
            print()
            print()
            print(f"Run {run}")
            print()
            print()

            # Update seed settings for each run
            # seed = 42    ############################
            seed = run * 10  # Example: 10, 20, 30 for the three runs
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed) 
                
            
            ################## Circuit ##################

            rbf = HBFLayer(in_features_dim=num_vars,
                num_kernels=num_mixtures,
                out_features_dim=1,
                radial_function=rbf_gaussian,
                norm_function=euclidean_norm,
                normalization=False)

            print("RBFN parameters: ")
            for param in rbf.parameters():
                print(param.shape)
                
            total_params = sum(p.numel() for p in rbf.parameters() if p.requires_grad)
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
            optimizer = optim.Adam(rbf.parameters(), lr=lr)
            
            
            print()
            print()
            
            # Before starting the training, record the start time and initial memory usage
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)  # Reset peak memory stats at the start of each run


            # Training and Validation Loop
            for epoch in range(epochs):
                # Training phase
                rbf.train()
                train_loss = 0.0
                
                for inputs, targets in dl_train:
                    optimizer.zero_grad()
                    outputs = rbf(inputs)
                    outputs = outputs.squeeze(1)  # Ensure outputs match the target's shape
                    
                    # if(torch.isnan(outputs).any() == False):
                    #     print("no NAN")

                    loss = criterion(outputs, targets)
                    if(torch.isnan(loss).any()):
                        print("loss", loss)
                        
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    
                train_loss /= len(dl_train.dataset)

                # Validation phas
                if (epoch % 10 == 0):
                # if (True):    ############################
                    rbf.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, targets in dl_val:
                            outputs = rbf(inputs)
                            outputs = outputs.squeeze(1)  # Ensure outputs match the target's shape
                            loss = criterion(outputs, targets)
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
            rbf.eval()  # Ensure model is in evaluation mode
            with torch.no_grad():  # No gradients needed
                for inputs, targets in dl_test:
                    outputs = rbf(inputs).squeeze(1)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * inputs.size(0)
            test_loss /= len(dl_test.dataset)
            rmse = np.sqrt(test_loss)  # Calculate RMSE
            print(f"Test RMSE: {rmse:.4f}")
            all_test_rmses.append(rmse)
            print()
            print()

        
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
