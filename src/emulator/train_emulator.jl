using Flux, Optimisers
using Statistics
using ProgressMeter



"""
    train_emulator(nn::NeuralNetwork, fd::FormattedData; sim_para::SimPara=fd.sim_para, 
                   batchsize::Int=32, n_epochs::Int=300, η0::Real=0.001)

Train an emulator (neural network) with the given architecture and data.

# Description
- Computes Z-score parameters from the training set.
- Constructs an `Emulator` with the given `NeuralNetwork`.
- Applies Z-score normalization to training and validation pairs.
- Trains the network using Adam with initial learning rate η0.
- Halves the learning rate every 30 epochs.
- Logs training and validation losses.
- Calls `compare_emulator` on the test set after training.

# Arguments
- `nn::NeuralNetwork`: Defines the structure (layer sizes) of the neural network.
- `fd::FormattedData`: Formatted dataset with train/valid/test splits of size (2*n_coeff, N).
- `sim_para::SimPara=fd.sim_para`: Optional different simulation parameters for better saving.
- `batchsize::Int=32`: Training batch size.
- `n_epochs::Int=300`: Number of training epochs.
- `η0::Real=0.001`: Initial learning rate.

# Returns
- `em::Emulator`: Trained emulator model (Flux chain + normalization).
- `losses::Losses`: Recorded training/validation losses and batches per epoch.

# Notes
- Test set evaluation is not part of the training loop; only `compare_emulator` is called at the end.
- Normalization statistics are always computed from the training set to avoid leakage.

# Examples
```julia
nn = NeuralNetwork(io_dim=54, hidden_dim=128, n_hidden=2)
fd = FormattedData(sim_data; splits=(train=0.7, valid=0.2, test=0.1))
em, losses = train_emulator(nn, fd; batchsize=64, n_epochs=100, η0=0.0005)
```
"""
function train_emulator(nn::NeuralNetwork, fd::FormattedData; sim_para::SimPara=fd.sim_para, 
        batchsize::Int = 32, n_epochs::Int=300, η0::Real=0.001)

    # Calculating parameters for Z-score trafo
    μ = Float32.(vec(mean(fd.data_pairs.x_train; dims=2)))            
    σ = Float32.(vec(std(fd.data_pairs.x_train; dims=2)))
    zscore_para = ZscorePara(μ, σ)

    # Defining the emulator
    em = Emulator(nn, zscore_para, sim_para)

    # Transforming the training and validation data
    x_train_norm = zscore(fd.data_pairs.x_train, zscore_para)
    y_train_norm = zscore(fd.data_pairs.y_train, zscore_para)

    x_valid_norm = zscore(fd.data_pairs.x_valid, zscore_para)
    y_valid_norm = zscore(fd.data_pairs.y_valid, zscore_para)

    # Loading the data in batches
    loader_train = Flux.DataLoader((x_train_norm, y_train_norm),
                                   batchsize=batchsize, shuffle=true)
    loader_valid = Flux.DataLoader((x_valid_norm, y_valid_norm),
                                   batchsize=batchsize, shuffle=true)

    # Implementing optimiser
    η = η0
    opt = Optimisers.Adam(η)
    opt_state = Optimisers.setup(opt, em.chain)

    # Defining the losses
    losses0 = Losses(sim_para, length(loader_train), length(loader_valid))

    # Training loop   
    train_time = @elapsed begin
        @showprogress for epoch in 1:n_epochs
            # Training loop
            for xy_cpu in loader_train                                  # loop over every batches of loader_train
                x,y = xy_cpu |> gpu                                     # shifting the training data to the GPU
                loss, grads = Flux.withgradient(em.chain) do chain      # calculating training losses and gradients
                    Flux.mse(chain(x), y)                               
                end

                Optimisers.update!(opt_state, em.chain, grads[1])       # update the Optimiser
                push!(losses0.train, Float32(loss))                      # store the training losses
            end

            # Adjust the learning rate every 30 epochs
            if (epoch %30) == 0                                             
                η /= 2
                Optimisers.adjust!(opt_state, η)                        # adjust the optimiser to the new learning rate
            end

            # Validation loop
            for (x,y) in loader_valid                                   # loop over every batches of loader_valid
                loss = Flux.mse(em.chain(x), y)                         # calculating validation losses
                push!(losses0.valid, Float32(loss))                      # store the validation losses
            end
        end
    end

    # Reinitialize Losses and emulator
    losses = Losses(losses0.sim_para,    
                    losses0.train,
                    losses0.valid,                
                    losses0.test,
                    losses0.bpe_train,
                    losses0.bpe_valid,
                    losses0.bpe_test,
                    train_time)


    # Compares the emulator with the training set for a single timestep.
    compare_emulator(em, x_test=fd.data_pairs.x_test, y_test=fd.data_pairs.y_test)

    return em, losses
end


