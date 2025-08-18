module TrainModel

using JLD2, Plots
using Flux, Optimisers, Statistics, ProgressMeter
using CUDA

using ..BasicStructs
using ..ZscoreTrafo
using ..ModelStructs
using ..DataFormatting


export train_model


"""
    train_model(nn::NeuralNetwork, fd::FormattedData; 
                batchsize::Int = 32, n_epochs::Int = 300, η0::Real = 0.001)  

Defines the model with parameters `nn` and trains it with `fd`.

Defines (according to nn) and trains a emulator (according to fd). Some hyperparameters are listed as arguments.
train_model does:
    - Z-score transform the data from `fd`
    - Implement a Adam optimizer
    - Train the emulator 
IMPORTANT: Until now, there is no implementation of testing (using test-set) the emulator!

# Arguments
- `nn::NeuralNetwork`:      Defines the structure of the neural network.
- `fd::FormattedData`:      Used training-, validation- and test- data.
- `batchsize::Int = 32`:    Batchsize of the training data.
- `n_epochs::Int = 300`:    Number of epochs.
- `η0::Real = 0.001`:       Initial value of the learning rate.

# Returns
- `tm::TrainedModel`:       The "ready to use" trained model.
- `losses::Losses`:         Losses occured at training.
"""
function train_model(nn::NeuralNetwork, fd::FormattedData; 
        batchsize::Int = 32, n_epochs::Int=300, η0::Real=0.001)

    # Calculating parameters for Z-score trafo
    μ = Float32.(vec(mean(fd.data_pairs.x_train; dims=2)))            
    σ = Float32.(vec(std(fd.data_pairs.x_train; dims=2)))
    zscore_para = ZscorePara(μ, σ)

    # Defining the emulator
    tm = TrainedModel(nn, zscore_para, fd.sim_para)

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
    opt_state = Optimisers.setup(opt, tm.chain)

    # Defining the losses
    losses = Losses(length(loader_train), length(loader_valid), fd.sim_para)


    # Training loop      
    @showprogress for epoch in 1:n_epochs
        # Training loop
        for xy_cpu in loader_train                                  # loop over every batches of loader_train
            x,y = xy_cpu |> gpu                                     # shifting the training data to the GPU
            loss, grads = Flux.withgradient(tm.chain) do chain      # calculating training losses and gradients
                Flux.mse(chain(x), y)                               
            end

            Optimisers.update!(opt_state, tm.chain, grads[1])       # update the Optimiser
            push!(losses.train, Float32(loss))                      # store the training losses
        end

        # Adjust the learning rate every 30 epochs
        if (epoch %30) == 0                                             
            η /= 2
            Optimisers.adjust!(opt_state, η)                        # adjust the optimiser to the new learning rate
        end

        # Validation loop
        for (x,y) in loader_valid                                   # loop over every batches of loader_valid
            loss = Flux.mse(tm.chain(x), y)                         # calculating validation losses
            push!(losses.valid, Float32(loss))                      # store the validation losses
        end
    end

    return tm, losses
end



end


